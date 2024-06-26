from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from typing import List
from experiments.base_experiment import BaseExperiment
import modal
from pathlib import Path
import random
from tqdm import tqdm
from models.black_box_model import GPTFamily
from models.open_source_model import OpenSourceModel
from utils.utils import ALL_ICL_EXEMPLARS, get_output_length_for_model, get_greedy_one_command, get_max_tokens_key
import wandb
import yaml

from experiments.composition import _CHARACTER_LEVEL_TRANSFORMS, CompositionPrompts, HarmBenchPrompt, INCOHERENCY, load_safety_data, QUERY_COMPOSITION_PROMPT, RESPONSE_COMPOSITION_PROMPT
from judging.harmbench_judge import HarmBenchJudge
from models.model_class import LanguageModel
from string_transformations.string_transformations import _EXTENDERS, _STYLES, ALL_TRANSFORMATIONS, AlternatingCase, AtbashCipher, BaseN, Binary, CaesarCipher, Delimiters, Id, JSON_Encapsulation, LanguageTranslation, LatexMode, Leetspeak, MorseCode, Palindrome, PerWordReversal, PrefixRotation, PythonMarkdown, ROT13Cipher, Reversal, StringTransformation, TokenizerAwareTransformation, VowelRepetition, WordLevelReversal, sample_transformations


CFG_PATH = Path(__file__).resolve().parents[1] / "configs/model_params.yaml"

with open(CFG_PATH) as cfg_file:
    CFG = yaml.safe_load(cfg_file)

PARENT_PATH = Path(__file__).resolve().parents[0]
if modal.is_local():
    MOUNT_PATH = PARENT_PATH
else:
    MOUNT_PATH = Path("/data")

EXPERIMENT_DATA_FOLDER = MOUNT_PATH / "experiment_data"


AdaptiveSamplingParams = namedtuple("AdaptiveSamplingParams", "composition_target num_transforms")

ADAPTIVE_SAMPLING_OPTIONS = [
    AdaptiveSamplingParams("query", 1),
    AdaptiveSamplingParams("query", 2),
    AdaptiveSamplingParams("response", 1),
    AdaptiveSamplingParams("response", 2),
    AdaptiveSamplingParams("response", 3),
]

QUERY_ALLOWED_TRANSFORMS = [Reversal, PerWordReversal, WordLevelReversal, CaesarCipher, ROT13Cipher, BaseN, Binary, Leetspeak, MorseCode, VowelRepetition, AlternatingCase, Palindrome, Delimiters, PrefixRotation]
RESPONSE_ALLOWED_TRANSFORMS = [PerWordReversal, CaesarCipher, ROT13Cipher, BaseN, Binary, Leetspeak, MorseCode, VowelRepetition, AlternatingCase, Palindrome, Delimiters, PythonMarkdown, JSON_Encapsulation, LatexMode]

def adaptive_default_hparams():
    return {
        "attack_budget": [25,],
    }
@dataclass
class AdaptiveAttackExperiment(BaseExperiment):
    target_model: LanguageModel

    hyperparameter_grid: dict = field(default_factory=adaptive_default_hparams)

    all_parameters = ["target_model", "attack_budget"]

    name = "adaptive_attack"
    description = "Experiment sampling compositions for multiple rounds per unsafe query, as an adaptive attack."

    def __post_init__(self):
        if bool(os.environ.get("WANDB_API_KEY")):
            exp_time = datetime.now().strftime('%m/%d/%Y-%H:%M:%S')
            wandb.init(
                project="string-transformation",
                entity="crtsn",
                config={
                    "target_model": self.target_model.name,
                    "hparam_grid": self.hyperparameter_grid,
                },
                name=f"{exp_time}-string-transformation",
            )
        for param_name in self.all_parameters:
            assert hasattr(self, param_name) or param_name in self.hyperparameter_grid
        super().__post_init__()

    def run(self, hyperparam_setting, config):
        val_or_eval_or_llamaguard = config["val_or_eval_or_llamaguard"]
        val_or_eval = "val" if val_or_eval_or_llamaguard == "llamaguard" else val_or_eval_or_llamaguard
        safety_dataset = load_safety_data(val_or_eval)
        judge = HarmBenchJudge(val_or_eval)

        attack_budget = hyperparam_setting["attack_budget"]

        load_path = self.data_save_path(**hyperparam_setting)
        if os.path.exists(load_path):
            print("WARNING: if results exist for a different dataset than the current run, then this run may have errors -- clean up your files!")
            with open(load_path, 'r') as fin:
                experiment_data = json.load(fin)
                progress = len(experiment_data["data"])
                print(f"{progress} of {len(safety_dataset)} samples already finished for this hyperparameter setting. Continuing...")
                bad_prompt_data = experiment_data["data"]
                raw_output_data = experiment_data["raw_outputs"]
                num_successful_attacks = sum(datum["successful"] for datum in bad_prompt_data)
        else:
            experiment_data = {}
            experiment_header = self._raw_data_path(**hyperparam_setting)
            experiment_data["HEADER"] = experiment_header
            progress = 0
            bad_prompt_data = []
            raw_output_data = []
            num_successful_attacks = 0

        for i in tqdm(range(len(safety_dataset)), desc="bad prompt"):
            if i < progress:
                if experiment_data["data"][i]["successful"]:
                    print(f"Attack on query \"{experiment_data['data'][i]['behavior']}\" already succeeded; onto the next one...")
                    continue
                single_attack_progress = len(experiment_data["raw_outputs"][i]["continuations"])
                if single_attack_progress >= attack_budget:
                    print(f"Attack budget on query \"{experiment_data['data'][i]['behavior']}\" already exhausted; onto the next one...")
                    continue
                continuations = experiment_data["raw_outputs"][i]["continuations"]
            else:
                single_attack_progress = 0
                continuations = []

            bad_prompt = safety_dataset[i]
            if isinstance(bad_prompt, HarmBenchPrompt):
                bad_prompt_behavior, bad_prompt_context = bad_prompt.behavior, bad_prompt.context
                bad_prompt_as_input = bad_prompt_behavior + (f'\n{bad_prompt_context}' if bad_prompt_context is not None else "")
            elif isinstance(bad_prompt, str):
                bad_prompt_behavior, bad_prompt_context = bad_prompt, None
                bad_prompt_as_input = bad_prompt_behavior

            successful = False

            for trial in tqdm(range(single_attack_progress, attack_budget), desc="attack trial", leave=False):

                adaptive_sampling_params = random.choice(ADAPTIVE_SAMPLING_OPTIONS)
                if adaptive_sampling_params.composition_target == "response":
                    composition_prompt = RESPONSE_COMPOSITION_PROMPT
                    allowed_composition_transforms = RESPONSE_ALLOWED_TRANSFORMS
                    allowed_other_transforms = QUERY_ALLOWED_TRANSFORMS
                elif adaptive_sampling_params.composition_target == "query":
                    composition_prompt = QUERY_COMPOSITION_PROMPT
                    allowed_composition_transforms = QUERY_ALLOWED_TRANSFORMS
                    allowed_other_transforms = RESPONSE_ALLOWED_TRANSFORMS

                other_transform = self._construct_transform(random.choice(allowed_other_transforms))
                
                transform_list = []
                transform_classes = sample_transformations(
                    adaptive_sampling_params.num_transforms,
                    allowed_transforms=allowed_composition_transforms
                )
                for transform_class in transform_classes:
                    transform = self._construct_transform(transform_class)
                    transform_list.append(transform)
                list_transformations_prompt = CompositionPrompts.list_transformations(transform_list)
                transformation_instructions_prompt = CompositionPrompts.transformation_instructions(transform_list)

                icl_exemplars = random.sample(ALL_ICL_EXEMPLARS, 2) #TODO 2 is actually a hyperparameter!
                icl_demonstrations_prompt = CompositionPrompts.icl_demonstrations(
                    transform_list,
                    icl_exemplars=icl_exemplars,
                    other_transform=other_transform,
                    composition_target=adaptive_sampling_params.composition_target)
                
                transformed_bad_prompt = bad_prompt_as_input
                if adaptive_sampling_params.composition_target == "query":
                    for transform in transform_list:
                        transformed_bad_prompt = transform(transformed_bad_prompt)
                elif adaptive_sampling_params.composition_target == "response":
                    transformed_bad_prompt = other_transform(transformed_bad_prompt)
                
                attack_prompt = composition_prompt.substitute(
                    LIST_TRANSFORMATIONS=list_transformations_prompt,
                    TRANSFORMATION_INSTRUCTIONS=transformation_instructions_prompt,
                    ICL_DEMONSTRATIONS=icl_demonstrations_prompt,
                    bad_prompt=transformed_bad_prompt,
                )

                greedy_one = get_greedy_one_command(self.target_model)
                if adaptive_sampling_params.composition_target == "response":
                    num_toks = self.choose_max_tokens(transform_list)
                elif adaptive_sampling_params.composition_target == "query":
                    num_toks = self.choose_max_tokens(other_transform)
                greedy_one[get_max_tokens_key(self.target_model)] = num_toks

                continuation = self.target_model.inference(attack_prompt, **greedy_one)[0]

                try:
                    if adaptive_sampling_params.composition_target == "response":
                        for transform in reversed(transform_list):
                            continuation = transform.invert(continuation)
                    elif adaptive_sampling_params.composition_target == "query":
                        continuation = other_transform.invert(continuation)
                except:
                    continuation = INCOHERENCY

                if adaptive_sampling_params.composition_target == "response":
                    query_transform_printout = str(other_transform)
                    response_transform_printout = list(map(str, transform_list))
                elif adaptive_sampling_params.composition_target == "query":
                    query_transform_printout = list(map(str, transform_list))
                    response_transform_printout = str(other_transform)

                continuations.append({
                    "query": query_transform_printout,
                    "response": response_transform_printout,
                    "continuation": continuation,
                })
                
                single_asr = judge(bad_prompt_behavior, continuation, bad_prompt_context)

                if single_asr == 100:
                    successful = True
                    num_successful_attacks += 1
                    break

            tries_until_success = trial + 1 if successful else None
            bad_prompt_datum = {
                "behavior": bad_prompt_behavior,
                "context": bad_prompt_context,
                "successful": successful,
                "tries_until_success": tries_until_success,
            }
            raw_output_datum = {
                "behavior": bad_prompt_behavior,
                "context": bad_prompt_context,
                "continuations": continuations,
            }
            if i < progress:
                bad_prompt_data[i] = bad_prompt_datum
                raw_output_data[i] = raw_output_datum
            else:
                bad_prompt_data.append(bad_prompt_datum)
                raw_output_data.append(raw_output_datum)
            experiment_data["data"] = bad_prompt_data
            experiment_data["raw_outputs"] = raw_output_data

            save_path = self.data_save_path(**hyperparam_setting)
            with open(save_path, 'w') as fout:
                json.dump(experiment_data, fout)
        
        asr = num_successful_attacks / len(safety_dataset)

        experiment_data["asr"] = asr

        save_path = self.data_save_path(**hyperparam_setting)
        with open(save_path, 'w') as fout:
            json.dump(experiment_data, fout)
        
        return experiment_data
    
    def evaluate_hyperparameter_grid(self, config=None):
        raise ValueError("No evaluation to be done... the run is the evaluation.")

    def evaluate(self, **kwargs):
        raise ValueError("No evaluation to be done... the run is the evaluation.")

    def data_save_path(self, **hyperparam_setting):
        hyperparam_data_path = self.data_path(**hyperparam_setting)
        full_save_path = EXPERIMENT_DATA_FOLDER / self.name / f"{hyperparam_data_path}.json"
        full_save_path.parent.mkdir(parents=True, exist_ok=True)
        return full_save_path

    def _construct_transform(self, transform_class: StringTransformation):
        if issubclass(transform_class, TokenizerAwareTransformation):
            if isinstance(self.target_model, GPTFamily):
                transform = transform_class.construct(openai_model_name=self.target_model.name)
            elif isinstance(self.target_model, OpenSourceModel):
                transform = transform_class.construct(open_source_model_name=self.target_model.name)
        elif issubclass(transform_class, (PythonMarkdown, LatexMode)):
            transform = transform_class.construct(model_type=self.target_model.__class__)
        else:
            transform = transform_class.construct()
        return transform
    
    def choose_max_tokens(self, transform_s: StringTransformation | List[StringTransformation]):
        """
        Most transformations cause the model to use shorter tokens than usual;
        i.e. complicated character-level transforms mean each token is ~ 1 character rather than ~ 1 word.
        We call this to compensate.
        """
        def issubclass_option(t_s, class_tup):
            if isinstance(t_s, List):
                return any(isinstance(transform, class_tup) for transform in t_s)
            elif isinstance(t_s, StringTransformation):
                return isinstance(t_s, class_tup)
        num_toks = CFG["max_tokens"]
        if issubclass_option(transform_s, _CHARACTER_LEVEL_TRANSFORMS):
            num_toks *= 3
        if issubclass_option(transform_s, _EXTENDERS):
            num_toks *= 6
        if issubclass_option(transform_s, _STYLES):
            num_toks += 200
        num_toks = int(num_toks)
        num_toks = min(num_toks, get_output_length_for_model(self.target_model))
        return num_toks


def baseline_empty_hparams():
    return {
        "placeholder": 0,
    }
@dataclass
class BaselineNoAttackExperiment(BaseExperiment):
    target_model: LanguageModel

    hyperparameter_grid: dict = field(default_factory=baseline_empty_hparams) # singleton & unused hparam grid to fit the experiment class specification

    all_parameters = ["target_model", "placeholder"]

    name = "baseline_no_attack"
    description = "What's the baseline ASR if we just feed the raw queries to the target model?"

    def run(self, hyperparam_setting, _):
        safety_dataset = load_safety_data("eval")
        judge = HarmBenchJudge("eval")

        load_path = self.data_save_path(**hyperparam_setting)
        if os.path.exists(load_path):
            print("WARNING: if results exist for a different dataset than the current run, then this run may have errors -- clean up your files!")
            with open(load_path, 'r') as fin:
                experiment_data = json.load(fin)
                progress = len(experiment_data["data"])
                if progress >= len(safety_dataset):
                    print(f"Run already finished for this hyperparameter setting.")
                    print(f"Skipping this run. Delete the file at {load_path} if you want to re-run.")
                    return
                else:
                    print(f"{progress} of {len(safety_dataset)} samples already finished for this hyperparameter setting. Continuing...")
                    bad_prompt_data = experiment_data["data"]
                    num_successful_attacks = sum(datum["successful"] for datum in bad_prompt_data)
        else:
            experiment_data = {}
            progress = 0
            bad_prompt_data = []
            num_successful_attacks = 0

        for i in tqdm(range(progress, len(safety_dataset)), desc="bad prompt"):
            bad_prompt = safety_dataset[i]
            if isinstance(bad_prompt, HarmBenchPrompt):
                bad_prompt_behavior, bad_prompt_context = bad_prompt.behavior, bad_prompt.context
                bad_prompt_as_input = bad_prompt_behavior + (f'\n{bad_prompt_context}' if bad_prompt_context is not None else "")
            elif isinstance(bad_prompt, str):
                bad_prompt_behavior, bad_prompt_context = bad_prompt, None
                bad_prompt_as_input = bad_prompt_behavior
            
            greedy_one = get_greedy_one_command(self.target_model)
            greedy_one[get_max_tokens_key(self.target_model)] = 512

            continuation = self.target_model.inference(bad_prompt_as_input, **greedy_one)[0]

            single_asr = judge(bad_prompt_behavior, continuation, bad_prompt_context)

            successful = False
            if single_asr == 100:
                successful = True
                num_successful_attacks += 1

            bad_prompt_data.append({
                "behavior": bad_prompt_behavior,
                "context": bad_prompt_context,
                "successful": successful,
                "continuation": continuation,
            })
            experiment_data["data"] = bad_prompt_data
            save_path = self.data_save_path(**hyperparam_setting)
            with open(save_path, 'w') as fout:
                json.dump(experiment_data, fout)
        
        asr = num_successful_attacks / len(safety_dataset)

        experiment_data["asr"] = asr

        save_path = self.data_save_path(**hyperparam_setting)
        with open(save_path, 'w') as fout:
            json.dump(experiment_data, fout)
        
        return experiment_data
    
    def data_save_path(self, **hyperparam_setting):
        hyperparam_data_path = self.data_path(**hyperparam_setting)
        full_save_path = EXPERIMENT_DATA_FOLDER / self.name / f"{hyperparam_data_path}.json"
        full_save_path.parent.mkdir(parents=True, exist_ok=True)
        return full_save_path

    def evaluate_hyperparameter_grid(self, config=None):
        raise ValueError("No evaluation to be done... the run is the evaluation.")

    def evaluate(self, **kwargs):
        raise ValueError("No evaluation to be done... the run is the evaluation.")