import argparse
from dotenv import load_dotenv
import json
import os
import wandb
import yaml
from pathlib import Path
from pprint import pprint

from experiments.composition import CompositionExperiment
from models.black_box_model import GPT4_Turbo
from string_transformations.string_transformations import AlternatingCase, AtbashCipher, BaseN, Binary, Id, Leetspeak, LanguageTranslation, MorseCode, Reversal, VowelRepetition
from utils.modal_utils import app, cheap_modal_wrap_experiment


load_dotenv()


CFG_PATH = Path(__file__).resolve().parents[0] / "configs/model_params.yaml"

with open(CFG_PATH) as cfg_file:
    CFG = yaml.safe_load(cfg_file)

def run(val_or_eval_or_llamaguard):
    target_model = GPT4_Turbo()
    num_attack_trials = 10

    # hyperparameter_grid = {
    #     "k_num_transforms": [1, 2, 3, 4],
    #     "maybe_transformation_instructions": [True, False],
    #     "other_transform": [Id, LanguageTranslation, Leetspeak],
    #     "composition_target": ["query", "response"],
    # } # this grid is ~50M tokens = $1500?

    hyperparameter_grid = {
        "k_num_transforms": [1, 2, 3, 4],
        "maybe_transformation_instructions": [True],
        "other_transform": [Id, Leetspeak, MorseCode, Binary, BaseN],
        "composition_target": ["response"],
    } # this grid is ~10M tokens = $300?
    response_composition_experiment = CompositionExperiment(
        target_model=target_model,
        num_attack_trials=num_attack_trials,
        hyperparameter_grid=hyperparameter_grid,
    )
    eval_config = {'val_or_eval_or_llamaguard': val_or_eval_or_llamaguard}
    response_composition_experiment.run_hyperparameter_grid(config=eval_config)

    hyperparameter_grid = {
        "k_num_transforms": [1, 2, 3, 4],
        "maybe_transformation_instructions": [True],
        "other_transform": [Id, Leetspeak, LanguageTranslation],
        "composition_target": ["query"],
    } # this grid is ~10M tokens = $300?
    query_composition_experiment = CompositionExperiment(
        target_model=target_model,
        num_attack_trials=num_attack_trials,
        hyperparameter_grid=hyperparameter_grid,
    )
    eval_config = {'val_or_eval_or_llamaguard': val_or_eval_or_llamaguard}
    query_composition_experiment.run_hyperparameter_grid(config=eval_config)

    # currently, len(bad_prompts) = 81 from harmbench val set
    if bool(os.environ.get("WANDB_API_KEY")):
        wandb.finish()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run composition experiments with optional evaluation mode.")
    parser.add_argument("--evalu", type=str, default="val", help="Specify the evaluation mode. 'val' = harmbench val classifier. 'eval' = harmbench official classifier. 'llamaguard' = Haize-Mistral with LlamaGuard prompt.")
    return parser.parse_args()


@app.local_entrypoint()
def main():
    cheap_modal_wrap_experiment.remote(run)

if __name__ == "__main__":
    args = parse_arguments()
    run(args.evalu)