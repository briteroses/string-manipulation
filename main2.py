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

def run(config):
    target_model = GPT4_Turbo()
    num_attack_trials = 15

    hyperparameter_grid = {
        "k_num_transforms": [1],
        "maybe_transformation_instructions": [True],
        "other_transform": [Reversal],
        "composition_target": ["response"],
    }
    response_composition_experiment = CompositionExperiment(
        target_model=target_model,
        num_attack_trials=num_attack_trials,
        hyperparameter_grid=hyperparameter_grid,
    )
    response_composition_experiment.run_hyperparameter_grid(config)

    # currently, len(bad_prompts) = 81 from harmbench val set
    if bool(os.environ.get("WANDB_API_KEY")):
        wandb.finish()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run composition experiments with optional evaluation mode.")
    parser.add_argument("--val_or_eval_or_llamaguard", type=str, default="val", help="Specify the evaluation mode. 'val' = harmbench val classifier. 'eval' = harmbench official classifier. 'llamaguard' = Haize-Mistral with LlamaGuard prompt.")
    parser.add_argument("--evaluate_during", type=bool, default=False, help="Evaluate each hyperparameter setting as part of its run?")
    return parser.parse_args()


@app.local_entrypoint()
def main():
    config = vars(parse_arguments())
    cheap_modal_wrap_experiment.remote(run, config)

if __name__ == "__main__":
    config = vars(parse_arguments())
    run(config)
