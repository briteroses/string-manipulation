import yaml
from pathlib import Path
from experiments.composition import CompositionExperiment

from string_transformations.string_transformations import AlternatingCase, Id, Leetspeak, LanguageTranslation, VowelRepetition
from models.black_box_model import GPT4_Turbo

from utils.modal_utils import app, cheap_modal_wrap_experiment

CFG_PATH = Path(__file__).resolve().parents[0] / "configs/model_params.yaml"

with open(CFG_PATH) as cfg_file:
    CFG = yaml.safe_load(cfg_file)

def run():
    target_model = GPT4_Turbo()
    num_attack_trials = 20

    # hyperparameter_grid = {
    #     "k_num_transforms": [1, 2, 3, 4],
    #     "maybe_transformation_instructions": [True, False],
    #     "other_transform": [Id, LanguageTranslation, Leetspeak],
    #     "composition_target": ["query", "response"],
    # } # this grid is ~50M tokens = $1500?

    hyperparameter_grid = {
        "k_num_transforms": [1, 2, 3, 4],
        "maybe_transformation_instructions": [True],
        "other_transform": [Id, LanguageTranslation, Leetspeak],
        "composition_target": ["query", "response"],
    } # this grid is ~10M tokens = $300?

    # currently, len(bad_prompts) = 81 from harmbench val set
    composition_experiment = CompositionExperiment(
        target_model=target_model,
        num_attack_trials=num_attack_trials,
        hyperparameter_grid=hyperparameter_grid,
    )
    composition_experiment.run_hyperparameter_grid()

@app.local_entrypoint()
def main():
    cheap_modal_wrap_experiment.local(run)