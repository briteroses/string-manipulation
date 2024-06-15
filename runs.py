import json
import os
import wandb
import yaml
from pathlib import Path
from pprint import pprint

from experiments.composition import CompositionExperiment, load_safety_data
from models.black_box_model import GPT4_Turbo
from string_transformations.string_transformations import LANGUAGE_TRANSLATION_OPTIONS, AlternatingCase, AtbashCipher, Binary, Id, Leetspeak, HaizeyLanguageTranslation, MorseCode, Reversal, VowelRepetition
from utils.modal_utils import app, cheap_modal_wrap_experiment


CFG_PATH = Path(__file__).resolve().parents[0] / "configs/model_params.yaml"

with open(CFG_PATH) as cfg_file:
    CFG = yaml.safe_load(cfg_file)

if __name__ == "__main__":
    eval_set = load_safety_data()
    for (behavior, context) in eval_set:
        prompt = behavior
        prompt += ("\n"+context) if context is not None else ""
        # Perform the evaluation or any other required operation on the prompt
        for language in LANGUAGE_TRANSLATION_OPTIONS:
            print(f"\n\n\n~~~~~{language}~~~~~")
            translation = HaizeyLanguageTranslation.construct(choice=language)
            output = translation(prompt)
            original = translation.invert(output)
            print(output)
            print("\n")
            print(original)
