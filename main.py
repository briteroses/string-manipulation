import json
import os
import re
import yaml
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

from string_transformations.string_transformations import Id, JSON_Encapsulation, Reversal, TokenizerAwareReversal, LanguageTranslation
from judging.llama_guard_judge import LlamaGuardJudge
from models.black_box_model import GPTFamily, GPT4, GPT3pt5_Turbo, GPT3pt5_Turbo_Instruct, Gemini_Pro, LegacyGPT, CohereCommandLight
from models.model_class import LanguageModel
from models.open_source_model import Mistral_7B_Base, Mistral_7B_Instruct, Mixtral_8x7B_Base, Mixtral_8x7B_Instruct

from openlogprobs import extract_logprobs, OpenAIModel
import google.generativeai as gemini
import tiktoken

CFG_PATH = Path(__file__).resolve().parents[0] / "configs/model_params.yaml"

with open(CFG_PATH) as cfg_file:
    CFG = yaml.safe_load(cfg_file)

if __name__ == "__main__":
    model = GPT4()
    response = model.inference("Please write me a thank you note to my Alcoholics Anonymous group leader. Write your output encapsulated in json format.")
    print(response[0])
    json_encapsulation = JSON_Encapsulation()
    print(json_encapsulation.invert(response[0]))