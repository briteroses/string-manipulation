from dataclasses import dataclass
from itertools import chain
import numpy as np
import os
from pathlib import Path
import time
from typing import Dict, List
import yaml

from models.model_class import LanguageModel

import cohere
import google.generativeai as gemini
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion as LegacyCompletion
from openlogprobs import extract_logprobs, OpenAIModel as OpenLogProbsOpenAIModel
import tiktoken

CFG_PATH = Path(__file__).resolve().parents[1] / "configs/model_params.yaml"

with open(CFG_PATH) as cfg_file:
    CFG = yaml.safe_load(cfg_file)


@dataclass
class BlackBoxModel(LanguageModel):
    """
    Denotes any model which is closed-source and is only accessed via API for text outputs.
    """
    def __post_init__(self):
        super().__post_init__()
    
    def destroy(self):
        pass #nothing to destroy


@dataclass
class GPTFamily(BlackBoxModel):
    """
    Denotes any OpenAI model.
    """
    system_prompt = CFG["openai"]["system_prompt"]
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    max_api_retries = CFG["openai"]["max_api_retries"]

    def __post_init__(self):
        if not hasattr(self, "scale_ranking"):
            raise ValueError("Must have field 'scale_ranking', with an int denoting relative strengths of different GPT family models (since GPT training flops aren't known)")
        assert isinstance(self.scale_ranking, int)
        self._encoding = tiktoken.encoding_for_model(self.name)
        super().__post_init__()
    
    @property
    def vocabulary(self):
        return self._encoding.name #placeholder, what we really want is the full token list...

    def inference(self, prompt: str | List[Dict], **optional_params):
        """
        Returns:
            If not getting logprobs:
                List of `n` continuations
            If getting logprobs:
                List of `n` pairs of (continuation, list of top-k tokens and respective logprobs at each sequence position)
        """
        for i in range(self.max_api_retries):
            try:
                inference_params = {
                    param: optional_params.get(param, value)
                    for param, value in chain(CFG["openai"]["inference_api"].items(), optional_params.items())
                }
                
                if isinstance(prompt, str):
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ]
                elif isinstance(prompt, List):
                    messages = prompt

                response = self.client.chat.completions.create(
                    model=self.name,
                    messages=messages,
                    **inference_params,
                )

                continuations = [choice.message.content for choice in response.choices]

                if inference_params.get("logprobs", False):
                    parsed_logprobs = self._parse_logprobs(response, "top_logprobs" in inference_params)
                    return list(zip(continuations, parsed_logprobs))

                return continuations
            
            except Exception as e:
                print(type(e), e)
                print(f"OpenAI API call failed, sleeping for {10 * (i+1)} seconds...")
                time.sleep(10 * (i + 1))

    def _parse_logprobs(self, response: ChatCompletion, has_top_logprobs: bool = True):
        if has_top_logprobs:
            parsed_logprobs = [
                [
                    (
                        {
                            single_top_logprob.token: single_top_logprob.logprob
                            for single_top_logprob in logprob_item.top_logprobs
                        }
                    )
                    for logprob_item in choice.logprobs.content
                ]
                for choice in response.choices
            ]
        else:
            parsed_logprobs = [
                [
                    {logprob_item.token: logprob_item.logprob}
                    for logprob_item in choice.logprobs.content
                ]
                for choice in response.choices
            ]
        return parsed_logprobs

    def get_output_logits(self, prompt: str, **optional_params):
        raise NotImplementedError("Not in the right output format yet, List[Tuple[str, Tensor]], this is a TODO")
        # model = OpenLogProbsOpenAIModel(self.name)
        # raw_logits = extract_logprobs(model, prompt, method="exact", parallel=True)
    
    def token_from_index(self, index: int):
        return self._encoding.decode_single_token_bytes(index).decode("utf-8")

    def finetune(self, dataset_path: str, save_name: str):
        raise NotImplementedError

@dataclass
class GPT4(GPTFamily):
    name = "gpt-4"
    description = "Strongest model in OpenAI GPT API model family."
    scale = None
    scale_ranking = 1

@dataclass
class GPT4_Turbo(GPTFamily):
    name = "gpt-4-turbo-preview"
    description = "Lighter variant of strongest model in OpenAI GPT API model family."
    scale = None
    scale_ranking = 2

@dataclass
class GPT3pt5_Turbo(GPTFamily):
    name = "gpt-3.5-turbo"
    description = "Medium chat model in OpenAI GPT API model family."
    scale = None
    scale_ranking = 3


@dataclass
class LegacyGPT(GPTFamily):
    # older model uses legacy api
    def __post_init__(self):
        super().__post_init__()

    def inference(self, prompt: str | List[Dict], **optional_params):
        if not isinstance(prompt, str):
            prompt = prompt[0]["content"] # TODO pretty bad abstraction
        
        for i in range(self.max_api_retries):
            try:
                inference_params = {
                    param: optional_params.get(param, value)
                    for param, value in chain(CFG["openai"]["legacy_inference_api"].items(), optional_params.items())
                }
                response = self.client.completions.create(
                    model=self.name,
                    prompt=prompt,
                    **inference_params,
                )

                continuations = [choice.text for choice in response.choices]

                if inference_params.get("logprobs", None):
                    parsed_logprobs = self._parse_logprobs(response)
                    return list(zip(continuations, parsed_logprobs))

                return continuations
            
            except Exception as e:
                print(type(e), e)
                print(f"OpenAI API call failed, sleeping for {10 * (i+1)} seconds...")
                time.sleep(10 * (i + 1))
    
    def _parse_logprobs(self, response: LegacyCompletion):
        parsed_logprobs = [choice.logprobs.top_logprobs[0] for choice in response.choices]
        return parsed_logprobs


@dataclass
class GPT3pt5_Turbo_Instruct(LegacyGPT):
    name = "gpt-3.5-turbo-instruct"
    description = "Medium instruct model in OpenAI GPT API model family."
    scale = None
    scale_ranking = 4

@dataclass
class GPT_Davinci(LegacyGPT):
    name = "davinci-002"
    description = "Weak base model in OpenAI GPT API model family."
    scale = None
    scale_ranking = 5


@dataclass
class GeminiFamily(BlackBoxModel):
    model = None

    def __post_init__(self):
        gemini.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.vocabulary = np.nan # hack to make vocabulary unequal to any other vocabulary
        super().__post_init__()
    
    def inference(self, prompt: str | List[Dict], **optional_params):
        if not isinstance(prompt, str):
            prompt = prompt[0]["content"] # TODO pretty bad abstraction
        
        # can't initialize with the name at dataclass init, so just init the first time inference is run
        if self.model is None:
            self.model = gemini.GenerativeModel(self.name)

        inference_params = {param: optional_params.get(param, value) for param, value in CFG["gemini"]["inference_api"].items()}
        generation_config = gemini.GenerationConfig(**inference_params)
        
        response = self.model.generate_content(prompt, generation_config=generation_config)

        #TODO either of these two return options should work, but I usually get zero completions. Looks like others online have this issue -- the API is a little shitty...
        return [candidate.content.parts for candidate in response.candidates]
        return response.text

    def get_output_logits(self, prompt: str, **optional_params):
        # no openlogprobs because no logit bias
        raise NotImplementedError
    
    def token_from_index(self, index: int):
        raise NotImplementedError

    def finetune(self, dataset_path: str, save_name: str):
        raise NotImplementedError
    

@dataclass
class Gemini1pt0_Pro(GeminiFamily):
    name = "gemini-1.0-pro"
    description = "Further iteration of medium model in Google Gemini API family."
    scale = None

@dataclass
class Gemini_Pro(GeminiFamily):
    name = "gemini-pro"
    description = "Medium model in Google Gemini API family."
    scale = None


@dataclass
class CohereFamily(BlackBoxModel):
    model = None
    system_prompt = CFG["cohere"]["system_prompt"]

    def __post_init__(self):
        self.model = cohere.Client(os.environ["COHERE_API_KEY"])
        self.vocabulary = np.nan
        super().__post_init__()

    def inference(self, prompt: str | List[Dict], **optional_params):
        if not isinstance(prompt, str):
            prompt = prompt[0]["content"] # TODO pretty bad abstraction
        
        inference_params = {
            param: optional_params.get(param, value)
            for param, value in chain(CFG["cohere"]["inference_api"].items(), optional_params.items())
        }
        n = inference_params.get("n", 1)
        del inference_params["n"]
        
        continuations = []
        for _ in range(n):
            response = self.model.chat(
                model=self.name,
                message=prompt,
                preamble_override=self.system_prompt,
                **inference_params
            )
            continuations.append(response.text)
        
        return continuations
    
    def get_output_logits(self, prompt: str, **optional_params):
        #TODO cohere.generate has a logit bias, so could add this to openlogprobs
        raise NotImplementedError
    
    def token_from_index(self, index: int):
        raise NotImplementedError

    def finetune(self, dataset_path: str, save_name: str):
        raise NotImplementedError


@dataclass
class CohereCommand(CohereFamily):
    name = "command"
    description = "Standard version of main Cohere model"
    scale = None

@dataclass
class CohereCommandLight(CohereFamily):
    name = "command-light"
    description = "'Turbo' version of main Cohere model"
    scale = None