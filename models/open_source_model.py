from dataclasses import dataclass
import gc
from itertools import chain
import os
from pathlib import Path
import torch
from typing import Dict, List
import yaml

from models.model_class import LanguageModel

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

CFG_PATH = Path(__file__).resolve().parents[1] / "configs/model_params.yaml"

with open(CFG_PATH) as cfg_file:
    CFG = yaml.safe_load(cfg_file)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class OpenSourceModel(LanguageModel):
    '''
    Currently performing all open-source model inference via HuggingFace,
    not vllm etc., so this stuff can be run on any machine
    '''
    def __post_init__(self):
        bnb_config_params = {param: value for param, value in CFG["opensource"]["bitsandbytes"].items()}
        bnb_config = BitsAndBytesConfig(
            **bnb_config_params
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            trust_remote_code=True,
            quantization_config=bnb_config,
            token=os.environ["HUGGINGFACE_TOKEN"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name, 
            max_tokens=CFG["opensource"]["inference"]["max_new_tokens"],
            trust_remote_code=True,
        )
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
          
        raw_vocab = self.tokenizer.get_vocab()
        self.vocabulary = sorted(raw_vocab.keys(), key=lambda word: raw_vocab[word])

        super().__post_init__()

    def _apply_chat_template(self, templated_prompt: str) -> str:
        chat_prompt = self.tokenizer.apply_chat_template(
            templated_prompt, tokenize=False, add_generation_prompt=True
        )
        return chat_prompt

    def _raw_inference(self, prompt_s: List[str] | List[List[Dict]], **optional_params):
        """
        Returns: `outputs` with fields:
            outputs.sequences: shape (batch_size, sequence_length) torch.Tensor, token ids corresponding to continuation string
            outputs.scores: len `sequence_length` tuple of shape (batch_size, vocab_size) torch.Tensors, output logits for each continuation token position across batch
        """
        inference_params = {
            param: optional_params.get(param, value)
            for param, value in chain(CFG["opensource"]["inference"].items(), optional_params.items())
        }
        if isinstance(prompt_s[0], Dict):
            prompt_s = [self._apply_chat_template(prompt) for prompt in prompt_s]

        inputs = self.tokenizer(prompt_s, return_tensors="pt", padding=True).to(DEVICE)
        outputs = self.model.generate(
            **inputs,
            do_sample=inference_params.get("temperature", 1.0) != 0.0,
            eos_token_id=self.eos_token_ids,
            pad_token_id=self.tokenizer.eos_token_id,
            **inference_params
        )
        return outputs
      
    def inference(self, prompt_s: str | List[Dict] | List[str] | List[List[Dict]], **optional_params):
        if isinstance(prompt_s, str) or isinstance(prompt_s[0], Dict): # jesus christ...
            prompt_s = [prompt_s,]
        raw_outputs = self._raw_inference(prompt_s, **optional_params)
        raw_continuations = self.tokenizer.batch_decode(
            raw_outputs.sequences, skip_special_tokens=True
        )
        continuations = [contd.removeprefix(prompt) for contd, prompt in zip(raw_continuations, prompt_s)]
        return continuations

    def get_output_logits(self, prompt_s: str | List[Dict] | List[str] | List[List[Dict]], **optional_params):
        """
        Unlike black-box, open-source is flexible enough for optional params
        
        Returns:
            logits: List of `batch_size` pairs of (continuation, list of tokens and respective logprobs at each sequence position)
        """
        if isinstance(prompt_s, str) or isinstance(prompt_s[0], Dict): # jesus christ...
            prompt_s = [prompt_s,]
        optional_params["output_scores"] = True
        raw_outputs = self._raw_inference(prompt_s, **optional_params)
        raw_continuations = self.tokenizer.batch_decode(
            raw_outputs.sequences, skip_special_tokens=True
        )
        continuations = [contd.removeprefix(prompt) for contd, prompt in zip(raw_continuations, prompt_s)]

        raw_scores = list(torch.stack(raw_outputs.scores, dim=1))

        logits = list(zip(continuations, raw_scores))

        return logits
    
    def token_from_index(self, index: int):
        return self.vocabulary[index]

    def finetune(self, dataset_path: str, save_name: str):
        raise NotImplementedError
    
    def destroy(self):
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        torch.cuda.empty_cache()


@dataclass
class MistralFamily(OpenSourceModel):
    pass

@dataclass
class Mistral_7B_Base(MistralFamily):
    name = "mistralai/Mistral-7B-v0.1"
    description = "Smallest model, base variant, of Mistral models"
    scale = None

@dataclass
class Mistral_7B_Instruct(MistralFamily):
    name = "mistralai/Mistral-7B-Instruct-v0.2"
    description = "Smallest model, instruct variant, of Mistral models"
    scale = None

@dataclass
class Mixtral_8x7B_Base(MistralFamily):
    name = "mistralai/Mixtral-8x7B-v0.1"
    description = "Strong MoE model, base variant, of Mistral models"
    scale = None

@dataclass
class Mixtral_8x7B_Instruct(MistralFamily):
    name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    description = "Strong MoE model, base variant, of Mistral models"
    scale = None


@dataclass
class LlamaFamily(OpenSourceModel):
    def __post_init__(self):
        self.eos_token_ids.extend([self.tokenizer.encode("}")[1], 29913, 9092, 16675])
        super().__post_init__()

@dataclass
class Llama2(LlamaFamily):
    def __post_init__(self):
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.padding_side = "left"
        super().__post_init__()

@dataclass
class Vicuna2(LlamaFamily):
    def __post_init__(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        super().__post_init__()

@dataclass
class MosaicFamily(OpenSourceModel):
    def __post_init__(self):
        self.tokenizer.padding_side = "left"
        super().__post_init__()

@dataclass
class MPT_7B_Base(MosaicFamily):
    name = "mosaicml/mpt-7b"
    description = "Smallest model, base variant, in MosaicML MPT family"
    scale = None #TODO I would imagine total train flops for MPTs are public, where do people get this info?