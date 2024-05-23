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


# ~~~~~~~~~~~~~~~~~~~~ LLAMA (META) ~~~~~~~~~~~~~~~~~~~~

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
class Llama2_7B(Llama2):
    name = "meta-llama/Llama-2-7b-hf"
    description = "Small variant of Llama2 models"
    scale = None

@dataclass
class Llama2_7B_Chat(Llama2):
    name = "meta-llama/Llama-2-7b-chat-hf"
    description = "Small chat variant of Llama2 models"
    scale = None

@dataclass
class Llama2_13B(Llama2):
    name = "meta-llama/Llama-2-13b-hf"
    description = "Medium variant of Llama2 models"
    scale = None

@dataclass
class Llama2_13B_Chat(Llama2):
    name = "meta-llama/Llama-2-13b-chat-hf"
    description = "Medium chat variant of Llama2 models"
    scale = None

@dataclass
class Llama2_70B(Llama2):
    name = "meta-llama/Llama-2-70b-hf"
    description = "Large variant of Llama2 models"
    scale = None

@dataclass
class Llama2_70B_Chat(Llama2):
    name = "meta-llama/Llama-2-70b-chat-hf"
    description = "Large chat variant of Llama2 models"
    scale = None

@dataclass
class Llama3(LlamaFamily):
    def __post_init__(self):
        super().__post_init__()

@dataclass
class Llama3_8B(Llama2):
    name = "meta-llama/Meta-Llama-3-8B"
    description = "Small variant of Llama3 models"
    scale = None

@dataclass
class Llama3_8B_Instruct(Llama2):
    name = "meta-llama/Meta-Llama-3-8B-Instruct"
    description = "Small instruct variant of Llama3 models"
    scale = None

@dataclass
class Llama3_70B(Llama2):
    name = "meta-llama/Meta-Llama-3-70B"
    description = "Large variant of Llama3 models"
    scale = None

@dataclass
class Llama3_70B_Chat(Llama2):
    name = "meta-llama/Meta-Llama-3-70B-Instruct"
    description = "Large instruct variant of Llama3 models"
    scale = None

@dataclass
class Vicuna2(LlamaFamily):
    def __post_init__(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        super().__post_init__()


# ~~~~~~~~~~~~~~~~~~~~ GEMMA (GOOGLE) ~~~~~~~~~~~~~~~~~~~~

@dataclass
class GemmaFamily(OpenSourceModel):
    def __post_init__(self):
        super().__post_init__()

@dataclass
class Gemma_2B(GemmaFamily):
    name = "google/gemma-2b"
    description = "Small variant of Gemma models"
    scale = None

@dataclass
class Gemma_2B_Instruct(GemmaFamily):
    name = "google/gemma-2b-it"
    description = "Small instruct variant of Gemma models"
    scale = None

@dataclass
class Gemma_7B(GemmaFamily):
    name = "google/gemma-7b"
    description = "Large variant of Gemma models"
    scale = None

@dataclass
class Gemma_7B_Instruct(GemmaFamily):
    name = "google/gemma-7b-it"
    description = "Large instruct variant of Gemma models"
    scale = None

# ~~~~~~~~~~~~~~~~~~~~ MISTRAL ~~~~~~~~~~~~~~~~~~~~

@dataclass
class MistralFamily(OpenSourceModel):
    def __post_init__(self):
        super().__post_init__()

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
    description = "Medium MoE model, base variant, of Mistral models"
    scale = None

@dataclass
class Mixtral_8x7B_Instruct(MistralFamily):
    name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    description = "Medium MoE model, instruct variant, of Mistral models"
    scale = None

@dataclass
class Mixtral_8x22B_Base(MistralFamily):
    name = "mistralai/Mixtral-8x22B-v0.1"
    description = "Strong MoE model, base variant, of Mistral models"

@dataclass
class Mixtral_8x22B_Instruct(MistralFamily):
    name = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    description = "Strong MoE model, instruct variant, of Mistral models"

# ~~~~~~~~~~~~~~~~~~~~ COMMAND-R (COHERE) ~~~~~~~~~~~~~~~~~~~~

@dataclass
class OpenCohereFamily(OpenSourceModel):
    def __post_init__(self):
        super().__post_init__()

@dataclass
class CommandR_Plus(OpenCohereFamily):
    name = "CohereForAI/c4ai-command-r-plus"
    description = "Newer version of Cohere open model. 104B params."

@dataclass
class CommandR(OpenCohereFamily):
    name = "CohereForAI/c4ai-command-r-v01"
    description = "Older version of Cohere open model. 35B params."

# ~~~~~~~~~~~~~~~~~~~~ DATABRICKS MOSAIC ~~~~~~~~~~~~~~~~~~~~

@dataclass
class DbrxMosaicFamily(OpenSourceModel):
    def __post_init__(self):
        super().__post_init__()

@dataclass
class Dbrx(DbrxMosaicFamily):
    name = "databricks/dbrx-base"
    description = "Strong MoE model from Databricks Mosaic. 132B total parameters, 36B active MoE parameters."
    scale = 132e9

@dataclass
class Dbrx_Instruct(DbrxMosaicFamily):
    name = "databricks/dbrx-instruct"
    description = "Instruct variant of strong MoE model from Databricks Mosaic. 132B total parameters, 36B active MoE parameters."
    scale = 36e9

# ~~~~~~~~~~~~~~~~~~~~ QWEN (ALIBABA) ~~~~~~~~~~~~~~~~~~~~

@dataclass
class QwenFamily(OpenSourceModel):
    def __post_init__(self):
        super().__post_init__()

@dataclass
class Qwen1pt5_0pt5B_Base(QwenFamily):
    name = "Qwen/Qwen1.5-0.5B"
    description = "Qwen1.5 0.5B dense base model"
    scale = 0.5e9

@dataclass
class Qwen1pt5_0pt5B_Chat(QwenFamily):
    name = "Qwen/Qwen1.5-0.5B-Chat" 
    description = "Qwen1.5 0.5B dense chat model"
    scale = 0.5e9

@dataclass
class Qwen1pt5_1pt8B_Base(QwenFamily):
    name = "Qwen/Qwen1.5-1.8B"
    description = "Qwen1.5 1.8B dense base model"
    scale = 1.8e9

@dataclass
class Qwen1pt5_1pt8B_Chat(QwenFamily):
    name = "Qwen/Qwen1.5-1.8B-Chat"
    description = "Qwen1.5 1.8B dense chat model" 
    scale = 1.8e9

@dataclass
class Qwen1pt5_4B_Base(QwenFamily):
    name = "Qwen/Qwen1.5-4B"
    description = "Qwen1.5 4B dense base model"
    scale = 4e9

@dataclass
class Qwen1pt5_4B_Chat(QwenFamily):
    name = "Qwen/Qwen1.5-4B-Chat"
    description = "Qwen1.5 4B dense chat model"
    scale = 4e9

@dataclass
class Qwen1pt5_7B_Base(QwenFamily):
    name = "Qwen/Qwen1.5-7B" 
    description = "Qwen1.5 7B dense base model"
    scale = 7e9

@dataclass
class Qwen1pt5_7B_Chat(QwenFamily):
    name = "Qwen/Qwen1.5-7B-Chat"
    description = "Qwen1.5 7B dense chat model"
    scale = 7e9

@dataclass
class Qwen1pt5_14B_Base(QwenFamily):
    name = "Qwen/Qwen1.5-14B"
    description = "Qwen1.5 14B dense base model"
    scale = 14e9

@dataclass 
class Qwen1pt5_14B_Chat(QwenFamily):
    name = "Qwen/Qwen1.5-14B-Chat"
    description = "Qwen1.5 14B dense chat model"
    scale = 14e9

@dataclass
class Qwen1pt5_32B_Base(QwenFamily):
    name = "Qwen/Qwen1.5-32B"
    description = "Qwen1.5 32B dense base model"
    scale = 32e9

@dataclass
class Qwen1pt5_32B_Chat(QwenFamily):
    name = "Qwen/Qwen1.5-32B-Chat"
    description = "Qwen1.5 32B dense chat model"
    scale = 32e9

@dataclass
class Qwen1pt5_72B_Base(QwenFamily):
    name = "Qwen/Qwen1.5-72B"
    description = "Qwen1.5 72B dense base model"
    scale = 72e9

@dataclass
class Qwen1pt5_72B_Chat(QwenFamily):
    name = "Qwen/Qwen1.5-72B-Chat"
    description = "Qwen1.5 72B dense chat model"
    scale = 72e9

@dataclass
class Qwen1pt5_110B_Base(QwenFamily):
    name = "Qwen/Qwen1.5-110B"
    description = "Qwen1.5 110B dense base model"
    scale = 110e9

@dataclass
class Qwen1pt5_110B_Chat(QwenFamily):
    name = "Qwen/Qwen1.5-110B-Chat"
    description = "Qwen1.5 110B dense chat model"
    scale = 110e9

@dataclass
class Qwen1pt5_MoE_Base(QwenFamily):
    name = "Qwen/Qwen1.5-MoE-A2.7B"
    description = "Qwen1.5 14B MoE base model with 2.7B activated"
    scale = 14e9

@dataclass
class Qwen1pt5_MoE_Chat(QwenFamily):
    name = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
    description = "Qwen1.5 14B MoE chat model with 2.7B activated"
    scale = 14e9