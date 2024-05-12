from dataclasses import dataclass, field, InitVar
import gc
from pathlib import Path
import torch
from typing import Callable
import yaml

from models.black_box_model import GPT3pt5_Turbo
from models.model_class import LanguageModel

import tiktoken
from transformers import AutoTokenizer

MODEL_CFG_PATH = Path(__file__).resolve().parents[1] / "configs/model_params.yaml"

with open(MODEL_CFG_PATH) as cfg_file:
    MODEL_CFG = yaml.safe_load(cfg_file)


class StringTransformation:
    """
    Encapsulates any invertible transformation on a text input/output for a language model.
    We have a soft requirement that inverse(f(`text`)) approximately equals `text` for any string text.
    """
    f: Callable = field(init=False)
    inverse: Callable = field(init=False)

    def __post_init__(self):
        if not hasattr(self, "name"):
            raise ValueError("Must have field 'name', with a small string name.")
        assert isinstance(self.name, str)

    def __call__(self, s):
        return self.f(s)
    
    def invert(self, s):
        """
        Returns inverse(str)
        """
        return self.inverse(s)
    
    def __str__(self):
        return self.name

@dataclass
class Id(StringTransformation):
    """
    Empty transformation mapping a string to itself.
    """
    name = "identity"

    def __post_init__(self):
        self.f = lambda s: s
        self.inverse = lambda s: s

@dataclass
class Reversal(StringTransformation):
    """
    Reverses a string.
    """
    name = "reversal"

    def __post_init__(self):
        self.f = lambda s: s[::-1]
        self.inverse = lambda s: s[::-1]


@dataclass
class TokenizerAwareTransformation(StringTransformation):
    """
    Specifies transformations which explicitly use the tokenizer of a given language model.
    """
    openai_model_name: InitVar[str | None] = None
    open_source_model_name: InitVar[str | None] = None

    def __post_init__(self):
        super().__post_init__()

@dataclass
class TokenizerAwareReversal(TokenizerAwareTransformation):
    """
    Reverses the tokens of a string based on the tokenizer of a given language model.
    """
    name = "tokenizer_aware_reversal"

    def __post_init__(self, openai_model_name, open_source_model_name):
        assert (openai_model_name is None) ^ (open_source_model_name is None), "(xor) exactly one of openai or open source should be specified"
        if openai_model_name is not None:
            self._tokenizer = tiktoken.encoding_for_model(openai_model_name)
            def transformation(s):
                tokens = self._tokenizer.encode(s)
                reversed_tokens = tokens[::-1]
                return self._tokenizer.decode(reversed_tokens)
            self.f = transformation
            self.inverse = transformation
        elif open_source_model_name is not None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.name, 
                max_tokens=MODEL_CFG["opensource"]["inference"]["max_new_tokens"],
                trust_remote_code=True,
            )
            def transformation(s):
                tokens = self._tokenizer(s, return_tensors="pt").input_ids
                reversed_tokens = tokens.flip(1)
                return self._tokenizer.decode(reversed_tokens[0], skip_special_tokens=True)
            self.f = transformation
            self.inverse = transformation
        super().__post_init__()


@dataclass
class NonDeterministicTransformation(StringTransformation):
    """
    Denotes transformations where inverse(f(text)) != text. Such transformations can be valid on a case-by-case basis.
    """
    def __post_init__(self):
        super().__post_init__()

@dataclass
class LanguageTranslation(NonDeterministicTransformation):
    """
    Translates text input into another specified language, with the ability to translate back.
    Note that translation is nondeterministic.
    """
    language: str = field(default="Chinese")
    name = "language_translation"

    def __post_init__(self):
        self._translator = GPT3pt5_Turbo()

        self._translation_task = f"Please translate the following text from English to {self.language}:\n"
        def transformation(s):
            just_one_output = {"n": 1}
            res = self._translator.inference(self._translation_task + s, **just_one_output)[0]
            return res
        self.f = transformation

        self._translation_back_task = f"Please translate the following text from {self.language} to English:\n"
        def inverse_transformation(s):
            just_one_output = {"n": 1}
            res = self._translator.inference(self._translation_back_task + s, **just_one_output)[0]
            return res
        self.inverse = inverse_transformation


TRANSFORMATIONS_BY_NAME = {
    transformation.name: transformation
    for transformation in (Id, Reversal, TokenizerAwareReversal, LanguageTranslation)
}


ICL_EXAMPLARS = [
    (
        "How does artificial intelligence work?",
        "Artificial intelligence (AI) operates by analyzing large sets of data, recognizing patterns, and making decisions based on that analysis. It combines algorithms in machine learning, neural networks, and sometimes natural language processing, to mimic human cognition. AI systems learn from data inputs and improve over time, enabling them to perform tasks ranging from simple classification to complex problem-solving."
    ),
    (
        "Can you explain how photosynthesis works?",
        "Photosynthesis is the process by which green plants, algae, and certain bacteria convert light energy, usually from the sun, into chemical energy stored in glucose, a type of sugar, using water and carbon dioxide. This occurs in two main stages: the light-dependent reactions and the Calvin cycle. In the light-dependent reactions, which take place in the thylakoid membranes of chloroplasts, sunlight is absorbed by chlorophyll, causing it to release electrons that help generate ATP and NADPH. These energy carriers then power the Calvin cycle in the stroma of the chloroplast, where carbon dioxide is fixed into glucose. Oxygen is released as a byproduct of this process, contributing to the oxygen we breathe."
    ),
    (
        "How does the stock market work?",
        "The stock market is a platform where buyers and sellers come together to trade shares of publicly listed companies. It operates through exchanges like the New York Stock Exchange or NASDAQ. Companies list their shares on an exchange through an initial public offering (IPO), allowing investors to buy and sell these shares. The price of stocks is determined by supply and demand dynamics, influenced by various factors including company performance, economic indicators, and market sentiment. Investors can make profits by buying stocks at a lower price and selling them at a higher price, or through dividends paid by the company."
    ),
    (
        "Can you give me a brief summary of World War II?",
        "World War II was a global conflict that lasted from 1939 to 1945, involving most of the world's nations. It was primarily fought between the Axis Powers (mainly Germany, Italy, and Japan) and the Allied Powers (primarily the United Kingdom, Soviet Union, and the United States). It resulted in significant loss of life and was marked by key events like the Holocaust, the bombing of Hiroshima and Nagasaki, and the eventual defeat of Nazi Germany and Imperial Japan. World War II had profound effects on the course of world history, including the emergence of the United States and the Soviet Union as superpowers and the beginning of the Cold War."
    ),
    (
        "What are the benefits of regular physical exercise?",
        "Regular physical exercise offers a multitude of benefits for both the body and mind. Physically, it helps improve cardiovascular health, reducing the risk of heart disease, stroke, and hypertension. It aids in maintaining a healthy weight, strengthening muscles and bones, and enhancing flexibility and balance, which can prevent falls and injuries. From a mental health perspective, exercise is known to release endorphins, often referred to as feel-good hormones, which can improve mood and reduce feelings of depression and anxiety. Additionally, regular physical activity can boost cognitive function, improve sleep quality, and increase lifespan. Engaging in a variety of exercises, including aerobic activities, strength training, and flexibility exercises, can maximize health benefits and prevent the monotony of a single routine."
    ),
    (
        "What is the concept of 'supply and demand' in economics?",
        "The concept of 'supply and demand' is a fundamental economic model that describes how the price and quantity of goods and services are determined in a market. Demand refers to how much (quantity) of a product or service is desired by buyers at various prices, while supply represents how much the market can offer. The price of a good or service is determined by the point at which supply and demand meet, known as the equilibrium price. When demand increases for a fixed supply, prices rise as consumers are willing to pay more to secure the item. Conversely, if supply increases for a fixed demand, prices fall as more of the product is available. This dynamic interaction between supply and demand ensures the efficient distribution of resources in a market economy, guiding the allocation of goods and services to those who value them most."
    ),
    (
        "How do cyberattacks impact businesses?",
        "Cyberattacks can have devastating impacts on businesses, ranging from financial losses to reputational damage. Financially, the direct costs include ransom payments in the case of ransomware attacks, system restoration, and data recovery expenses. There are also indirect costs, such as operational downtime, lost productivity, and legal fees associated with breaches of customer or employee data. Beyond the tangible impacts, cyberattacks can erode customer trust and loyalty, especially if sensitive personal information is compromised. This loss of trust can lead to a decline in business and may be difficult to recover from. Furthermore, businesses may face regulatory fines if found non-compliant with data protection laws. In the long term, cyberattacks can disrupt strategic plans and lead to competitive disadvantages, as resources are diverted to recovery and strengthening cybersecurity defenses instead of growth or innovation initiatives."
    ),
    (
        "How do black holes form?",
        "A black hole forms from the remnants of a massive star that has ended its life cycle. When such a star depletes its nuclear fuel, it can no longer sustain the nuclear fusion reactions that counterbalance gravitational forces. As a result, the core collapses under its own immense gravity, leading to a supernova explosion that ejects the star's outer layers into space. If the remaining core mass is sufficiently large—typically more than about 2.5 times the mass of the Sun—it collapses to a point of infinite density known as a singularity, surrounded by an event horizon. This event horizon marks the boundary beyond which nothing, not even light, can escape the black hole's gravitational pull, rendering the black hole invisible and detectable only by its gravitational effects on nearby matter and radiation."
    )
]