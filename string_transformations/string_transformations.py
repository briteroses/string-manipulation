"""
String transformations to implement:
    Identity (trivial)
    Reversal
    Tokenizer-aware reversal
    Language translation
    Word-level reversal
    Caesar ciphers
    Vowel-only or consonant-only Caesar ciphers
    Atbash cipher
    Base64 encoding; (also, base16, base32, base85, base58)
    Binary encoding
    Leetspeak
    Morse code
    Vowel repetition
    Alternating case
    Markdown
    JSON
"""

import base58
import base64
import gc
import json
import torch
import yaml
from dataclasses import dataclass, field, InitVar
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Callable

from models.black_box_model import GPT3pt5_Turbo, GPTFamily
from models.model_class import LanguageModel
from string_transformations.consts import ALPHABET, LEETSPEAK_DICT, MORSE_DICT, VOWELS

import tiktoken
from transformers import AutoTokenizer

MODEL_CFG_PATH = Path(__file__).resolve().parents[1] / "configs/model_params.yaml"

with open(MODEL_CFG_PATH) as cfg_file:
    MODEL_CFG = yaml.safe_load(cfg_file)


class StringTransformation:
    """
    Encapsulates any invertible transformation on a text input/output for a language model.
    We have a soft requirement that _inverse(_f(`text`)) approximately equals `text` for any string text.
    """
    _f: Callable = field(init=False)
    _inverse: Callable = field(init=False)

    def __post_init__(self):
        if not hasattr(self, "name"):
            raise ValueError("Must have field 'name', with a small string name.")
        if not hasattr(self, "instruction"):
            raise ValueError("Must have field 'instruction', with a short text description describing how to do the transformation.")
        assert isinstance(self.name, str)

    def __call__(self, s):
        return self._f(s)
    
    def invert(self, s):
        """
        Returns _inverse(str)
        """
        return self._inverse(s)
    
    def __str__(self):
        return self.name

# ~~~~~~~~~~~~~~~~~~~~
# identity, reversal, word-level reversal, caesar ciphers, rot13, atbash cipher, base64, baseXX, binary, leetspeak, morse code, vowel repetition, alternating case, markdown, json
# ~~~~~~~~~~~~~~~~~~~~

@dataclass
class Id(StringTransformation):
    name = "identity"
    instruction = "Maintain the original string without any alterations."

    def __post_init__(self):
        self._f = lambda s: s
        self._inverse = lambda s: s

@dataclass
class Reversal(StringTransformation):
    name = "reversal"
    instruction = "Change all the characters in the string to be in reverse order."

    def __post_init__(self):
        self._f = lambda s: s[::-1]
        self._inverse = lambda s: s[::-1]

@dataclass
class WordLevelReversal(StringTransformation):
    # WARNING: squashes newlines, every whitespace becomes an \s. Can change this convention later, but this is what was currently decided.
    name = "word-level reversal"
    instruction = "Change all the words in a string to be in reverse order, without altering the order of characters in any word."

    def __post_init__(self):
        def transformation(s):
            words = s.split(" ")
            reversed_words = words[::-1]
            return " ".join(reversed_words)
        
        self._f = transformation
        self._inverse = transformation

def caesar_cipher_transformation(s, cipher_step):
    encoded = ""
    QUICK_FIND_ALPHABET = {ALPHABET[i]: i for i in range(len(ALPHABET))}
    for char in s:
        if not char.isalpha():
            encoded += char
            continue
        was_upper = char.isupper()
        char_position = QUICK_FIND_ALPHABET[char.lower()]
        replacement = ALPHABET[(char_position + cipher_step) % len(ALPHABET)]
        if was_upper:
            replacement = replacement.upper()
        encoded += replacement
    return encoded

@dataclass
class CaesarCipher(StringTransformation):
    name = "Caesar cipher"
    instruction = "Encode the string using the well-known Caesar cipher, in which each alphabetical character is replaced with the letter 3 positions down the alphabet."

    def __post_init__(self):
        self._f = partial(caesar_cipher_transformation, cipher_step=3)
        self._inverse = partial(caesar_cipher_transformation, cipher_step=-3)

@dataclass
class ROT13Cipher(StringTransformation):
    name = "ROT13 cipher"
    instruction = "Encode the string using the well-known ROT13 cipher, in which each alphabetical character is replaced with the letter 13 positions down the alphabet."

    def __post_init__(self):
        self._f = partial(caesar_cipher_transformation, cipher_step=13)
        self._inverse = partial(caesar_cipher_transformation, cipher_step=-13)
    
@dataclass
class AtbashCipher(StringTransformation):
    name = "Atbash cipher"
    instruction = "Encode the string using the well-known Atbash cipher, in which the alphabet is mapped to the reverse-order alphabet. For instance, the first letter A maps to the last letter Z, the second letter B maps to the second-to-last letter Y, and so on."

    def __post_init__(self):
        def transformation(s):
            encoded = ""
            for char in s:
                if not char.isalpha():
                    encoded += char
                    continue
                was_upper = char.isupper()
                char_position = ALPHABET.index(char.lower())
                replacement = ALPHABET[len(ALPHABET) - 1 - char_position]
                if was_upper:
                    replacement = replacement.upper()
                encoded += replacement
            return encoded
    
        self._f = transformation
        self._inverse = transformation

@dataclass
class Base16(StringTransformation):
    name = "Base16 encoding"
    instruction = "Encode the string using Base16 encoding."

    def __post_init__(self):
        def transformation(s):
            encoded_bytes = base64.b16encode(s.encode('utf-8'))
            return encoded_bytes.decode('utf-8')
        
        def inverse_transformation(s):
            decoded_bytes = base64.b16decode(s.encode('utf-8'))
            return decoded_bytes.decode('utf-8')

        self._f = transformation
        self._inverse = inverse_transformation

@dataclass
class Base32(StringTransformation):
    name = "Base32 encoding"
    instruction = "Encode the string using Base32 encoding."

    def __post_init__(self):
        def transformation(s):
            encoded_bytes = base64.b32encode(s.encode('utf-8'))
            return encoded_bytes.decode('utf-8')
        
        def inverse_transformation(s):
            decoded_bytes = base64.b32decode(s.encode('utf-8'))
            return decoded_bytes.decode('utf-8')

        self._f = transformation
        self._inverse = inverse_transformation

@dataclass
class Base64(StringTransformation):
    name = "Base64 encoding"
    instruction = "Encode the string using Base64 encoding."

    def __post_init__(self):
        def transformation(s):
            encoded_bytes = base64.b64encode(s.encode('utf-8'))
            return encoded_bytes.decode('utf-8')
        
        def inverse_transformation(s):
            decoded_bytes = base64.b64decode(s.encode('utf-8'))
            return decoded_bytes.decode('utf-8')

        self._f = transformation
        self._inverse = inverse_transformation

@dataclass
class Base85(StringTransformation):
    name = "Base85 encoding"
    instruction = "Encode the string using Base85 encoding."

    def __post_init__(self):
        def transformation(s):
            encoded_bytes = base64.b85encode(s.encode('utf-8'))
            return encoded_bytes.decode('utf-8')
        
        def inverse_transformation(s):
            decoded_bytes = base64.b85decode(s.encode('utf-8'))
            return decoded_bytes.decode('utf-8')

        self._f = transformation
        self._inverse = inverse_transformation

@dataclass
class Base58(StringTransformation):
    name = "Base58 encoding"
    instruction = "Encode the string using Base58 encoding."

    def __post_init__(self):
        def transformation(s):
            encoded_bytes = base58.b58encode(s.encode('utf-8'))
            return encoded_bytes.decode('utf-8')
        
        def inverse_transformation(s):
            decoded_bytes = base58.b58decode(s.encode('utf-8'))
            return decoded_bytes.decode('utf-8')

        self._f = transformation
        self._inverse = inverse_transformation

@dataclass
class Binary(StringTransformation):
    name = "binary encoding"
    instruction = "Convert the string into binary, with each character represented by its 8-bit ASCII code."

    def __post_init__(self):
        def transformation(s):
            return " ".join(format(ord(ch), "08b") for ch in s)
        
        def inverse_transformation(s):
            return "".join(chr(int(code, 2)) for code in s.split())

        self._f = transformation
        self._inverse = inverse_transformation

@dataclass
class Leetspeak(StringTransformation):
    # WARNING: naive algorithm, will destroy numbers; still works for research purposes, but not a rigorous inversion
    name = "leetspeak"
    instruction = "Convert the string into the well-known leetspeak alphabet used in some Internet communities."

    def __post_init__(self):
        def transformation(s, replacement_dict):
            return "".join([replacement_dict.get(ch.lower(), ch) for ch in s])

        self._f = partial(transformation, replacement_dict=LEETSPEAK_DICT)
        self._inverse = partial(transformation, replacement_dict={v: k for k, v in LEETSPEAK_DICT.items()})

@dataclass
class MorseCode(StringTransformation):
    name = "Morse code"
    instruction = "Convert the string into Morse code."

    def __post_init__(self):
        def transformation(s):
            return " ".join(MORSE_DICT.get(c.lower(), c) for c in s)
        
        def inverse_transformation(s):
            inv_morse_dict = {v: k for k, v in MORSE_DICT.items()}
            return "".join(inv_morse_dict.get(code, code) for code in s.split(" "))

        self._f = transformation
        self._inverse = inverse_transformation

@dataclass
class VowelRepetition(StringTransformation):
    name = "vowel repetition"
    instruction = "Change the string to have every vowel repeated 3 times. For example, any instance of `a` becomes `aaa`, and so on."

    def __post_init__(self):
        def transformation(s):
            encoded = ""
            for char in s:
                if char.lower() not in VOWELS:
                    encoded += char
                    continue
                repeated_vowel = char * 3
                encoded += repeated_vowel
            return encoded

        self._f = transformation

        def inverse_transformation(s):
            for vowel in chain(VOWELS, map(lambda ch: ch.upper(), VOWELS)):
                s = s.replace(vowel * 3, vowel)
            return s
        
        self._inverse = inverse_transformation

@dataclass
class AlternatingCase(StringTransformation):
    # WARNING: be careful using this in transformations that interfere with case, such as leetspeak and morse code
    name = "alternating case"
    instruction = "Change the string to be in alternating case, in which the first character is uppercase and each subsequent character alternates between lowercase and uppercase."
    
    def __post_init__(self):
        def transformation(s):
            encoded = ""
            do_upper_or_lower = True
            for char in s:
                if not char.isalpha():
                    encoded += char
                    continue
                encoded += char.upper() if do_upper_or_lower else char.lower()
                do_upper_or_lower = not do_upper_or_lower
            return encoded
        
        self._f = transformation
        self._inverse = lambda s: s.lower()

# ~~~~~~~~~~~~~~~~~~~~
# Transforms that I separated from the above for different "vibes"; maybe some structure/function-calling encapsulation, maybe inherent to GPT models, etc.
# These can't be composed willy-nilly with the above transformations; will need to use these in special cases only, i.e. only as the outermost transform in a composition?
# markdown, json
# ~~~~~~~~~~~~~~~~~~~~

@dataclass
class PythonMarkdown(StringTransformation):
    # WARNING: relies on how the given model interprets Python markdown; possibly need per-model casework
    name = "Python markdown"
    instruction = "Change the string to be in Python in a markdown format."
    model_type: InitVar[type] = GPTFamily

    def __post_init__(self, model_type):
        if issubclass(model_type, GPTFamily):
            # currently, just removes the ```python {message} ``` scaffolding, and removes the multiline string quotations that are often put around the message
            def transformation(s):
                markdown_s = '```python\n"""' + s + '"""\n```\n'
                return markdown_s
                
            self._f = transformation

            def inverse_transformation(s):
                start_tag, end_tag = "```python\n", "```\n"
                start_tag_idx = s.find(start_tag)
                end_tag_idx = s.rfind(end_tag)
                stripped_s = s[:start_tag_idx] + s[start_tag_idx+len(start_tag):end_tag_idx] + s[end_tag_idx+len(end_tag):]
                stripped_s = stripped_s.replace('"""', '')
                return stripped_s
            
            self._inverse = inverse_transformation
        else:
            raise NotImplementedError

@dataclass
class JSON_Encapsulation(StringTransformation):
    # WARNING: relies on how the given model interprets json encapsulation; possibly need per-model casework
    name = "JSON encapsulation"
    instruction = "Change the string to be encapsulated in a json format."

    def __post_init__(self):
        # currently, just removes 
        def transformation(s):
            to_json = {
                "message": s
            }
            encapsulated_s = json.dumps(to_json)
            return encapsulated_s
        
        self._f = transformation

        def unpack_dict(d):
            ret = []
            for v in d.values():
                if isinstance(v, str):
                    ret.append(v)
                if isinstance(v, dict):
                    ret.extend(unpack_dict(v))
            return ret

        def inverse_transformation(s):
            messages = unpack_dict(json.loads(s, strict=False))
            unencapsulated_s = "\n".join(messages)
            return unencapsulated_s

        self._inverse = inverse_transformation

# ~~~~~~~~~~~~~~~~~~~~
# Transforms that count as special cases, for whatever reason
# tokenizer-aware reversal, language translation
# ~~~~~~~~~~~~~~~~~~~~

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
    # WARNING: It's not always true that encode(decode(tokens)) == tokens, so this actually isn't deterministic.
    name = "tokenizer-aware reversal"
    instruction = "Print all the tokens of a string in reverse order. The tokenization should come from the tokenizer corresponding to this model."

    def __post_init__(self, openai_model_name, open_source_model_name):
        assert (openai_model_name is None) ^ (open_source_model_name is None), "(xor) exactly one of openai or open source should be specified"
        if openai_model_name is not None:
            self._tokenizer = tiktoken.encoding_for_model(openai_model_name)
            def transformation(s):
                tokens = self._tokenizer.encode(s)
                reversed_tokens = tokens[::-1]
                return self._tokenizer.decode(reversed_tokens)
            self._f = transformation
            self._inverse = transformation
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
            self._f = transformation
            self._inverse = transformation
        super().__post_init__()


@dataclass
class NonDeterministicTransformation(StringTransformation):
    """
    Denotes transformations where _inverse(_f(text)) != text. Such transformations can be valid on a case-by-case basis. Currently, the primary case for this is language translation.
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
    name = "language translation"

    def __post_init__(self):
        self.instruction = f"Translate the string from English to {self.language}."

        self._translator = GPT3pt5_Turbo()

        self._translation_task = f"Please translate the following text from English to {self.language}:\n"
        def transformation(s):
            just_one_output = {"n": 1}
            res = self._translator.inference(self._translation_task + s, **just_one_output)[0]
            return res
        self._f = transformation

        self._translation_back_task = f"Please translate the following text from {self.language} to English:\n"
        def inverse_transformation(s):
            just_one_output = {"n": 1}
            res = self._translator.inference(self._translation_back_task + s, **just_one_output)[0]
            return res
        self._inverse = inverse_transformation


TRANSFORMATIONS_BY_NAME = {
    transformation.name: transformation
    for transformation in (Id, Reversal, TokenizerAwareReversal, LanguageTranslation, WordLevelReversal, CaesarCipher, ROT13Cipher, AtbashCipher, Base16, Base32, Base64, Base85, Base58, Binary, Leetspeak, MorseCode, VowelRepetition, AlternatingCase, PythonMarkdown, JSON_Encapsulation)
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