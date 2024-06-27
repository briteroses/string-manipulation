from copy import copy
import random
import re
from string import Template
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

from models.black_box_model import ClaudeFamily, GPT3pt5_Turbo, GPTFamily, HaizerMistral
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

    @classmethod
    def construct(cls):
        return cls()

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
class PerWordReversal(StringTransformation):
    name = "per-word reversal"
    instruction = "Reverse the characters of each word, while keeping the words in the original order."

    def __post_init__(self):
        def transformation(s):
            words = s.split()
            reversed_words_in_order = list(map(lambda word: word[::-1], words))
            return " ".join(reversed_words_in_order)
        
        self._f = transformation
        self._inverse = transformation

@dataclass
class WordLevelReversal(StringTransformation):
    # WARNING: squashes newlines, every whitespace becomes an \s. Can change this convention later, but this is what was currently decided.
    name = "word-level reversal"
    instruction = "Change all the words in a string to be in reverse order, without altering the order of characters in any word."

    def __post_init__(self):
        def transformation(s):
            words = s.split()
            reversed_words = words[::-1]
            return " ".join(reversed_words)
        
        self._f = transformation
        self._inverse = transformation

def caesar_cipher_transformation(s, cipher_step):
    encoded = ""
    QUICK_FIND_ALPHABET = {ALPHABET[i]: i for i in range(len(ALPHABET))}
    for char in s:
        if char.lower() not in ALPHABET:
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
                if char not in ALPHABET:
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
class BaseN(StringTransformation):
    base_number: int = field(default=64)
    name: str = "BaseN encoding"
    instruction: str = field(init=False)

    def __post_init__(self):
        if self.base_number == 16:
            self.name = "Base16 encoding"
            self.instruction = "Encode the string using Base16 encoding. Output only the encoding."
            transformation = lambda s: base64.b16encode(s.encode('utf-8')).decode('utf-8')
            def inverse_transformation(s):
                try:
                    return base64.b16decode(s.encode('utf-8')).decode('utf-8')
                except:
                    return s
        elif self.base_number == 32:
            self.name = "Base32 encoding"
            self.instruction = "Encode the string using Base32 encoding. Output only the encoding."
            transformation = lambda s: base64.b32encode(s.encode('utf-8')).decode('utf-8')
            def inverse_transformation(s):
                try:
                    return base64.b32decode(s.encode('utf-8')).decode('utf-8')
                except:
                    return s
        elif self.base_number == 64:
            self.name = "Base64 encoding"
            self.instruction = "Encode the string using Base64 encoding. Output only the encoding."
            transformation = lambda s: base64.b64encode(s.encode('utf-8')).decode('utf-8')
            def inverse_transformation(s):
                try:
                    return base64.b64decode(s.encode('utf-8')).decode('utf-8')
                except:
                    return s
        elif self.base_number == 85:
            self.name = "Base85 encoding"
            self.instruction = "Encode the string using Base85 encoding. Output only the encoding."
            transformation = lambda s: base64.b85encode(s.encode('utf-8')).decode('utf-8')
            def inverse_transformation(s):
                try:
                    return base64.b85decode(s.encode('utf-8')).decode('utf-8')
                except:
                    return s
        elif self.base_number == 58:
            self.name = "Base58 encoding"
            self.instruction = "Encode the string using Base58 encoding. Output only the encoding."
            transformation = lambda s: base58.b58encode(s.encode('utf-8')).decode('utf-8')
            def inverse_transformation(s):
                try:
                    return base58.b58decode(s.encode('utf-8')).decode('utf-8')
                except:
                    return s
        else:
            raise ValueError(f"Unsupported base number: {self.base_number}")

        self._f = transformation
        self._inverse = inverse_transformation
    
    @classmethod
    def construct(cls, choice=None):
        if choice is None:
            random_base_number = random.choice(BASE_N_OPTIONS)
            return cls(base_number=random_base_number)
        return cls(base_number=choice)

@dataclass
class Binary(StringTransformation):
    name = "binary encoding"
    instruction = "Convert the string into binary, with each character represented by its 8-bit ASCII code."

    def __post_init__(self):
        def transformation(s):
            return " ".join(format(ord(ch), "08b") for ch in s)
        
        def inverse_transformation(s):
            try:
                return "".join(chr(int(code, 2)) for code in s.split())
            except ValueError:
                return s

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

@dataclass
class Palindrome(StringTransformation):
    # WARNING: could get confusing with punctuation, like commas and emdashes
    name = "palindrome"
    instruction = "Convert each word into a palindrome by appending each word's reverse to itself."

    def __post_init__(self):
        def transformation(s):
            words = s.split()
            palindromes = list(map(lambda word: word + word[:-1][::-1], words))
            return " ".join(palindromes)
        
        self._f = transformation

        def inverse_transformation(s):
            palindromes = s.split()
            words = []
            for palindrome in palindromes:
                if palindrome.lower() != palindrome.lower()[::-1]:
                    words.append(palindrome)
                else:
                    word = palindrome[:(len(palindrome)+1)//2]
                    words.append(word)
            return " ".join(words)
        
        self._inverse = inverse_transformation

@dataclass
class Delimiters(StringTransformation):
    delimiter: str
    name = "interleaving with delimiter"
    instruction: str = field(init=False)

    def __post_init__(self):
        self.instruction = f"Interleave the delimiter \"{self.delimiter}\" between the characters of each word."

        def transformation(s):
            words = s.split()
            obfuscated_words = list(map(self.delimiter.join, words))
            return " ".join(obfuscated_words)
        self._f = transformation

        def inverse_transformation(s):
            obfuscated_words = s.split()
            words = list(map(lambda word: "".join(word.split(self.delimiter)), obfuscated_words))
            return " ".join(words)
        self._inverse = inverse_transformation
    
    @classmethod
    def construct(cls, choice=None):
        if choice is None:
            random_delimiter = random.choice(DELIMITER_OPTIONS)
            return cls(delimiter=random_delimiter)
        return cls(delimiter=choice)
    
    def __str__(self):
        return self.name + " " + self.delimiter
    
@dataclass
class PrefixRotation(StringTransformation):
    # WARNING: could get confusing with punctuation, like commas and emdashes
    name = "prefix rotation"
    instruction = "Change the string by moving each word's first three characters to the end of the word; leave any word unchanged that is three characters or less."

    def __post_init__(self):
        def transformation(s):
            words = s.split()
            rotated_words = []
            for word in words:
                if len(word) > 3:
                    rotated_word = word[3:] + word[:3]
                else:
                    rotated_word = word
                rotated_words.append(rotated_word)
            return " ".join(rotated_words)
        
        self._f = transformation
        
        def inverse_transformation(s):
            words = s.split()
            original_words = []
            for word in words:
                if len(word) > 3:
                    original_word = word[-3:] + word[:-3]
                else:
                    original_word = word
                original_words.append(original_word)
            return " ".join(original_words)
        
        self._inverse = inverse_transformation

# ~~~~~~~~~~~~~~~~~~~~
# Transforms that I separated from the above for different "vibes"; maybe some structure/function-calling encapsulation, maybe inherent to GPT models, etc.
# These can't be composed willy-nilly with the above transformations; will need to use these in special cases only, i.e. only as the outermost transform in a composition?
# markdown, json
# ~~~~~~~~~~~~~~~~~~~~

@dataclass
class PythonMarkdown(StringTransformation):
    # WARNING: relies on how the given model interprets Python markdown; possibly need per-model casework
    model_type: InitVar[type]
    name = "Python markdown"
    instruction = "Change the string to be in Python in a markdown format."

    def __post_init__(self, model_type):
        if issubclass(model_type, (GPTFamily, ClaudeFamily)):
            # currently, just removes the ```python {message} ``` scaffolding, and removes the multiline string quotations that are often put around the message
            def transformation(s):
                markdown_s = '```python\n"""' + s + '"""\n```'
                return markdown_s
                
            self._f = transformation

            def inverse_transformation(s):
                start_tags, end_tag = ("```python\n", "```markdown\n"), "\n```"
                start_tag_idxs = [s.find(start_tag) for start_tag in start_tags]
                end_tag_idx = s.rfind(end_tag)
                if all(start_tag_idx == -1 for start_tag_idx in start_tag_idxs) or end_tag_idx == -1: #style was not formed correctly... just bypass
                    return s
                start_tag_idx, start_tag = next(st for st in zip(start_tag_idxs, start_tags) if st[0] != -1)
                stripped_s = s[:start_tag_idx] + s[start_tag_idx+len(start_tag):end_tag_idx] + s[end_tag_idx+len(end_tag):]
                stripped_s = stripped_s.replace('"""', '')
                return stripped_s
            
            self._inverse = inverse_transformation
        else:
            raise NotImplementedError
    
    @classmethod
    def construct(cls, model_type=GPTFamily):
        return cls(model_type=model_type)

@dataclass
class JSON_Encapsulation(StringTransformation):
    # WARNING: relies on how the given model interprets json encapsulation; possibly need per-model casework
    name = "JSON encapsulation"
    instruction = "Change the string to be encapsulated in a json format. Output only the json."

    def __post_init__(self):
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
            try:
                beginning_of_json = "{\"message\":"
                assert s.find(beginning_of_json) != -1
                truncate_s = s[s.find(beginning_of_json):]
                messages = unpack_dict(json.loads(truncate_s, strict=False))
                unencapsulated_s = "\n".join(messages)
                return unencapsulated_s
            except:
                # try just the json "prefix"; if still no, then bypass
                json_prefixes = ["{\"message\": \"", "{\"message\":\""]
                for json_prefix in json_prefixes:
                    if s.startswith(json_prefix):
                        s_no_json_prefix = s[len(json_prefix):]
                        return s_no_json_prefix
                return s

        self._inverse = inverse_transformation


@dataclass
class LatexMode(StringTransformation):
    # WARNING: relies on how the given model interprets latex markdown; possibly need per-model casework
    model_type: InitVar[type]
    name = "LaTeX"
    instruction = "Change the string to be part of a LaTeX document."

    def __post_init__(self, model_type):
        if issubclass(model_type, (GPTFamily, ClaudeFamily)):
            def transformation(s):
                markdown_s = "```latex\n\\documentclass[12pt]{article}\n\n\\begin{document}\n\n" + s + "\n\n\\end{document}\n```"
                return markdown_s
            
            self._f = transformation

            def inverse_transformation(s):
                def inner_latex_inversion(latexed):
                    latexed_start_tag, latexed_end_tag = "\\begin{document}", "\\end{document}"
                    latexed_start_tag_idx = latexed.find(latexed_start_tag)
                    latexed_end_tag_idx = latexed.rfind(latexed_end_tag)
                    if latexed_start_tag_idx != -1:
                        if latexed_end_tag_idx != -1:
                            latexed = latexed[latexed_start_tag_idx+len(latexed_start_tag):latexed_end_tag_idx]
                        else:
                            latexed = latexed[latexed_start_tag_idx+len(latexed_start_tag):]
                    latexed = re.sub(r"\\(\w+\*?)(?:\[(.*?)\])?(?:\{(.*?)\}:?)?", "", latexed)
                    return latexed

                start_tag, end_tag = "```latex\n", "\n```"
                start_tag_idx = s.find(start_tag)
                end_tag_idx = s.rfind(end_tag)
                if start_tag_idx != -1:
                    before_latex = s[:start_tag_idx]
                    if end_tag_idx != -1: #style was not formed correctly... just bypass
                        latexed = s[start_tag_idx+len(start_tag):end_tag_idx]
                        after_latex = s[end_tag_idx+len(end_tag):]
                    else:
                        latexed = s[start_tag_idx+len(start_tag):]
                        after_latex = ""
                else:
                    yolo_the_whole_string = inner_latex_inversion(s)
                    return yolo_the_whole_string

                latexed = inner_latex_inversion(latexed)

                return before_latex + latexed + after_latex

            self._inverse = inverse_transformation
        else:
            raise NotImplementedError
    
    @classmethod
    def construct(cls, model_type=GPTFamily):
        return cls(model_type=model_type)

# ~~~~~~~~~~~~~~~~~~~~
# Transforms that count as special cases, for whatever reason
# tokenizer-aware reversal, language translation
# ~~~~~~~~~~~~~~~~~~~~

@dataclass
class TokenizerAwareTransformation(StringTransformation):
    """
    Specifies transformations which explicitly use the tokenizer of a given language model.
    """
    openai_model_name: InitVar[str | None]
    open_source_model_name: InitVar[str | None]

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
    
    @classmethod
    def construct(cls, openai_model_name=None, open_source_model_name=None):
        return cls(openai_model_name=openai_model_name, open_source_model_name=open_source_model_name)


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
    instruction: str = field(init=False)

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
    
    @classmethod
    def construct(cls, choice=None):
        if choice is None:
            random_language = random.choice(LANGUAGE_TRANSLATION_OPTIONS)
            return cls(language=random_language)
        return cls(language=choice)
    
    def __str__(self):
        return self.language + " " + "translation"


def remove_caveat_in_haizey_translation(s, caveat_word):
    caveat_cutoff = s.rfind(caveat_word)
    if caveat_cutoff == -1:
        return s
    no_caveat_s = s[:caveat_cutoff]
    return no_caveat_s


@dataclass
class HaizeyLanguageTranslation(NonDeterministicTransformation):
    """
    Translates text input into another specified language, with the ability to translate back.
    Note that translation is nondeterministic.
    Uses a custom Haize model; set up to translate any text, no matter how toxic, without safety interventions.
    """
    language: str
    name = "language translation"
    instruction: str = field(init=False)

    def __post_init__(self):
        self.instruction = f"Translate the string from English to {self.language}."

        self._translator = HaizerMistral()

        self._translation_task = Template(f"TEXT_TO_TRANSLATE: ' $s ' \n\nPlease translate the TEXT_TO_TRANSLATE in the previous paragraphs from $SOURCE_LANG to $TARGET_LANG, returning only the translated text and nothing else. If you need to provide a safety note, please begin the note after the translation, in a new paragraph, starting with the word 'NOTE' in English.\n")
        def transformation(s):
            just_one_unlimited_output = {"n": 1, "max_tokens": 2048}
            translation_prompt = self._translation_task.substitute(s=s, SOURCE_LANG="English", TARGET_LANG=self.language)
            res = self._translator.inference(translation_prompt, **just_one_unlimited_output)[0]
            no_caveat_res = remove_caveat_in_haizey_translation(res, "\nNOTE")
            no_caveat_res = remove_caveat_in_haizey_translation(no_caveat_res, "\nNote")
            return no_caveat_res
        self._f = transformation

        def inverse_transformation(s):
            just_one_unlimited_output = {"n": 1, "max_tokens": 2048}
            translation_prompt = self._translation_task.substitute(s=s, SOURCE_LANG=self.language, TARGET_LANG="English")
            res = self._translator.inference(translation_prompt, **just_one_unlimited_output)[0]
            no_caveat_res = remove_caveat_in_haizey_translation(res, "\nNOTE")
            no_caveat_res = remove_caveat_in_haizey_translation(no_caveat_res, "\nNote")
            return no_caveat_res
        self._inverse = inverse_transformation
    
    @classmethod
    def construct(cls, choice=None):
        if choice is None:
            random_language = random.choice(LANGUAGE_TRANSLATION_OPTIONS)
            return cls(language=random_language)
        return cls(language=choice)
    
    def __str__(self):
        return self.language + " " + "translation"

ALL_TRANSFORMATIONS = [Reversal, PerWordReversal, WordLevelReversal, CaesarCipher, ROT13Cipher, AtbashCipher, BaseN, Binary, Leetspeak, MorseCode, VowelRepetition, AlternatingCase, Palindrome, Delimiters, PrefixRotation, PythonMarkdown, JSON_Encapsulation, LatexMode, LanguageTranslation] # no TokenizerAwareReversal

# NOTE
# The style transformations, Python markdown and json encapsulation, only make sense as the outermost transformation in a composition
# Morse code, BaseN, binary change the string from alphabetical -> numeral chars, so they need to be after all "main" compositions (but can still be before style transformations)
# Language translation only works on intact English text, so can only be used once and as the innermost transformation in a composition
_transformation_priorities = {
    LanguageTranslation: -1,
    Reversal: 0,
    PerWordReversal: 0,
    TokenizerAwareReversal: 0,
    WordLevelReversal: 0,
    CaesarCipher: 0,
    ROT13Cipher: 0,
    AtbashCipher: 0,
    Leetspeak: 0,
    VowelRepetition: 0,
    AlternatingCase: 0,
    Palindrome: 0,
    Delimiters: 0,
    PrefixRotation: 0,
    MorseCode: 1,
    Binary: 1,
    BaseN: 1,
    PythonMarkdown: 2,
    JSON_Encapsulation: 2,
    LatexMode: 2,
}
_REVERSALS = (Reversal, PerWordReversal, TokenizerAwareReversal, WordLevelReversal)
_CIPHERS = (CaesarCipher, ROT13Cipher, AtbashCipher)
_EXTENDERS = (MorseCode, Binary, BaseN)
_STYLES = (PythonMarkdown, JSON_Encapsulation, LatexMode)
def sample_transformations(k: int, allowed_transforms=ALL_TRANSFORMATIONS):
    """
    Call this instead of `random.sample(allowed_transforms, k)`.
    """
    # There are a ton of rules around what compositions are valid or not, so we'll have to hardcode these a lot.
    # Basically, we want to follow the priorities as above, and we don't want to sample multiple of the same "type".
    t_choice = random.choice(allowed_transforms)
    remaining_transformations = copy(allowed_transforms)
    ret = [t_choice, ]
    k -= 1
    
    while k > 0:
        remaining_transformations = [t for t in remaining_transformations if not issubclass(t, t_choice)]
        for transformation_group in (_REVERSALS, _CIPHERS, _STYLES, _EXTENDERS):
            if issubclass(t_choice, transformation_group):
                remaining_transformations = [t for t in remaining_transformations if not issubclass(t, transformation_group)]
        if len(remaining_transformations) == 0:
            raise ValueError("Ran out of transformations to sample. Your `k` is probably too high...")
        t_choice = random.choice(remaining_transformations)
        ret.append(t_choice)
        k -= 1
    ret = sorted(ret, key = lambda t: _transformation_priorities[t])

    return ret

# BASE_N_OPTIONS = [16, 32, 64, 85, 58]
BASE_N_OPTIONS = [64,]

# These language defaults should reflect cultural + geographic diversity while all using the English alphabet (mostly).
LANGUAGE_TRANSLATION_OPTIONS = ["Spanish", "German", "Turkish", "Indonesian", "Swahili"]

# DELIMITER_OPTIONS = ["!", "@", "#", "$", "%"]
DELIMITER_OPTIONS = ["@", ]