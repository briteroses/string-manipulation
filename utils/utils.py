from itertools import product
import numpy as np
from typing_extensions import assert_never

from models.model_class import LanguageModel
from models.black_box_model import BlackBoxModel
from models.open_source_model import OpenSourceModel


def product_dict(**kwargs):
    nonempty_kwargs = {k: v for k, v in kwargs.items() if not (type(v) is list and len(v) == 0)}
    keys = nonempty_kwargs.keys()
    for instance in product(*nonempty_kwargs.values()):
        yield dict(zip(keys, instance))

def linear_probability(logprob):
    return np.round(np.exp(logprob)*100,2)

def get_max_tokens_key(model: LanguageModel):
    match model:
        case BlackBoxModel():
            return "max_tokens"
        case OpenSourceModel():
            return "max_new_tokens"
        case _ as unreachable:
            assert_never(unreachable)

class RegexTools():
    pass #TODO

class StringManipulationUtils():
    pass #TODO