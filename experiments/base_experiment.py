from abc import ABC, abstractmethod
from dataclasses import dataclass
import gc
import hashlib
from itertools import chain
import torch

from models.model_class import LanguageModel
from utils.utils import product_dict


@dataclass
class BaseExperiment(ABC):
    """
    Establishes the same abstraction for all experiments:
    - running over a hyperparameter grid
    - API to save and load data
    """
    def __post_init__(self):
        if not hasattr(self, "name"):
            name_value_error = """
            Must have field 'name', with a small string name.
            This name will be used for filepathing, so feel free to write hierarchical names with forward slashes.
            Only inner forward slashes needed: for example, 'tkbk/open_source/test'
            """
            raise ValueError(name_value_error)
        if not hasattr(self, "description"):
            raise ValueError("Must have field 'description'; use this to specify important details.")
        if not hasattr(self, "hyperparameter_grid"):
            raise ValueError("Must have field 'hyperparameter_grid'; use this to specify the full hyperparameter grid.")
        if not hasattr(self, "all_parameters"):
            raise ValueError("Must have field 'all_parameters'; use this to specify important parameters that should be part of the save/load metadata.")
        assert type(self.name) == str
        assert type(self.description) == str
        assert type(self.hyperparameter_grid) == dict
        assert type(self.all_parameters) == list
    
    def run_hyperparameter_grid(self, config=None):
        productable_hyperparams = {
            k: v if type(v) == list else [v, ] for k, v in self.hyperparameter_grid.items()
        }
        all_hyperparam_settings = product_dict(**productable_hyperparams)

        for hyperparam_setting in all_hyperparam_settings:
            self.run(hyperparam_setting, config)
        
    def evaluate_hyperparameter_grid(self, config=None):
        productable_hyperparams = {
            k: v if type(v) == list else [v, ] for k, v in self.hyperparameter_grid.items()
        }
        all_hyperparam_settings = product_dict(**productable_hyperparams)

        for hyperparam_setting in all_hyperparam_settings:
            self.evaluate(hyperparam_setting, config)
    
    @abstractmethod
    def run(self, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    def destroy(self):
        fields_to_delete = [field for field in self.__dict__ if isinstance(getattr(self, field), LanguageModel)]
        for field in fields_to_delete:
            delattr(self, field)
        gc.collect()
        torch.cuda.empty_cache()

    def data_path(self, **runtime_params):
        raw_pathname = self._raw_data_path(**runtime_params)
        pathname = hashlib.sha256(raw_pathname.encode()).hexdigest()
        return pathname[:48]
    
    def _raw_data_path(self, **runtime_params):
        def populate_data_path(param):
            if param in runtime_params:
                value = runtime_params[param]
            else:
                value = getattr(self, param)
            return str(value).replace("/", "_")
        all_parameters_dict = {
            param: populate_data_path(param)
            for param in self.all_parameters
        }
        # the raw path can be arbitrarily long and then lead to OsError filename too long
        # so we just use a slice of the sha256 hash
        raw_pathname = "__".join([f"{k}_{v}" for k, v in sorted(all_parameters_dict.items())])
        return raw_pathname
