from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch import Tensor
from typing import Dict, List, Tuple


@dataclass
class LanguageModel(ABC):
    """
    General interface for language models that includes both open-source and black-box.
    """
    def __post_init__(self):
        """
        I use these __post_init__'s (also present in some subclasses) to create a "soft" registry
        for descriptive fields we want our model classes to have.
        Works around the restrictions that dataclasses have with inheritance.
        """
        if not hasattr(self, "name"):
            raise ValueError("Must have field 'name', with a small string name")
        if not hasattr(self, "description"):
            raise ValueError("Must have field 'description'; use this to specify important details like model size, model family, open-source vs. blackbox API, etc.")
        if not hasattr(self, "scale"):
            raise ValueError("Must have field 'scale'; specify parameter count or FLOPS count if known, otherwise None")
        assert isinstance(self.name, str)
        assert isinstance(self.description, str)
        assert isinstance(self.scale, int) or self.scale is None

    @property
    def vocabulary(self) -> List[str]:
        """
        Return the list of all tokens in the tokenizer used by this language model.
        """
        return self._vocabulary

    @vocabulary.setter
    def vocabulary(self, value):
        self._vocabulary = value

    @abstractmethod
    def inference(self, prompt: str | List[Dict], **optional_params) -> List[str]:
        """
        Perform inference on a given prompt.
        
        Args
            prompt: The input string to generate text from.
            optional_params: Customize inference params if not using default.
                available hyperparams will vary between models,
                but we usually care about temperature, max token limit, batch size, num outputs...
        Returns
            continuations: the generated text.
        """
        pass

    @abstractmethod
    def get_output_logits(self, prompt: str | List[Dict], **optional_params) -> List[Tuple[str, Tensor]]:
        """
        Get output logits for the next token given a prompt.
        
        Args
            prompt: The input string to get continuation logits for.
            optional_params: Customize inference params if not using default.
                available hyperparams will vary between models,
                but we usually care about temperature, max token limit, batch size, num outputs...
        Returns
            continuation_logits: The full vector of output logits for the next token given the prompt.
        """
        pass

    @abstractmethod
    def token_from_index(self, index: int) -> str:
        """
        Get the token corresponding to the index as fed through the model's tokenizer.
        """
        pass

    @abstractmethod
    def finetune(self, dataset_path: str, save_name: str):
        """
        Finetune the model on a given dataset.
        """
        pass

    @abstractmethod
    def destroy(self):
        """
        Destroy any large objects in memory; intended to be called when this LanguageModel is completely done with use!
        """
        pass

    def __str__(self):
        return self.__class__.__name__ + "_" + self.name