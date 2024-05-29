from abc import ABC, abstractmethod


class Judge(ABC):
    """
    A judge model that can be called.
    """
    @abstractmethod
    def __call__(self, prompt, response, **kwargs):
        """
        Perform inference on a given prompt.
        
        Args
            prompt: The attack-containing prompt for some target model.
            response: The completion from that target model.
        Returns
            score: (int) a numerical safety score.
            report: (str) judge model output as commentary on the accompanying safety score.
        """
        pass