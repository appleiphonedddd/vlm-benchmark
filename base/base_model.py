from abc import ABC, abstractmethod
from typing import List, Union
from PIL import Image

class BaseVLM(ABC):
    """Abstract base class for all VLM"""

    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        self.model_path = model_path
        self.device = device
        self.kwargs = kwargs
        self.model = None
        self.processor = None
        self.load_model()

    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer"""
        pass

    @abstractmethod
    def generate(self, image: Union[Image.Image, str], prompt: str, **gen_kwargs) -> str:
        """Inference interface for a single image and text prompt

        Args:
            image: PIL.Image object or image file path
            prompt: Text prompt or query
            **gen_kwargs: Generation parameters such as temperature, max_new_tokens, etc

        Returns:
            str: Model response text
        """
        pass

    def batch_generate(self, images: List[Union[Image.Image, str]], prompts: List[str], **gen_kwargs) -> List[str]:
        """Batch inference interface. Defaults to iterative generation, can be overridden for tensor batching.

        Args:
            images: List of PIL.Image objects or image file paths
            prompts: List of text prompts or queries
            **gen_kwargs: Generation parameters

        Returns:
            List[str]: Model response texts
        """
        return [self.generate(img, p, **gen_kwargs) for img, p in zip(images, prompts)]