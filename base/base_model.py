from abc import ABC, abstractmethod
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
    def generate(self, image: Image.Image | str, prompt: str, **gen_kwargs) -> str:
        """Inference interface for a single image and text prompt

        Args:
            image: PIL.Image object or image file path
            prompt: Text prompt or query
            **gen_kwargs: Generation parameters such as temperature, max_new_tokens, etc

        Returns:
            str: Model response text
        """
        pass