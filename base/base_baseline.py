from abc import ABC
from typing import Any, Optional, Union
from PIL import Image
from base.base_model import BaseVLM


class BaseBaseline(ABC):
    """Abstract base class for all baseline methods (Model Decorator).

    Wraps a BaseVLM instance so that the wrapped model is a drop-in
    replacement in evaluation pipelines, conforming to the BaseVLM.generate interface.
    """

    def __init__(self, model: Optional[BaseVLM] = None, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def generate(self, image: Union[Image.Image, str], prompt: str, **gen_kwargs) -> str:
        """Inference interface matching BaseVLM.generate.
        Delegates generation to the wrapped model by default.
        """
        if self.model is None:
            raise ValueError("No model attached to this baseline.")
        return self.model.generate(image, prompt, **{**self.kwargs, **gen_kwargs})

    def __getattr__(self, name: str) -> Any:
        """Transparently delegate attribute and method access to the underlying model."""
        return getattr(self.model, name)
