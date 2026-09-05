import torch
from abc import abstractmethod
from typing import Any, Dict, List, Union
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from base.base_model import BaseVLM


class HuggingFaceBaseVLM(BaseVLM):
    """Base class for Vision-Language Models implemented using HuggingFace Transformers."""

    default_dtype: torch.dtype = torch.float16
    clean_up_tokenization_spaces: bool = False

    def load_model(self):
        dtype = self.kwargs.get("dtype", self.kwargs.get("torch_dtype", self.default_dtype))
        device_map = self.kwargs.get("device_map", self.device)
        extra_kwargs = {
            k: v for k, v in self.kwargs.items()
            if k not in ["dtype", "torch_dtype", "device_map"]
        }

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map=device_map,
            **extra_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    @abstractmethod
    def build_messages(self, image: Image.Image, prompt: str) -> List[Dict[str, Any]]:
        """Construct chat template messages formatted for the specific VLM."""
        pass

    def process_inputs(self, text: str, image: Image.Image):
        """Preprocess text and image into tensors on the model's device."""
        return self.processor(
            text=text,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

    def generate(self, image: Union[Image.Image, str], prompt: str, **gen_kwargs) -> str:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        messages = self.build_messages(image, prompt)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.process_inputs(text, image)

        default_gen_kwargs = {"max_new_tokens": 128}
        default_gen_kwargs.update(gen_kwargs)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **default_gen_kwargs)

        generated_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=self.clean_up_tokenization_spaces
        )[0]
