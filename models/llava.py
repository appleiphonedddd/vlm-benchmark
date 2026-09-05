import torch
from typing import Any, Dict, List
from PIL import Image
from base.base_hf_model import HuggingFaceBaseVLM


class LlavaModel(HuggingFaceBaseVLM):
    """LLaVA Vision-Language Model implementation."""

    default_dtype = torch.float16

    def build_messages(self, image: Image.Image, prompt: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]