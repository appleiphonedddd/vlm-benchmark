import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from base.base_model import BaseVLM

class QwenVLModel(BaseVLM):
    def load_model(self):
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=self.kwargs.get("dtype", torch.bfloat16),
            device_map=self.kwargs.get("device_map", "auto"),
            **{k: v for k, v in self.kwargs.items() if k not in ["torch_dtype", "device_map"]}
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def generate(self, image: Image.Image | str, prompt: str, **gen_kwargs) -> str:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
            
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], 
            images=[image], 
            padding=True, 
            return_tensors="pt"
        ).to(self.model.device)

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
            clean_up_tokenization_spaces=False
        )[0]