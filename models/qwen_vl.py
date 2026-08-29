import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from base.base_model import BaseVLM

class Qwen2_5_VLModel(BaseVLM):
    def load_model(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def generate(self, image: Image.Image, prompt: str, **gen_kwargs) -> str:
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
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=gen_kwargs.get("max_new_tokens", 128)
            )
            
        generated_ids_trimmed = [
            out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
        ]
        return self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]