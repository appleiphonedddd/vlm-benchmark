from .llava import LlavaModel
from .qwen_vl import QwenVLModel

MODEL_REGISTRY = {
    "llava": LlavaModel,
    "qwen_vl": QwenVLModel,
}

def build_model(model_name: str, **kwargs):
    model_cls = MODEL_REGISTRY.get(model_name.lower())
    if model_cls is None:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(MODEL_REGISTRY.keys())}")
    return model_cls(**kwargs)