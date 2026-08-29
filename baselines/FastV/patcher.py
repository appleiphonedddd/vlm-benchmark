import torch
from typing import Optional
from .config import FastVConfig
from .ranking import AttentionScoreRanker

class FastVPatcher:
    def __init__(self, config: FastVConfig):
        self.config = config
        self.ranker = AttentionScoreRanker()
        self.pruning_done = False

    def patch_model(self, model: torch.nn.Module):
        target_layer = self._get_layer(model, self.config.k)
        target_layer.register_forward_hook(self._create_hook())

    def _get_layer(self, model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer_idx]
        raise AttributeError("Unsupported model layer structure.")

    def _create_hook(self):
        def hook(module, inputs, outputs):
            pass
        return hook