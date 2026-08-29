from base.base_baseline import BaseBaseline
from typing import Any, Dict, Optional
from .config import FastVConfig
from .patcher import FastVPatcher

class FastVBaseline(BaseBaseline):
    def __init__(self, model, config: Optional[FastVConfig] = None):
        super().__init__(model)
        self.config = config or FastVConfig()
        self.patcher = FastVPatcher(self.config)
        self.patcher.patch_model(self.model.model)

    def predict(self, sample: Dict[str, Any]) -> str:
        return self.model.generate(sample["image"], sample["prompt"], **self.kwargs)