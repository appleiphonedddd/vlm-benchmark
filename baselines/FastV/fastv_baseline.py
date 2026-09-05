from base.base_baseline import BaseBaseline
from typing import Optional
from .config import FastVConfig
from .patcher import FastVPatcher

class FastVBaseline(BaseBaseline):
    def __init__(self, model, config: Optional[FastVConfig] = None):
        super().__init__(model)
        self.config = config or FastVConfig()
        self.patcher = FastVPatcher(self.config)
        self.patcher.patch_model(self.model.model)