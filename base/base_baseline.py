from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from base.base_model import BaseVLM


class BaseBaseline(ABC):
    """Abstract base class for all baseline methods"""

    def __init__(self, model: Optional[BaseVLM] = None, **kwargs):
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    def predict(self, sample: Dict[str, Any]) -> str:
        """Produce a prediction for a single sample

        Args:
            sample: Standard sample dict as produced by BaseBenchmarkDataset,
                containing at least "image", "prompt", and "ground_truth"

        Returns:
            str: Predicted answer text
        """
        pass

    def batch_predict(self, samples: List[Dict[str, Any]]) -> List[str]:
        """
        Batch prediction interface
        implemented iteratively by default can be overridden for parallel acceleration
        """
        return [self.predict(sample) for sample in samples]
