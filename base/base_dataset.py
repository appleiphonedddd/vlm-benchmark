from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, List, Optional
from tqdm import tqdm
from base.base_model import BaseVLM
from utils.metrics import Accuracy

class BaseBenchmarkDataset(ABC):
    """Abstract base class for all benchmark evaluation datasets"""

    def __init__(self, data_path: str, split: str = "test", metric: Optional[Any] = None):
        self.data_path = data_path
        self.split = split
        self.metric = metric or Accuracy()
        self.data = self.load_data()

    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load dataset, where each sample must be formatted into a standard dictionary:
        {
            "id": sample_id,
            "image": Image / image_path,
            "prompt": formatted_prompt,
            "ground_truth": ground_truth_answer
        }
        """
        pass

    def evaluate_sample(self, prediction: str, ground_truth: Any) -> Dict[str, float]:
        """Compute metrics for a single sample (defaults to self.metric.score)"""
        return self.metric.score(prediction, ground_truth)

    def aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics across all samples (defaults to self.metric.aggregate)"""
        return self.metric.aggregate(results)

    def run_evaluation(self, model: BaseVLM,
                       on_sample: Callable[[Dict[str, Any], Dict[str, float]], None] = None,
                       **gen_kwargs) -> Dict[str, Any]:
        """
        Execute the complete evaluation pipeline

        if given is called after every sample with that sample's
        result and the running aggregate, so callers can report progress
        without this class knowing how results are displayed
        """
        results = []

        for item in tqdm(self.data, desc=f"Evaluating on {self.__class__.__name__}"):
            pred = model.generate(item["image"], item["prompt"], **gen_kwargs)
            metric_score = self.evaluate_sample(pred, item["ground_truth"])

            results.append({
                "id": item.get("id"),
                "prompt": item["prompt"],
                "prediction": pred,
                "ground_truth": item["ground_truth"],
                "metrics": metric_score
            })

            if on_sample:
                on_sample(results[-1], self.aggregate_metrics(results))

        summary = self.aggregate_metrics(results)
        return {"summary": summary, "details": results}