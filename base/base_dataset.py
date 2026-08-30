from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, List
from tqdm import tqdm
from base.base_model import BaseVLM

class BaseBenchmarkDataset(ABC):
    """Abstract base class for all benchmark evaluation datasets"""

    def __init__(self, data_path: str, split: str = "test"):
        self.data_path = data_path
        self.split = split
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

    @abstractmethod
    def evaluate_sample(self, prediction: str, ground_truth: Any) -> Dict[str, float]:
        """
        Compute metrics for a single sample
        """
        pass

    def run_evaluation(self, model: BaseVLM, limit: int = None,
                       on_sample: Callable[[Dict[str, Any], Dict[str, float]], None] = None,
                       **gen_kwargs) -> Dict[str, Any]:
        """
        Execute the complete evaluation pipeline

        if given is called after every sample with that sample's
        result and the running aggregate, so callers can report progress
        without this class knowing how results are displayed
        """
        samples = self.data[:limit] if limit else self.data
        results = []

        for item in tqdm(samples, desc=f"Evaluating on {self.__class__.__name__}"):
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

    @abstractmethod
    def aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate metrics across all samples"""
        pass