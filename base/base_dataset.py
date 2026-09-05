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
                       on_sample: Optional[Callable[[Dict[str, Any], Dict[str, float]], None]] = None,
                       batch_size: int = 1,
                       **gen_kwargs) -> Dict[str, Any]:
        """
        Execute the complete evaluation pipeline

        Args:
            model: Model to evaluate
            on_sample: If given, called after every sample with that sample's
                       result and the running aggregate
            batch_size: Number of samples to process per batch (default: 1)
            **gen_kwargs: Additional generation parameters
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")

        results = []
        batches = [self.data[i:i + batch_size] for i in range(0, len(self.data), batch_size)]

        for batch_items in tqdm(batches, desc=f"Evaluating on {self.__class__.__name__}"):
            batch_images = [item["image"] for item in batch_items]
            batch_prompts = [item["prompt"] for item in batch_items]

            if batch_size == 1:
                preds = [model.generate(batch_images[0], batch_prompts[0], **gen_kwargs)]
            else:
                preds = model.batch_generate(batch_images, batch_prompts, **gen_kwargs)

            for item, pred in zip(batch_items, preds):
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