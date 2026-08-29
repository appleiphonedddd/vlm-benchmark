import ast
import re
from typing import Dict, Any, List
from datasets import load_dataset
from base.base_dataset import BaseBenchmarkDataset

OPTION_KEYS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
IMAGE_KEYS = [f"image_{i}" for i in range(1, 8)]


class MMMUProDataset(BaseBenchmarkDataset):
    """MMMU-Pro robust multi-discipline multimodal understanding benchmark"""

    def __init__(self, data_path: str = "MMMU/MMMU_Pro", split: str = "test", config_name: str = "standard (10 options)"):
        self.config_name = config_name
        super().__init__(data_path, split)

    def load_data(self) -> List[Dict[str, Any]]:
        hf_dataset = load_dataset(self.data_path, self.config_name, split=self.split)

        samples = []
        for item in hf_dataset:
            # MMMU-Pro questions may reference several images as <image 1>, <image 2>, ...
            
            image = next((item[key] for key in IMAGE_KEYS if item.get(key) is not None), None)

            options = ast.literal_eval(item["options"])
            option_text = "\n".join(f"{key}. {value}" for key, value in zip(OPTION_KEYS, options))
            prompt = "\n".join([item["question"], option_text, "Answer with the letter of the correct option."])

            samples.append({
                "id": item.get("id"),
                "image": image,
                "prompt": prompt,
                "ground_truth": item["answer"],
                "subject": item.get("subject"),
                "topic_difficulty": item.get("topic_difficulty"),
            })

        return samples

    def evaluate_sample(self, prediction: str, ground_truth: Any) -> Dict[str, float]:
        predicted_letter = self._extract_choice(prediction)
        correct = float(predicted_letter == str(ground_truth).strip().upper())
        return {"accuracy": correct}

    def aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {"accuracy": 0.0}
        total_accuracy = sum(r["metrics"]["accuracy"] for r in results)
        return {"accuracy": total_accuracy / len(results)}

    @staticmethod
    def _extract_choice(prediction: str) -> str:
        prediction = prediction.strip()

        # Prefer a standalone option letter, e.g. "B", "(B)", "B.", "Answer: B"
        match = re.search(r"\b([A-J])\b", prediction.upper())
        if match:
            return match.group(1)

        return ""
