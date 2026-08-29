from typing import Dict, Any, List
from datasets import load_dataset
from base.base_dataset import BaseBenchmarkDataset
from utils.metrics import accuracy

OPTION_KEYS = ["A", "B", "C", "D"]


class MMBenchDataset(BaseBenchmarkDataset):
    """MMBench multiple-choice VQA benchmark"""

    def __init__(self, data_path: str = "lmms-lab/MMBench", split: str = "dev", config_name: str = "en"):
        self.config_name = config_name
        self.accuracy = accuracy(OPTION_KEYS)
        super().__init__(data_path, split)

    def load_data(self) -> List[Dict[str, Any]]:
        hf_dataset = load_dataset(self.data_path, self.config_name, split=self.split)

        samples = []
        for item in hf_dataset:
            options = {
                key: item[key] for key in OPTION_KEYS
                if isinstance(item.get(key), str) and item[key].strip() and item[key].strip().lower() != "nan"
            }
            option_text = "\n".join(f"{key}. {value}" for key, value in options.items())

            prompt_parts = []
            hint = item.get("hint")
            if isinstance(hint, str) and hint.strip() and hint.strip().lower() != "nan":
                prompt_parts.append(hint)
            prompt_parts.append(item["question"])
            prompt_parts.append(option_text)
            prompt_parts.append("Answer with the letter of the correct option.")
            prompt = "\n".join(prompt_parts)

            samples.append({
                "id": item.get("index"),
                "image": item["image"],
                "prompt": prompt,
                "ground_truth": item["answer"],
                "category": item.get("category"),
            })

        return samples

    def evaluate_sample(self, prediction: str, ground_truth: Any) -> Dict[str, float]:
        return self.accuracy.score(prediction, ground_truth)

    def aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        return self.accuracy.aggregate(results)
