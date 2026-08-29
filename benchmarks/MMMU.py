import ast
from typing import Dict, Any, List
from datasets import load_dataset
from base.base_dataset import BaseBenchmarkDataset
from utils.metrics import accuracy

OPTION_KEYS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
IMAGE_KEYS = [f"image_{i}" for i in range(1, 8)]


class MMMUDataset(BaseBenchmarkDataset):
    """MMMU multi-discipline multimodal understanding benchmark"""

    def __init__(self, data_path: str = "MMMU/MMMU", split: str = "validation", config_name: str = "Accounting"):
        self.config_name = config_name
        self.accuracy = accuracy(OPTION_KEYS)
        super().__init__(data_path, split)

    def load_data(self) -> List[Dict[str, Any]]:
        hf_dataset = load_dataset(self.data_path, self.config_name, split=self.split)

        samples = []
        for item in hf_dataset:
            # MMMU questions may reference several images as <image 1>, <image 2>, ...
            image = next((item[key] for key in IMAGE_KEYS if item.get(key) is not None), None)

            question_type = item.get("question_type")
            prompt_parts = [item["question"]]

            if question_type == "multiple-choice":
                options = ast.literal_eval(item["options"])
                option_text = "\n".join(f"{key}. {value}" for key, value in zip(OPTION_KEYS, options))
                prompt_parts.append(option_text)
                prompt_parts.append("Answer with the letter of the correct option.")
            else:
                prompt_parts.append("Answer with a short, exact response.")

            prompt = "\n".join(prompt_parts)

            samples.append({
                "id": item.get("id"),
                "image": image,
                "prompt": prompt,
                "ground_truth": item["answer"],
                "question_type": question_type,
                "subfield": item.get("subfield"),
            })

        return samples

    def evaluate_sample(self, prediction: str, ground_truth: Any) -> Dict[str, float]:
        return self.accuracy.score(prediction, ground_truth)

    def aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        return self.accuracy.aggregate(results)
