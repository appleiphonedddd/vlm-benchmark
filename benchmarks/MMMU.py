import ast
import re
from typing import Dict, Any, List
from datasets import load_dataset
from base.base_dataset import BaseBenchmarkDataset

OPTION_KEYS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
IMAGE_KEYS = [f"image_{i}" for i in range(1, 8)]


class MMMUDataset(BaseBenchmarkDataset):
    """MMMU multi-discipline multimodal understanding benchmark"""

    def __init__(self, data_path: str = "MMMU/MMMU", split: str = "validation", config_name: str = "Accounting"):
        self.config_name = config_name
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
        if self._is_choice(ground_truth):
            predicted_letter = self._extract_choice(prediction)
            correct = float(predicted_letter == str(ground_truth).strip().upper())
        else:
            accepted_answers = self._parse_answers(ground_truth)
            normalized_prediction = self._normalize(prediction)
            correct = float(any(normalized_prediction == self._normalize(ans) for ans in accepted_answers))
        return {"accuracy": correct}

    def aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {"accuracy": 0.0}
        total_accuracy = sum(r["metrics"]["accuracy"] for r in results)
        return {"accuracy": total_accuracy / len(results)}

    @staticmethod
    def _is_choice(ground_truth: Any) -> bool:
        return bool(re.fullmatch(r"[A-J]", str(ground_truth).strip().upper()))

    @staticmethod
    def _extract_choice(prediction: str) -> str:
        prediction = prediction.strip()

        match = re.search(r"\b([A-J])\b", prediction.upper())
        if match:
            return match.group(1)

        return ""

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())

    @staticmethod
    def _parse_answers(ground_truth: Any) -> List[str]:
        text = str(ground_truth).strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return [str(v) for v in parsed]
            except (ValueError, SyntaxError):
                pass
        return [text]
