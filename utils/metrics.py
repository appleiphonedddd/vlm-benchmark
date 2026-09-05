import ast
import re
from typing import Any, Dict, List

OPTION_KEYS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


class Accuracy:
    """Accuracy metric shared across benchmarks.

    Supports both multiple-choice (letter) grading and free-text exact-match
    grading against one or more accepted answers.
    """

    def __init__(self, option_keys: List[str] = OPTION_KEYS):
        self.option_keys = option_keys
        self._choice_pattern = re.compile(rf"\b([{''.join(option_keys)}])\b")
        self._choice_only_pattern = re.compile(rf"[{''.join(option_keys)}]")

    def score(self, prediction: str, ground_truth: Any) -> Dict[str, float]:
        """Compute the accuracy of a single prediction against its ground truth"""
        if self.is_choice(ground_truth):
            predicted_letter = self.extract_choice(prediction)
            correct = float(predicted_letter == str(ground_truth).strip().upper())
        else:
            accepted_answers = self.parse_answers(ground_truth)
            normalized_prediction = self.normalize(prediction)
            correct = float(any(normalized_prediction == self.normalize(ans) for ans in accepted_answers))
        return {"accuracy": correct}

    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate per-sample accuracy scores into a final accuracy"""
        if not results:
            return {"accuracy": 0.0}
        total_accuracy = sum(r["metrics"]["accuracy"] for r in results)
        return {"accuracy": total_accuracy / len(results)}

    def is_choice(self, ground_truth: Any) -> bool:
        return bool(self._choice_only_pattern.fullmatch(str(ground_truth).strip().upper()))

    def extract_choice(self, prediction: str) -> str:
        prediction = prediction.strip()

        match = self._choice_pattern.search(prediction.upper())
        if match:
            return match.group(1)

        return ""

    @staticmethod
    def normalize(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())

    @staticmethod
    def parse_answers(ground_truth: Any) -> List[str]:
        text = str(ground_truth).strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return [str(v) for v in parsed]
            except (ValueError, SyntaxError):
                pass
        return [text]


# Backward-compatibility alias
accuracy = Accuracy
