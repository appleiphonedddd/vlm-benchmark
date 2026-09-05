import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tests  # Initialize mocks for external dependencies

from base.base_dataset import BaseBenchmarkDataset
from base.base_model import BaseVLM
from utils.metrics import Accuracy, OPTION_KEYS


class DummyDataset(BaseBenchmarkDataset):
    def load_data(self):
        return [
            {"id": 1, "image": None, "prompt": "Question 1", "ground_truth": "A"},
            {"id": 2, "image": None, "prompt": "Question 2", "ground_truth": "B"},
        ]


class TestBaseBenchmarkDataset(unittest.TestCase):
    def test_default_evaluation_and_aggregation(self):
        dataset = DummyDataset(data_path="dummy")
        
        # Test Accuracy metric instance
        self.assertIsInstance(dataset.metric, Accuracy)

        # Test evaluate_sample
        score1 = dataset.evaluate_sample("A", "A")
        self.assertEqual(score1, {"accuracy": 1.0})
        
        score2 = dataset.evaluate_sample("The answer is B", "B")
        self.assertEqual(score2, {"accuracy": 1.0})

        score3 = dataset.evaluate_sample("C", "A")
        self.assertEqual(score3, {"accuracy": 0.0})

        # Test aggregate_metrics
        results = [
            {"metrics": {"accuracy": 1.0}},
            {"metrics": {"accuracy": 0.0}},
        ]
        agg = dataset.aggregate_metrics(results)
        self.assertAlmostEqual(agg["accuracy"], 0.5)

    def test_base_vlm_batch_generate(self):
        class ConcreteVLM(BaseVLM):
            def load_model(self):
                pass
            def generate(self, image, prompt, **gen_kwargs):
                return f"ans_{prompt}"

        vlm = ConcreteVLM(model_path="dummy")
        results = vlm.batch_generate(["img1", "img2"], ["p1", "p2"])
        self.assertEqual(results, ["ans_p1", "ans_p2"])

    def test_run_evaluation(self):
        dataset = DummyDataset(data_path="dummy")
        mock_model = MagicMock()
        mock_model.generate.side_effect = ["A", "C"]  # 1st correct, 2nd wrong

        on_sample_calls = []
        def reporter(sample, running_agg):
            on_sample_calls.append((sample["id"], sample["metrics"]["accuracy"]))

        eval_res = dataset.run_evaluation(mock_model, on_sample=reporter, batch_size=1)

        self.assertEqual(len(eval_res["details"]), 2)
        self.assertAlmostEqual(eval_res["summary"]["accuracy"], 0.5)
        self.assertEqual(len(on_sample_calls), 2)
        self.assertEqual(on_sample_calls[0], (1, 1.0))
        self.assertEqual(on_sample_calls[1], (2, 0.0))

    def test_run_evaluation_batched(self):
        dataset = DummyDataset(data_path="dummy")
        mock_model = MagicMock()
        mock_model.batch_generate.return_value = ["A", "B"]  # both correct

        on_sample_calls = []
        def reporter(sample, running_agg):
            on_sample_calls.append((sample["id"], sample["metrics"]["accuracy"]))

        eval_res = dataset.run_evaluation(mock_model, on_sample=reporter, batch_size=2)

        mock_model.batch_generate.assert_called_once_with(
            [None, None], ["Question 1", "Question 2"]
        )
        self.assertEqual(len(eval_res["details"]), 2)
        self.assertAlmostEqual(eval_res["summary"]["accuracy"], 1.0)
        self.assertEqual(len(on_sample_calls), 2)
        self.assertEqual(on_sample_calls[0], (1, 1.0))
        self.assertEqual(on_sample_calls[1], (2, 1.0))

        # Test invalid batch_size
        with self.assertRaises(ValueError):
            dataset.run_evaluation(mock_model, batch_size=0)


    def test_custom_metric(self):
        custom_metric = MagicMock()
        custom_metric.score.return_value = {"custom_score": 42.0}
        custom_metric.aggregate.return_value = {"mean_custom": 42.0}

        dataset = DummyDataset(data_path="dummy", metric=custom_metric)
        self.assertEqual(dataset.evaluate_sample("pred", "gt"), {"custom_score": 42.0})
        self.assertEqual(dataset.aggregate_metrics([]), {"mean_custom": 42.0})


class TestBenchmarkSubclasses(unittest.TestCase):
    def setUp(self):
        # Mock datasets module in case it's not installed in the testing environment
        if "datasets" not in sys.modules:
            mock_datasets = MagicMock()
            sys.modules["datasets"] = mock_datasets

    def test_mmbench_subclass(self):
        from benchmarks.MMBench import MMBenchDataset
        with patch("benchmarks.MMBench.load_dataset") as mock_load:
            mock_load.return_value = [
                {
                    "index": 1,
                    "image": "img1.png",
                    "hint": "hint text",
                    "question": "What color is the apple?",
                    "A": "Red",
                    "B": "Green",
                    "C": "Blue",
                    "D": "Yellow",
                    "answer": "A",
                    "category": "color",
                }
            ]
            dataset = MMBenchDataset()
            self.assertEqual(len(dataset.data), 1)
            self.assertEqual(dataset.data[0]["ground_truth"], "A")
            self.assertIn("Red", dataset.data[0]["prompt"])

            # Verify evaluation methods work via base class inheritance
            score = dataset.evaluate_sample("A", "A")
            self.assertEqual(score, {"accuracy": 1.0})
            agg = dataset.aggregate_metrics([{"metrics": score}])
            self.assertEqual(agg, {"accuracy": 1.0})

    def test_mmmu_subclass(self):
        from benchmarks.MMMU import MMMUDataset
        with patch("benchmarks.MMMU.load_dataset") as mock_load:
            mock_load.return_value = [
                {
                    "id": "val_1",
                    "image_1": "img1.png",
                    "question_type": "multiple-choice",
                    "options": "['Option 1', 'Option 2']",
                    "question": "Sample Question",
                    "answer": "B",
                    "subfield": "Finance",
                }
            ]
            dataset = MMMUDataset()
            self.assertEqual(len(dataset.data), 1)
            self.assertEqual(dataset.data[0]["ground_truth"], "B")

            # Verify evaluation methods work via base class inheritance
            score = dataset.evaluate_sample("Answer is B", "B")
            self.assertEqual(score, {"accuracy": 1.0})
            agg = dataset.aggregate_metrics([{"metrics": score}])
            self.assertEqual(agg, {"accuracy": 1.0})

    def test_mmmupro_subclass(self):
        from benchmarks.MMMUPro import MMMUProDataset
        with patch("benchmarks.MMMUPro.load_dataset") as mock_load:
            mock_load.return_value = [
                {
                    "id": "pro_1",
                    "image_1": "img1.png",
                    "options": "['Option 1', 'Option 2', 'Option 3']",
                    "question": "Pro Question",
                    "answer": "C",
                    "subject": "Math",
                    "topic_difficulty": "Hard",
                }
            ]
            dataset = MMMUProDataset()
            self.assertEqual(len(dataset.data), 1)
            self.assertEqual(dataset.data[0]["ground_truth"], "C")

            # Verify evaluation methods work via base class inheritance
            score = dataset.evaluate_sample("C", "C")
            self.assertEqual(score, {"accuracy": 1.0})
            agg = dataset.aggregate_metrics([{"metrics": score}])
            self.assertEqual(agg, {"accuracy": 1.0})


class TestCliArguments(unittest.TestCase):
    def test_parse_args_batch_size(self):
        import eval as eval_module
        test_args = [
            "eval.py",
            "--model", "llava",
            "--model_path", "test/path",
            "--benchmark", "mmbench",
            "--batch_size", "4"
        ]
        with patch("sys.argv", test_args):
            parsed = eval_module.parse_args()
            self.assertEqual(parsed.batch_size, 4)
            self.assertEqual(parsed.model, "llava")

        default_args = [
            "eval.py",
            "--model", "llava",
            "--model_path", "test/path",
            "--benchmark", "mmbench",
        ]
        with patch("sys.argv", default_args):
            parsed = eval_module.parse_args()
            self.assertEqual(parsed.batch_size, 1)


if __name__ == "__main__":
    unittest.main()

