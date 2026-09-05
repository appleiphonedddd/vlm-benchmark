import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tests  # Initialize mocks for external dependencies

from base.base_baseline import BaseBaseline
from baselines import build_baseline, BASELINE_REGISTRY
from baselines.FastV.fastv_baseline import FastVBaseline


class TestBaseBaseline(unittest.TestCase):
    def test_decorator_delegation(self):
        mock_model = MagicMock()
        mock_model.generate.return_value = "answer text"
        mock_model.device = "cuda:0"
        mock_model.custom_attribute = "test_val"

        baseline = BaseBaseline(model=mock_model)

        # Test generate delegation
        output = baseline.generate("image.png", "Describe image", max_new_tokens=64)
        self.assertEqual(output, "answer text")
        mock_model.generate.assert_called_once_with("image.png", "Describe image", max_new_tokens=64)

        # Test transparent attribute delegation via __getattr__
        self.assertEqual(baseline.device, "cuda:0")
        self.assertEqual(baseline.custom_attribute, "test_val")

    def test_missing_model_raises_error(self):
        baseline = BaseBaseline(model=None)
        with self.assertRaises(ValueError):
            baseline.generate("image.png", "prompt")


class TestFastVBaseline(unittest.TestCase):
    @patch("baselines.FastV.fastv_baseline.FastVPatcher")
    def test_fastv_baseline_initialization_and_generation(self, mock_patcher_cls):
        mock_patcher = MagicMock()
        mock_patcher_cls.return_value = mock_patcher

        mock_model = MagicMock()
        mock_model.model = MagicMock()
        mock_model.generate.return_value = "fastv answer"

        # Build baseline via registry
        fastv = build_baseline("fastv", model=mock_model)
        self.assertIsInstance(fastv, FastVBaseline)
        self.assertIsInstance(fastv, BaseBaseline)

        # Verify patcher was initialized and patch_model was called
        mock_patcher_cls.assert_called_once()
        mock_patcher.patch_model.assert_called_once_with(mock_model.model)

        # Verify generate works via inherited BaseBaseline method
        output = fastv.generate("img.jpg", "Question?")
        self.assertEqual(output, "fastv answer")
        mock_model.generate.assert_called_once_with("img.jpg", "Question?")


if __name__ == "__main__":
    unittest.main()
