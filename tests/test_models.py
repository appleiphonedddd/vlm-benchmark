import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tests  # Initialize mocks for external dependencies

import torch
from base.base_hf_model import HuggingFaceBaseVLM
from models import build_model, MODEL_REGISTRY
from models.llava import LlavaModel
from models.qwen_vl import QwenVLModel


class TestHuggingFaceModels(unittest.TestCase):
    @patch("transformers.AutoProcessor.from_pretrained")
    @patch("transformers.AutoModelForImageTextToText.from_pretrained")
    def test_llava_initialization_and_generation(self, mock_model_cls, mock_proc_cls):
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_proc_cls.return_value = mock_processor

        # Setup processor and model behaviors
        mock_processor.apply_chat_template.return_value = "formatted prompt"
        mock_inputs = MagicMock()
        mock_inputs.input_ids = [[1, 2, 3]]
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        mock_model.generate.return_value = [[1, 2, 3, 10, 20]]
        mock_processor.batch_decode.return_value = ["A"]

        # Build model via registry
        llava = build_model("llava", model_path="test-llava-path")
        self.assertIsInstance(llava, LlavaModel)
        self.assertIsInstance(llava, HuggingFaceBaseVLM)

        # Check default dtype used
        mock_model_cls.assert_called_once_with(
            "test-llava-path",
            dtype=torch.float16,
            device_map="cuda"
        )

        # Test message formatting
        mock_image = MagicMock()
        messages = llava.build_messages(mock_image, "What is this?")
        self.assertEqual(messages, [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is this?"}
                ]
            }
        ])

        # Test generate pipeline
        output = llava.generate(mock_image, "What is this?")
        self.assertEqual(output, "A")
        mock_processor.batch_decode.assert_called_once()

    @patch("transformers.AutoProcessor.from_pretrained")
    @patch("transformers.AutoModelForImageTextToText.from_pretrained")
    def test_qwen_vl_initialization_and_generation(self, mock_model_cls, mock_proc_cls):
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_proc_cls.return_value = mock_processor

        # Setup processor and model behaviors
        mock_processor.apply_chat_template.return_value = "formatted qwen prompt"
        mock_inputs = MagicMock()
        mock_inputs.input_ids = [[1, 2]]
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        mock_model.generate.return_value = [[1, 2, 99]]
        mock_processor.batch_decode.return_value = ["B"]

        # Build model via registry
        qwen = build_model("qwen_vl", model_path="test-qwen-path")
        self.assertIsInstance(qwen, QwenVLModel)
        self.assertIsInstance(qwen, HuggingFaceBaseVLM)

        # Check default dtype used
        mock_model_cls.assert_called_once_with(
            "test-qwen-path",
            dtype=torch.bfloat16,
            device_map="cuda"
        )

        # Test message formatting
        mock_image = MagicMock()
        messages = qwen.build_messages(mock_image, "Describe image")
        self.assertEqual(messages, [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": mock_image},
                    {"type": "text", "text": "Describe image"}
                ]
            }
        ])

        # Test generate pipeline
        output = qwen.generate(mock_image, "Describe image")
        self.assertEqual(output, "B")

    @patch("transformers.AutoProcessor.from_pretrained")
    @patch("transformers.AutoModelForImageTextToText.from_pretrained")
    def test_huggingface_batch_generate(self, mock_model_cls, mock_proc_cls):
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_model_cls.return_value = mock_model
        mock_proc_cls.return_value = mock_processor

        mock_inputs = MagicMock()
        mock_inputs.input_ids = [[0, 1, 2], [3, 4, 5]]
        mock_inputs.to.return_value = mock_inputs
        mock_processor.return_value = mock_inputs

        mock_model.generate.return_value = [
            [0, 1, 2, 10, 20],
            [3, 4, 5, 30, 40],
        ]
        mock_processor.batch_decode.return_value = ["Ans 1", "Ans 2"]

        llava = build_model("llava", model_path="test-llava-path")
        self.assertEqual(mock_processor.tokenizer.padding_side, "left")

        mock_img1, mock_img2 = MagicMock(), MagicMock()
        outputs = llava.batch_generate([mock_img1, mock_img2], ["P1", "P2"], max_new_tokens=64)

        self.assertEqual(outputs, ["Ans 1", "Ans 2"])
        mock_model.generate.assert_called_once_with(
            **mock_inputs, max_new_tokens=64
        )
        mock_processor.batch_decode.assert_called_once_with(
            [[10, 20], [30, 40]],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # Test empty inputs
        self.assertEqual(llava.batch_generate([], []), [])


if __name__ == "__main__":
    unittest.main()


