import unittest
import torch
from transformers import GenerationConfig
from easyroutine.interpretability.hooked_model import HookedModel, HookedModelConfig

# FILE: easyroutine/interpretability/test_hooked_model.py


class TestHookedModel(unittest.TestCase):

    def setUp(self):
        config = HookedModelConfig(
            model_name="bert-base-uncased",
            device_map="cpu",
            torch_dtype=torch.float32,
            attn_implementation="eager",
            batch_size=1
        )
        self.model = HookedModel(config)

    def test_get_tokenizer(self):
        tokenizer = self.model.get_tokenizer()
        self.assertIsNotNone(tokenizer)

    def test_get_text_tokenizer(self):
        text_tokenizer = self.model.get_text_tokenizer()
        self.assertIsNotNone(text_tokenizer)

    def test_get_processor(self):
        processor = self.model.get_processor()
        self.assertIsNone(processor)

    def test_eval(self):
        self.model.eval()
        self.assertFalse(self.model.hf_model.training)

    def test_device(self):
        device = self.model.device()
        self.assertEqual(device.type, "cpu")

    def test_to_string_tokens(self):
        tokens = torch.tensor([101, 102])
        string_tokens = self.model.to_string_tokens(tokens)
        self.assertEqual(string_tokens, ["[CLS]", "[SEP]"])

    def test_forward(self):
        inputs = {
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]])
        }
        extracted_token_position = ["last"]
        cache = self.model.forward(inputs, extracted_token_position)
        self.assertIn("logits", cache)

    def test_generate(self):
        inputs = {
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]])
        }
        generation_config = GenerationConfig(max_length=5)
        output = self.model.generate(inputs, generation_config=generation_config)
        self.assertIsNotNone(output)

if __name__ == "__main__":
    unittest.main()