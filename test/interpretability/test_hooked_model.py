import unittest
import torch
from easyroutine.interpretability.hooked_model import HookedModel, HookedModelConfig
from transformers import (
    GenerationConfig,
)


class TestHookedModel(unittest.TestCase):
    def setUp(self):
        config = HookedModelConfig(
            model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
            device_map="balanced",
            torch_dtype=torch.float32,
            attn_implementation="eager",
            batch_size=1,
        )
        self.model = HookedModel(config)

    def test_device(self):
        device = self.model.device()
        self.assertEqual(device.type, "cuda")

    def test_to_string_tokens(self):
        tokens = torch.tensor([101, 102])
        string_tokens = self.model.to_string_tokens(tokens)
        self.assertEqual(len(string_tokens), 2)

    def test_forward_without_split_positions(self):
        inputs = {
            "input_ids": torch.tensor([[101, 102]], device=self.model.device()),
            "attention_mask": torch.tensor([[1, 1]], device=self.model.device()),
        }
        extracted_token_position = ["last"]
        cache = self.model.forward(inputs, extracted_token_position)
        self.assertIn("logits", cache)

    def test_forward_with_split_positions(self):
        inputs = {
            "input_ids": torch.tensor(
                [[101, 102, 103, 104, 105, 106]], device=self.model.device()
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1]], device=self.model.device()
            ),
        }
        extracted_token_position = ["position-group-0"]
        cache = self.model.forward(
            inputs, extracted_token_position, split_positions=[4], extract_resid_out=True
        )
        # assert that cache["resid_out_0"] has shape (1,3,16)
        self.assertIn("resid_out_0", cache)
        self.assertEqual(cache["resid_out_0"].shape, (1, 4, 16))
        
        extracted_token_position = ["position-group-1"]
        cache = self.model.forward(
            inputs, extracted_token_position, split_positions=[4], extract_resid_out=True
        )
        # assert that cache["resid_out_0"] has shape (1,2,16)
        self.assertIn("resid_out_0", cache)
        self.assertEqual(cache["resid_out_0"].shape, (1, 2, 16))

    def test_hook_resid_out(self):
        inputs = {
            "input_ids": torch.tensor(
                [[101, 102, 103, 104, 105, 106]], device=self.model.device()
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1]], device=self.model.device()
            ),
        }
        extracted_token_position = ["position-group-0"]
        cache = self.model.forward(
            inputs, extracted_token_position, split_positions=[4], extract_resid_out=True
        )
        # assert that cache["resid_out_0"] has shape (1,3,16)
        self.assertIn("resid_out_0", cache)
        self.assertEqual(cache["resid_out_0"].shape, (1, 4, 16))
        
    def test_hook_resid_in(self):
        inputs = {
            "input_ids": torch.tensor(
                [[101, 102, 103, 104, 105, 106]], device=self.model.device()
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1]], device=self.model.device()
            ),
        }
        extracted_token_position = ["position-group-0"]
        cache = self.model.forward(
            inputs, extracted_token_position, split_positions=[4], extract_resid_in=True
        )
        # assert that cache["resid_in_0"] has shape (1,3,16)
        self.assertIn("resid_in_0", cache)
        self.assertEqual(cache["resid_in_0"].shape, (1, 4, 16))
        
    def test_hook_resid_mid(self):
        inputs = {
            "input_ids": torch.tensor(
                [[101, 102, 103, 104, 105, 106]], device=self.model.device()
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1]], device=self.model.device()
            ),
        }
        extracted_token_position = ["position-group-0"]
        cache = self.model.forward(
            inputs, extracted_token_position, split_positions=[4], extract_resid_mid=True
        )
        # assert that cache["resid_mid_0"] has shape (1,3,16)
        self.assertIn("resid_mid_0", cache)
        self.assertEqual(cache["resid_mid_0"].shape, (1, 4, 16))
        


if __name__ == "__main__":
    unittest.main()
