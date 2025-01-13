import unittest
import torch
from torch.utils.data import DataLoader, Dataset
from easyroutine.interpretability.hooked_model import HookedModel, HookedModelConfig
from easyroutine.interpretability.activation_cache import ActivationCache


class TestHookedModel(unittest.TestCase):
    def setUp(self):
        """
        Set the config and load a tiny llama model for testing
        """
        config = HookedModelConfig(
            model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
            device_map="balanced",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            batch_size=1,
        )
        self.model = HookedModel(config)

    def test_device(self):
        device = self.model.device()
        self.assertEqual(device.type, "cuda")

    def test_to_string_tokens(self):
        """
        Test the conversion of tokens to string tokens
        """
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
            inputs,
            extracted_token_position,
            split_positions=[4],
            extract_resid_out=True,
        )
        # assert that cache["resid_out_0"] has shape (1,3,16)
        self.assertIn("resid_out_0", cache)
        self.assertEqual(cache["resid_out_0"].shape, (1, 4, 16))

        extracted_token_position = ["position-group-1"]
        cache = self.model.forward(
            inputs,
            extracted_token_position,
            split_positions=[4],
            extract_resid_out=True,
        )
        # assert that cache["resid_out_0"] has shape (1,2,16)
        self.assertIn("resid_out_0", cache)
        self.assertEqual(cache["resid_out_0"].shape, (1, 2, 16))

    def test_extract_cache(self):
        """
        Test the extract_cache method of HookedModel.
        """

        class CustomDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        dataloader = [
            {
                "input_ids": torch.tensor([[101, 102]], device=self.model.device()),
                "attention_mask": torch.tensor([[1, 1]], device=self.model.device()),
            },
            {
                "input_ids": torch.tensor([[103, 104]], device=self.model.device()),
                "attention_mask": torch.tensor([[1, 1]], device=self.model.device()),
            },
        ]
        # dataset = CustomDataset(data)
        # dataloader = DataLoader(dataset)

        target_token_positions = ["last"]

        def batch_saver(batch):
            return {"batch_info": batch}

        final_cache = self.model.extract_cache(
            dataloader,
            target_token_positions=target_token_positions,
            batch_saver=batch_saver,
            extract_resid_out=True,
        )

        self.assertIn("logits", final_cache)
        self.assertIn("resid_out_0", final_cache)
        self.assertIn("mapping_index", final_cache)
        self.assertIn("example_dict", final_cache)
        self.assertTrue(torch.is_tensor(final_cache["logits"]))


class TestHookFromHookedModel(unittest.TestCase):
    def setUp(self):
        """
        Set the config and load a tiny llama model for testing
        """
        config = HookedModelConfig(
            model_name="hf-internal-testing/tiny-random-LlamaForCausalLM",
            device_map="balanced",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            batch_size=1,
        )
        self.model = HookedModel(config)

        self.inputs = {
            "input_ids": torch.tensor(
                [[101, 102, 103, 104, 105, 106]], device=self.model.device()
            ),
            "attention_mask": torch.tensor(
                [[1, 1, 1, 1, 1]], device=self.model.device()
            ),
        }

        self.extracted_token_position = ["position-group-0"]

    def test_hook_resid_out(self):
        cache = self.model.forward(
            self.inputs,
            self.extracted_token_position,
            split_positions=[4],
            extract_resid_out=True,
        )
        # assert that cache["resid_out_0"] has shape (1,3,16)
        self.assertIn("resid_out_0", cache)
        self.assertEqual(cache["resid_out_0"].shape, (1, 4, 16))

    def test_hook_resid_in(self):
        cache = self.model.forward(
            self.inputs,
            self.extracted_token_position,
            split_positions=[4],
            extract_resid_in=True,
        )
        # assert that cache["resid_in_0"] has shape (1,3,16)
        self.assertIn("resid_in_0", cache)
        self.assertEqual(cache["resid_in_0"].shape, (1, 4, 16))

    def test_hook_resid_mid(self):
        cache = self.model.forward(
            self.inputs,
            self.extracted_token_position,
            split_positions=[4],
            extract_resid_mid=True,
        )
        # assert that cache["resid_mid_0"] has shape (1,3,16)
        self.assertIn("resid_mid_0", cache)
        self.assertEqual(cache["resid_mid_0"].shape, (1, 4, 16))

    def test_hook_extract_attn_in(self):
        cache = self.model.forward(
            self.inputs,
            self.extracted_token_position,
            split_positions=[4],
            extract_attn_in=True,
        )
        # assert that cache["attn_in_0"] has shape (1, 4, )
        self.assertIn("attn_in_0", cache)
        self.assertEqual(cache["attn_in_0"].shape, (1, 4, 16))

    def test_hook_extract_attn_out(self):
        cache = self.model.forward(
            self.inputs,
            self.extracted_token_position,
            split_positions=[4],
            extract_attn_out=True,
        )
        # assert that cache["attn_out_0"] has shape (1, 4, )
        self.assertIn("attn_out_0", cache)
        self.assertEqual(cache["attn_out_0"].shape, (1, 4, 16))

    def test_hook_extract_avg_attn_pattern(self):
        external_cache = ActivationCache()
        external_cache["avg_pattern_L1H1"] = torch.randn(1, 6, 6)
        cache = self.model.forward(
            self.inputs,
            self.extracted_token_position,
            split_positions=[4],
            extract_avg_attn_pattern=True,
            external_cache=external_cache,
            batch_idx=1
        )
        # assert that cache["avg_attn_pattern_0"] has shape (1, 4, 16, 16)
        self.assertIn("avg_pattern_L1H1", external_cache)
        self.assertEqual(external_cache["avg_pattern_L1H1"].shape, (1, 6, 6))

    def test_hook_extract_attn_pattern(self):
        cache = self.model.forward(
            self.inputs,
            self.extracted_token_position,
            split_positions=[4],
            extract_attn_pattern=True,
        )
        # assert that cache["attn_pattern_0"] has shape (1, 4, 16, 16)
        self.assertIn("pattern_L1H1", cache)
        self.assertEqual(cache["pattern_L1H1"].shape, (1, 6, 6))

    # def test_hook_extract_head_out(self):
    #     cache = self.model.forward(
    #         self.inputs, self.extracted_token_position, split_positions=[4], extract_head_out=True
    #     )
    #     # assert that cache["head_out_0"] has shape (1, 4, 16)
    #     self.assertIn("head_out_L1H1", cache)
    #     self.assertEqual(cache["head_out_L1H1"].shape, (1,4,16))

    # TODO: Add test for all the extraction
    # TODO: Add test for the ablation
    # TODO: Add test for patching


if __name__ == "__main__":
    unittest.main()
