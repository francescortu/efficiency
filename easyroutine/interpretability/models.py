from dataclasses import dataclass
from easyroutine.logger import LambdaLogger
from transformers import (
    ChameleonProcessor,
    ChameleonForConditionalGeneration,
    LlavaForConditionalGeneration,
    PixtralProcessor,
    LlamaForCausalLM,
    LlamaTokenizer
)
import random
from typing import List, Literal, Union, Dict, Optional
import torch


@dataclass
class ModelConfig:
    r"""
    Model configuration class for storing model specific parameters.
    """

    residual_stream_input_hook_name: str
    residual_stream_hook_name: str
    intermediate_stream_hook_name: str
    attn_value_hook_name: str
    attn_in_hook_name: str
    attn_out_hook_name: str
    attn_matrix_hook_name: str

    attn_out_proj_weight: str
    attn_out_proj_bias: str
    embed_tokens: str

    num_hidden_layers: int
    num_attention_heads: int
    hidden_size: int
    num_key_value_heads: int
    num_key_value_groups: int
    head_dim: int


# SPECIFIC MODEL CONFIGURATIONS


class ModelFactory:
    """
    This class is a factory to load the model and the processor. It supports the following models:
    - Chameleon-7b

    TO ADD A NEW MODEL:
    - add the model in the load_model
    """

    @staticmethod
    def load_model(
        model_name: str,
        attn_implementation: str,
        torch_dtype: torch.dtype,
        device_map: str,
    ):
        if attn_implementation != "eager":
            LambdaLogger.log(
                "Using an attention type different from eager could have unexpected beheviour in some experiment!",
                "WARNING",
            )
        if model_name in ["facebook/chameleon-7b", "facebook/chameleon-30b"]:
            model = ChameleonForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )
            model_config = ModelConfig(
                residual_stream_input_hook_name="model.layers[{}].input",
                residual_stream_hook_name="model.layers[{}].output",
                intermediate_stream_hook_name="model.layers[{}].post_attention_layernorm.output",
                attn_value_hook_name="model.layers[{}].self_attn.v_proj.output",
                attn_out_hook_name="model.layers[{}].self_attn.o_proj.output",
                attn_in_hook_name="model.layers[{}].self_attn.input",
                attn_matrix_hook_name="model.layers[{}].self_attn.attention_matrix_hook.output",
                attn_out_proj_weight="model.layers[{}].self_attn.o_proj.weight",
                attn_out_proj_bias="model.layers[{}].self_attn.o_proj.bias",
                embed_tokens="model.embed_tokens.input",
                num_hidden_layers=model.config.num_hidden_layers,
                num_attention_heads=model.config.num_attention_heads,
                hidden_size=model.config.hidden_size,
                num_key_value_heads=model.config.num_key_value_heads,
                num_key_value_groups=model.config.num_attention_heads
                // model.config.num_key_value_heads,
                head_dim=model.config.hidden_size // model.config.num_attention_heads,
            )

        elif model_name in ["mistral-community/pixtral-12b"]:
            model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )

            model_config = ModelConfig(
                residual_stream_input_hook_name="language_model.model.layers[{}].input",
                residual_stream_hook_name="language_model.model.layers[{}].output",
                intermediate_stream_hook_name="language_model.model.layers[{}].post_attention_layernorm.output",
                attn_value_hook_name="language_model.model.layers[{}].self_attn.v_proj.output",
                attn_out_hook_name="language_model.model.layers[{}].self_attn.o_proj.output",
                attn_in_hook_name="language_model.model.layers[{}].self_attn.input",
                attn_matrix_hook_name="language_model.model.layers[{}].self_attn.attention_matrix_hook.output",
                attn_out_proj_weight="language_model.model.layers[{}].self_attn.o_proj.weight",
                attn_out_proj_bias="language_model.model.layers[{}].self_attn.o_proj.bias",
                embed_tokens="language_model.model.embed_tokens.input",
                num_hidden_layers=model.config.text_config.num_hidden_layers,
                num_attention_heads=model.config.text_config.num_attention_heads,
                hidden_size=model.config.text_config.hidden_size,
                num_key_value_heads=model.config.text_config.num_key_value_heads,
                num_key_value_groups=model.config.text_config.num_attention_heads
                // model.config.text_config.num_key_value_heads,
                head_dim=model.config.text_config.head_dim,
            )
        elif model_name in ["Emu3-Chat", "Emu3-Gen", "Emu3-Stage1"]:
            raise NotImplementedError("Emu3 model not implemented yet")
            # TODO: Implement Emu3 model with the new model loading from huggingface transformers
            model_config = ModelConfig(
                residual_stream_input_hook_name="model.layers[{}].input",
                residual_stream_hook_name="model.layers[{}].output",
                intermediate_stream_hook_name="model.layers[{}].post_attention_layernorm.output",
                attn_value_hook_name="model.layers[{}].self_attn.v_proj.output",
                attn_out_hook_name="model.layers[{}].self_attn.o_proj.output",
                attn_in_hook_name="model.layers[{}].self_attn.input",
                attn_matrix_hook_name="model.layers[{}].self_attn.attention_matrix_hook.output",
                attn_out_proj_weight="model.layers[{}].self_attn.o_proj.weight",
                attn_out_proj_bias="model.layers[{}].self_attn.o_proj.bias",
                embed_tokens="model.embed_tokens.input",
                num_hidden_layers=model.config.num_hidden_layers,
                num_attention_heads=model.config.num_attention_heads,
                hidden_size=model.config.hidden_size,
                num_key_value_heads=model.config.num_key_value_heads,
                num_key_value_groups=model.config.num_attention_heads
                // model.config.num_key_value_heads,
                head_dim=model.config.hidden_size // model.config.num_attention_heads,
            )
        elif model_name in ["hf-internal-testing/tiny-random-LlamaForCausalLM"]:
            model = LlamaForCausalLM.from_pretrained(
                model_name, torch_dtype=torch_dtype, device_map=device_map
            )
            
            model_config = ModelConfig(
                residual_stream_input_hook_name="model.layers[{}].input",
                residual_stream_hook_name="model.layers[{}].output",
                intermediate_stream_hook_name="model.layers[{}].post_attention_layernorm.output",
                attn_value_hook_name="model.layers[{}].self_attn.v_proj.output",
                attn_out_hook_name="model.layers[{}].self_attn.o_proj.output",
                attn_in_hook_name="model.layers[{}].self_attn.input",
                attn_matrix_hook_name="model.layers[{}].self_attn.attention_matrix_hook.output",
                attn_out_proj_weight="model.layers[{}].self_attn.o_proj.weight",
                attn_out_proj_bias="model.layers[{}].self_attn.o_proj.bias",
                embed_tokens="model.embed_tokens.input",
                num_hidden_layers=model.config.num_hidden_layers,
                num_attention_heads=model.config.num_attention_heads,
                hidden_size=model.config.hidden_size,
                num_key_value_heads=model.config.num_key_value_heads,
                num_key_value_groups=model.config.num_attention_heads
                // model.config.num_key_value_heads,
                head_dim=model.config.hidden_size // model.config.num_attention_heads,
            )

        else:
            raise ValueError("Unsupported model_name")

        return model, model_config


class TokenizerFactory:
    @staticmethod
    def load_tokenizer(model_name: str, torch_dtype: torch.dtype, device_map: str):
        if model_name in ["facebook/chameleon-7b", "facebook/chameleon-30b"]:
            processor = ChameleonProcessor.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            is_a_processor = True
        elif model_name in ["mistral-community/pixtral-12b"]:
            processor = PixtralProcessor.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            is_a_processor = True
        elif model_name in ["Emu3-Chat", "Emu3-Gen", "Emu3-Stage1"]:
            raise NotImplementedError("Emu3 model not implemented yet")
        elif model_name in ["hf-internal-testing/tiny-random-LlamaForCausalLM"]:
            processor = LlamaTokenizer.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            is_a_processor = False
        else:
            raise ValueError("Unsupported model_name")

        return processor, is_a_processor


SUPPORTED_MODELS = {
    "mistral-community/pixtral-12b": ("[IMG]", "[BREAK]", "[IMG_END]"),
    "Emu3-Chat": ("<|image start|>", None, "<|image end|>"),
    "Emu3-Gen": ("<|image start|>", None, "<|image end|>"),
    "Emu3-Stage1": ("<|image start|>", None, "<|image end|>"),
    "facebook/chameleon-7b": ("<racm3:break>", None, "<eoss>"),
    "facebook/chameleon-30b": ("<racm3:break>", None, "<eoss>"),
    "hf-internal-testing/tiny-random-LlamaForCausalLM": (None, None, None)
}

SUPPORTED_TOKENS = [
    "last",
    "last-2",
    "last-3",
    "last-image",
    "end-image",
    "all-image",
    "all-text",
    "all",
    "special",
    "random-text",
    "random-image",
    "random-image-10",
]


class TokenIndex:
    def __init__(
        self,
        model_name: str,
        split_positions: Optional[List[int]] = None,
        split_tokens: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.split_tokens = split_tokens
        self.split_positions = sorted(split_positions) if split_positions else []

    def find_occurrences(self, lst: List[str], target: str) -> List[int]:
        return [i for i, x in enumerate(lst) if x == target]

    def categorize_tokens(self, string_tokens: List[str]) -> Dict[str, List[int]]:
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError("Unsupported model_name")

        start_image_token, special, end_image_token = SUPPORTED_MODELS[self.model_name]

        image_start_tokens, image_end_tokens, image_tokens, last_line_image_tokens = (
            [],
            [],
            [],
            [],
        )
        text_tokens, special_tokens = [], []

        in_image_sequence = False

        for i, token in enumerate(string_tokens):
            if token == start_image_token and not in_image_sequence:
                in_image_sequence = True
                image_start_tokens.append(i)
            elif in_image_sequence and token == end_image_token:
                in_image_sequence = False
                image_end_tokens.append(i)
                last_line_image_tokens.append(i - 1)
            elif in_image_sequence and special and token == special:
                special_tokens.append(i)
            elif in_image_sequence:
                image_tokens.append(i)
            else:
                text_tokens.append(i)

        tokens_group, positions_group = self.group_tokens(string_tokens)

        position_dict = {
            f"position-group-{i}": positions_group[i] for i in positions_group
        }

        return {
            "image_start": image_start_tokens,
            "image_end": image_end_tokens,
            "image": image_tokens,
            "last_line_image": last_line_image_tokens,
            "text": text_tokens,
            "special": special_tokens,
            **position_dict,
        }

    def group_tokens(
        self, string_tokens: List[str]
    ) -> (Dict[int, List[str]], Dict[int, List[int]]):
        if self.split_tokens:
            return self.group_tokens_by_split_tokens(string_tokens)
        elif self.split_positions:
            return self.group_tokens_by_positions(string_tokens)
        else:
            return {0: string_tokens}, {0: list(range(len(string_tokens)))}

    def group_tokens_by_positions(
        self, string_tokens: List[str]
    ) -> (Dict[int, List[str]], Dict[int, List[int]]):
        tokens_group, positions_group = {}, {}
        for i, pos in enumerate(self.split_positions):
            if i == 0:
                positions_group[i] = [0, pos]
            else:
                positions_group[i] = [self.split_positions[i - 1], pos]
        positions_group[len(self.split_positions)] = [
            self.split_positions[-1],
            len(string_tokens),
        ]
        
        # modify the positions_group to include all the indexes and not just the start and end
        for i in range(len(positions_group)):
            positions_group[i] = list(range(positions_group[i][0], positions_group[i][1]))
            

        for i, group in positions_group.items():
            tokens_group[i] = string_tokens[group[0] : group[1]]

        return tokens_group, positions_group

    def group_tokens_by_split_tokens(
        self, string_tokens: List[str]
    ) -> (Dict[int, List[str]], Dict[int, List[int]]):
        tokens_group, positions_group = {}, {}
        current_group = 0
        start_pos = 0

        for i, token in enumerate(string_tokens):
            if token in self.split_tokens:
                positions_group[current_group] = [start_pos, i]
                tokens_group[current_group] = string_tokens[start_pos:i]
                current_group += 1
                start_pos = i + 1

        positions_group[current_group] = [start_pos, len(string_tokens)]
        tokens_group[current_group] = string_tokens[start_pos:]

        return tokens_group, positions_group

    def get_token_index(
        self,
        tokens: List[str],
        string_tokens: List[str],
        return_type: Literal["list", "int", "dict"] = "list",
    ) -> Union[List[int], int, Dict]:
        if not all(
            token in SUPPORTED_TOKENS
            or token.startswith("position-group-")
            or token.startswith("random-position-group-")
            for token in tokens
        ):
            raise ValueError(
                f"Unsupported token type: {tokens}. Supported tokens are: {SUPPORTED_TOKENS} and position-group-0, position-group-1, etc or random-position-group-0, random-position-group-1, etc"
            )

        # Check if split_positions is required but not provided
        if self.split_positions is None and any(
            token.startswith("position-group-")
            or token.startswith("random-position-group-")
            for token in tokens
        ):
            raise ValueError(
                "split_positions cannot be None when a group position token is requested"
            )

        token_indexes = self.categorize_tokens(string_tokens)
        tokens_positions = self.get_tokens_positions(tokens, token_indexes)

        if return_type == "int":
            if len(tokens_positions) > 1:
                raise ValueError(
                    "More than one token requested: return_type should be list, got int"
                )
            return tokens_positions[0]
        if return_type == "dict":
            return self.get_token_dict(token_indexes)
        return tokens_positions

    def get_tokens_positions(
        self, tokens: List[str], token_indexes: Dict[str, List[int]]
    ) -> List[int]:
        tokens_positions = []
        position_dict = {
            k: v for k, v in token_indexes.items() if k.startswith("position-group-")
        }
        random_position_dict = {
            f"random-{k}": random.sample(v, 1) for k, v in position_dict.items()
        }

        for token in tokens:
            if token.startswith("random-position-group-"):
                group, n = self.parse_random_group_token(token)
                random_position_dict[token] = random.sample(
                    position_dict[f"position-group-{group}"], int(n)
                )
            elif token.startswith("random-image"):
                n = token.split("-")[-1]
                random_position_dict[token] = random.sample(
                    token_indexes["image"], int(n) if n else 1
                )

        token_dict = self.get_token_dict(token_indexes, random_position_dict)

        for token in tokens:
            tokens_positions.extend(token_dict[token])

        return tokens_positions

    def parse_random_group_token(self, token: str) -> (str, int):
        group_and_n = token.split("-")[2:]
        if len(group_and_n) > 1:
            group, n = group_and_n
        else:
            group = group_and_n[0]
            n = 1
        return group, int(n)

    def get_token_dict(
        self,
        token_indexes: Dict[str, List[int]],
        random_position_dict: Dict[str, List[int]] = {},
    ) -> Dict[str, List[int]]:
        return {
            "last": [-1],
            "last-2": [-2],
            "last-4": [-4],
            "last-image": token_indexes["last_line_image"],
            "end-image": token_indexes["image_end"],
            "all-text": token_indexes["text"],
            "all": list(range(len(token_indexes["text"]))),
            "all-image": token_indexes["image"],
            "special": token_indexes["special"],
            "random-text": None if len(token_indexes["text"])==0 else [random.choice(token_indexes["text"])],
            "random-image": None if len(token_indexes["image"])==0 else [random.choice(token_indexes["image"])],
            "special-pixtral": [1052, 1051, 1038, 991, 1037, 1047],
            **{
                k: v
                for k, v in token_indexes.items()
                if k.startswith("position-group-")
            },
            **random_position_dict,
        }
