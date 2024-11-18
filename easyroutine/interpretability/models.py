from dataclasses import dataclass
from easyroutine.logger import LambdaLogger
from transformers import (
    ChameleonProcessor,
    ChameleonForConditionalGeneration,
    LlavaForConditionalGeneration,
    PixtralProcessor,
)
import random
from typing import List, Literal, Union, Dict
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
            processor = ChameleonProcessor.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
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
            processor = PixtralProcessor.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
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
        else:
            raise ValueError("Unsupported model_name")

        return processor, is_a_processor


class TokenIndex:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def find_occurrences(self, lst: List[str], target: str) -> List[int]:
        r"""
        return a list of the occurrences of the target in the string
        """

        return [i for i, x in enumerate(lst) if x == target]

    def categorize_tokens(self, string_tokens: List[str]):
        """
        Categorize token in disjoin set of tokens:
            - image_start_tokens: list of the index of the start image token
            - image_end_tokens: list of the index of the end image token
            - image_tokens: list of the index of the image tokens
            - text_tokens: list of the index of the text tokens
            - special_tokens: list of the index of the special tokens
        """
        image_start_tokens = []
        image_end_tokens = []
        image_tokens = []
        last_line_image_tokens = []
        text_tokens = []
        special_tokens = []

        if self.model_name == "mistral-community/pixtral-12b":
            start_image_token = "[IMG]"
            special = "[BREAK]"
            end_image_token = "[IMG_END]"
        elif self.model_name in ["Emu3-Chat", "Emu3-Gen", "Emu3-Stage1"]:
            start_image_token = "<|image start|>"
            special = None
            end_image_token = "<|image end|>"
        elif self.model_name in ["facebook/chameleon-7b", "facebook/chameleon-30b"]:
            start_image_token = "<racm3:break>"
            special = None
            end_image_token = "<eoss>"
        else:
            raise ValueError("Unsupported model_name")

        in_image_sequence = False

        for i, token in enumerate(string_tokens):
            # check for the start
            if token == start_image_token and not in_image_sequence:
                in_image_sequence = True
                image_start_tokens.append(i)

            # check for the end
            elif in_image_sequence and token == end_image_token:
                in_image_sequence = False
                image_end_tokens.append(i)
                last_line_image_tokens.append(i - 1)

            # cehck for special tokens
            elif in_image_sequence and special is not None and token == special:
                special_tokens.append(i)

            # check for image tokens
            elif in_image_sequence:
                image_tokens.append(i)

            # check for text tokens
            elif not in_image_sequence:
                text_tokens.append(i)

        return {
            "image_start": image_start_tokens,
            "image_end": image_end_tokens,
            "image": image_tokens,
            "last_line_image": last_line_image_tokens,
            "text": text_tokens,
            "special": special_tokens,
        }

    def get_token_index(
        self,
        tokens: List[str],
        string_tokens: List[str],
        return_type: Literal["list", "int", "dict"] = "list",
    ) -> Union[List[int], int, Dict]:
        """
        Unified method to extract token indices based on the model type (pixtral or emu3).

        Args:
            - tokens (List[str]): List of tokens to extract the activations.
            - string_tokens (List[str]): List of string tokens of the input.
            - return_type (Literal): Type of return, either "list", "int", or "dict".

        Returns:
            - Token positions based on the specified return_type.
        """
        if tokens not in [
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
        ]:
            raise ValueError(
                f"Unsupported token type: {tokens}. Supported tokens are: ['last', 'last-2', 'last-3', 'last-image', 'end-image', 'all-image', 'all-text', 'all', 'special', 'random-text', 'random-image', 'random-image-10']"
            )
        token_indexes = self.categorize_tokens(string_tokens)

        tokens_positions = []

        token_dict = {
            "last": [-1],
            "last-2": [-2],
            "last-4": [-4],
            "last-image": token_indexes["last_line_image"],
            "end-image": token_indexes["image_end"],
            "all-text": token_indexes["text"],
            "all": [i for i in range(0, len(string_tokens))],
            "all-image": token_indexes["image"],
            "special": token_indexes["special"],
            "random-text": [random.choice(token_indexes["text"])],
            "random-image": [random.choice(token_indexes["image"])],
            "random-image-10": random.sample(token_indexes["image"], 10),
            "special-pixtral": [
                1052,
                1051,
                1038,
                991,
                1037,
                1047,
            ],  # 1032,  925,  988, 1050, 1046,
            # 1034, 1048, 1040, 1027, 1023, 1022, 1049, 1033, 1041, 1026, 1055,
            # 1053, 1054, 1024,   33, 1056,   66, 1025,] #! hard coded and hard finded by Ema
        }

        for token in tokens:
            tokens_positions.extend(token_dict[token])

        if return_type == "int":
            if len(tokens_positions) > 1:
                raise ValueError(
                    "More than one token requested: return_type should be list, got int"
                )
            return tokens_positions[0]
        if return_type == "dict":
            return token_dict
        return tokens_positions
