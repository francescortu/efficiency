from dataclasses import dataclass
from easyroutine.logger import LambdaLogger
from transformers import (
    ChameleonProcessor,
    ChameleonForConditionalGeneration,
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
    PixtralProcessor,
    LlamaForCausalLM,
    LlamaTokenizer,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    LlamaTokenizerFast,
    LlavaNextProcessor,
)
import random
from typing import List, Literal, Union, Dict, Optional, Tuple
import torch

SUPPORTED_MODELS = {
    "mistral-community/pixtral-12b": ("[IMG]", "[BREAK]", "[IMG_END]"),
    "Emu3-Chat": ("<|image start|>", None, "<|image end|>"),
    "Emu3-Gen": ("<|image start|>", None, "<|image end|>"),
    "Emu3-Stage1": ("<|image start|>", None, "<|image end|>"),
    "facebook/chameleon-7b": ("<racm3:break>", None, "<eoss>"),
    "facebook/chameleon-30b": ("<racm3:break>", None, "<eoss>"),
    "hf-internal-testing/tiny-random-LlamaForCausalLM": (None, None, None),
    "CohereForAI/aya-101": (None, None, None),
    "meta-llama/Llama-3.2-1B": (None, None, None),
    "meta-llama/Llama-3.2-3B": (None, None, None),
    "llava-hf/llava-v1.6-mistral-7b-hf": ("<image>", None, None),
}


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
        if model_name in [
            "facebook/chameleon-7b",
            "facebook/chameleon-30b",
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-3B",
        ]:
            if model_name in ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"]:
                model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    attn_implementation=attn_implementation,
                )
                language_model = None
            elif model_name in ["facebook/chameleon-7b", "facebook/chameleon-30b"]:
                model = ChameleonForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    attn_implementation=attn_implementation,
                )
                language_model = None
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

        elif model_name in [
            "mistral-community/pixtral-12b",
            "llava-hf/llava-v1.6-mistral-7b-hf",
        ]:
            if model_name == "mistral-community/pixtral-12b":
                model = LlavaForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    attn_implementation=attn_implementation,
                )
            elif model_name == "llava-hf/llava-v1.6-mistral-7b-hf":
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    attn_implementation=attn_implementation,
                )
            language_model = model.language_model

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
            language_model = None

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
        elif model_name in ["CohereForAI/aya-101"]:
            model = T5ForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch_dtype, device_map=device_map
            )
            language_model = None
            model_config = ModelConfig(
                residual_stream_input_hook_name="encoder.block[{}].input",
                residual_stream_hook_name="encoder.block[{}].output",
                intermediate_stream_hook_name="encoder.block[{}].layer[1].input",
                attn_value_hook_name="encoder.block[{}].layer[0].SelfAttention.v.output",
                attn_out_hook_name="encoder.block[{}].layer[0].SelfAttention.o.output",
                attn_in_hook_name="encoder.block[{}].layer[0].SelfAttention.input",
                attn_matrix_hook_name="encoder.block[{}].layer[0].SelfAttention.attention_matrix_hook.output",
                attn_out_proj_weight="encoder.block[{}].layer[0].SelfAttention.o.weight",
                attn_out_proj_bias="encoder.block[{}].layer[0].SelfAttention.o.bias",
                embed_tokens="encoder.embed_tokens",
                num_hidden_layers=model.config.num_layers,
                num_attention_heads=model.config.num_heads,
                hidden_size=model.config.d_model,
                num_key_value_heads=model.config.num_heads,
                num_key_value_groups=model.config.num_heads,
                head_dim=model.config.d_model // model.config.num_heads,
            )
        else:
            raise ValueError("Unsupported model_name")

        return model, language_model, model_config


class TokenizerFactory:
    @staticmethod
    def load_tokenizer(
        model_name: str, torch_dtype: torch.dtype, device_map: str
    ) -> Tuple[
        Union[
            LlamaTokenizer,
            LlamaTokenizerFast,
            PixtralProcessor,
            LlavaNextProcessor,
            T5TokenizerFast,
        ],
        bool,
    ]:
        if model_name in ["facebook/chameleon-7b", "facebook/chameleon-30b"]:
            processor = ChameleonProcessor.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            is_a_processor = True
        elif model_name in ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"]:
            processor = LlamaTokenizerFast.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            is_a_processor = False
        elif model_name in ["mistral-community/pixtral-12b"]:
            processor = PixtralProcessor.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            is_a_processor = True
        elif model_name in ["llava-hf/llava-v1.6-mistral-7b-hf"]:
            processor = LlavaNextProcessor.from_pretrained(
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
        elif model_name in ["CohereForAI/aya-101"]:
            processor = T5TokenizerFast.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            is_a_processor = False

        else:
            raise ValueError("Unsupported model_name")

        return processor, is_a_processor


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


class InputHandler:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def prepare_inputs(
        self,
        batch_dict: Dict[str, torch.Tensor],
        device: Union[str, torch.device],
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        if self.model_name in [
            "facebook/chameleon-7b",
            "facebook/chameleon-30b",
            "mistral-community/pixtral-12b",
        ]:
            input_dict = {
                "input_ids": batch_dict["input_ids"],
                "attention_mask": batch_dict["attention_mask"],
                "pixel_values": batch_dict["pixel_values"].to(torch_dtype),
            }
        elif self.model_name in ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"]:
            input_dict = {
                "input_ids": batch_dict["input_ids"],
                "attention_mask": batch_dict["attention_mask"],
            }
        elif self.model_name in ["Emu3-Chat", "Emu3-Gen", "Emu3-Stage1"]:
            raise NotImplementedError("Emu3 model not implemented yet")
        elif self.model_name in ["hf-internal-testing/tiny-random-LlamaForCausalLM"]:
            input_dict = {
                "input_ids": batch_dict["input_ids"],
                "attention_mask": batch_dict["attention_mask"],
            }
        elif self.model_name in ["llava-hf/llava-v1.6-mistral-7b-hf"]:
            if "pixel_values" not in batch_dict:
                input_dict = {
                    "input_ids": batch_dict["input_ids"],
                    "attention_mask": batch_dict["attention_mask"],
                }
            else:
                input_dict = {
                    "input_ids": batch_dict["input_ids"],
                    "attention_mask": batch_dict["attention_mask"],
                    "pixel_values": batch_dict["pixel_values"],
                    "image_sizes": batch_dict["image_sizes"],
                }
        elif self.model_name in ["CohereForAI/aya-101"]:
            input_dict = {
                "input_ids": batch_dict["input_ids"],
                "decoder_input_ids": batch_dict["input_ids"],
                "attention_mask": batch_dict["attention_mask"],
            }
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")
        input_dict = {k: v.to(device) for k, v in input_dict.items()}
        return input_dict

    def get_input_ids(
        self,
        input_dict: Dict[str, torch.Tensor],
    ):
        return input_dict["input_ids"]
