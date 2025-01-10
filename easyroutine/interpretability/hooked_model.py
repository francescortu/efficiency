import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize as torchvision_resize
from torchvision.transforms.functional import resized_crop as torchvision_resize_crop
from transformers import (
    GenerationConfig,
)
from typing import (
    Union,
    Literal,
    Optional,
    List,
    Dict,
    Callable,
    Any,
)

from easyroutine.interpretability.models import (
    ModelFactory,
    TokenizerFactory,
    InputHandler,
)
from easyroutine.interpretability.token_index import TokenIndex
from easyroutine.interpretability.utils import get_attribute_by_name
import numpy as np
from easyroutine.logger import Logger, LambdaLogger
from tqdm import tqdm
from dataclasses import dataclass, field
import os
from easyroutine.interpretability.ablation import AblationManager
import random
import json

# from src.model.emu3.
from easyroutine.interpretability.utils import (
    left_pad,
    aggregate_cache_efficient,
    aggregate_metrics,
    to_string_tokens,
    map_token_to_pos,
    preprocess_patching_queries,
    logit_diff,
    get_attribute_from_name,
    resize_img_with_padding,
    kl_divergence_diff,
)
from easyroutine.interpretability.hooks import (
    partial,
    embed_hook,
    save_resid_hook,
    # save_resid_in_hook,
    avg_hook,
    projected_value_vectors_head,
    avg_attention_pattern_head,
    attention_pattern_head,
    ablate_tokens_hook_flash_attn,
    get_module_by_path,
)

from functools import partial
from random import randint
import pandas as pd
from copy import deepcopy
from pathlib import Path


LambdaLogger.log(
    "This implementation use a fork of the HuggingFace transformer library to perform some experiment. Be sure to have the right version of the library (pip install git+https://github.com/francescortu/transformers.git@easyroutine)",
    level="WARNING",
)

# to avoid running out of shared memory
# torch.multiprocessing.set_sharing_strategy("file_system")


@dataclass
class HookedModelConfig:
    model_name: str
    device_map: Literal["balanced", "cuda", "cpu", "auto"] = "balanced"
    torch_dtype: torch.dtype = torch.bfloat16
    attn_implementation: Literal["eager", "flash_attention_2"] = "eager"
    batch_size: int = 1


class HookedModel:
    """
    This class is a wrapper around the huggingface model that allows to extract the activations of the model. It is support
    advanced mechanistic intepretability methods like ablation, patching, etc.
    """

    def __init__(self, config: HookedModelConfig, log_file_path: Optional[str] = None):
        self.logger = Logger(
            logname="HookedModel",
            level="info",
            log_file_path=log_file_path,
        )

        self.config = config
        self.hf_model, self.hf_language_model, self.model_config = ModelFactory.load_model(
            model_name=config.model_name,
            device_map=config.device_map,
            torch_dtype=config.torch_dtype,
            attn_implementation=config.attn_implementation,
        )

        tokenizer, processor = TokenizerFactory.load_tokenizer(
            model_name=config.model_name,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
        )
        self.hf_tokenizer = tokenizer
        self.input_handler = InputHandler(model_name=config.model_name)
        if processor is True:
            self.processor = tokenizer
            self.text_tokenizer = self.processor.tokenizer  # type: ignore
        else:
            self.processor = None
            self.text_tokenizer = tokenizer
        
            
        # self.hf_language_model = extract_language_model(self.hf_model)
        
            
        self.first_device = next(self.hf_model.parameters()).device
        device_num = torch.cuda.device_count()
        self.logger.info(
            f"Model loaded in {device_num} devices. First device: {self.first_device}",
            std_out=True,
        )
        self.act_type_to_hook_name = {
            "resid_in": self.model_config.residual_stream_input_hook_name,
            "resid_out": self.model_config.residual_stream_hook_name,
            "resid_mid": self.model_config.intermediate_stream_hook_name,
            "attn_out": self.model_config.attn_out_hook_name,
            "attn_in": self.model_config.attn_in_hook_name,
            "values": self.model_config.attn_value_hook_name,
            # Add other act_types if needed
        }
        self.additional_hooks = []
        self.assert_all_modules_exist()
        
    def __repr__(self):
        return f"""HookedModel(model_name={self.config.model_name}):
        {self.hf_model.__repr__()}
    """

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        return cls(HookedModelConfig(model_name=model_name, **kwargs))

    def assert_module_exists(self, component: str):
        # Remove '.input' or '.output' from the component
        component = component.replace(".input", "").replace(".output", "")

        # Check if '{}' is in the component, indicating layer indexing
        if "{}" in component:
            for i in range(0, self.model_config.num_hidden_layers):
                attr_name = component.format(i)
                try:
                    get_attribute_by_name(self.hf_model, attr_name)
                except AttributeError:
                    raise ValueError(
                        f"Component '{attr_name}' does not exist in the model. Please check the model configuration."
                    )
        else:
            try:
                get_attribute_by_name(self.hf_model, component)
            except AttributeError:
                raise ValueError(
                    f"Component '{component}' does not exist in the model. Please check the model configuration."
                )

    def assert_all_modules_exist(self):
        # get the list of all attributes of model_config
        all_attributes = [attr_name for attr_name in self.model_config.__dict__.keys()]
        # save just the attributes that have "hook" in the name
        hook_attributes = [
            attr_name for attr_name in all_attributes if "hook" in attr_name
        ]
        for hook_attribute in hook_attributes:
            self.assert_module_exists(getattr(self.model_config, hook_attribute))
            
    def use_full_model(self):
        self.use_language_model = False
        if self.processor is not None:
            self.logger.info("Using full model capabilities", std_out=True)
        else:
            self.logger.info("Using full text only model capabilities", std_out=True)

    def use_language_model_only(self):
        if self.hf_language_model is None:
            self.logger.warning("The model does not have a separate language model that can be used", std_out=True)
        else:
            self.use_language_model = True
            self.logger.info("Using only language model capabilities", std_out=True)
            
    
    def get_tokenizer(self):
        return self.hf_tokenizer

    def get_text_tokenizer(self):
        """
        If the tokenizer is a processor, return just the tokenizer. If the tokenizer is a tokenizer, return the tokenizer
        """
        if self.processor is not None:
            if not hasattr(self.processor, "tokenizer"):
                raise ValueError("The processor does not have a tokenizer")
            return self.processor.tokenizer  # type: ignore
        return self.hf_tokenizer

    def get_processor(self):
        if self.processor is None:
            raise ValueError("The model does not have a processor")
        return self.processor

    def eval(self):
        self.hf_model.eval()

    def device(self):
        return self.first_device

    def register_forward_hook(self, component: str, hook_function: Callable):
        self.additional_hooks.append(
            {
                "component": component,
                "intervention": hook_function,
            }
        )

    def to_string_tokens(
        self,
        tokens: Union[list, torch.Tensor],
    ):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        string_tokens = []
        for tok in tokens:
            string_tokens.append(self.hf_tokenizer.decode(tok))  # type: ignore
        return string_tokens

    def create_hooks(
        self,
        inputs,
        cache: Dict[str, torch.Tensor],
        token_index: List,
        token_dict: Dict,
        # string_tokens: List[str],
        attn_heads: Union[list[dict], Literal["all"]] = "all",
        extract_attn_pattern: bool = False,
        extract_attn_out: bool = False,
        extract_attn_in: bool = False,
        extract_avg_attn_pattern: bool = False,
        extract_avg_values_vectors_projected: bool = False,
        extract_resid_in: bool = False,
        extract_resid_out: bool = False,
        extract_values: bool = False,
        extract_resid_mid: bool = False,
        save_input_ids: bool = False,
        extract_head_out: bool = False,
        extract_values_vectors_projected: bool = False,
        extract_avg: bool = False,
        ablation_queries: Optional[Union[dict, pd.DataFrame]] = None,
        patching_queries: Optional[Union[dict, pd.DataFrame]] = None,
        batch_idx: Optional[int] = None,
        external_cache: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """TODO: Rewrite the docstring
        Unique routine to extract the activations of multiple models. It uses both a standard huggingface model and pyvene model, which is a wrapper around the huggingface model
        that allows to set sum hooks around the modules of the model. It supports the following args
        Args:
            - inputs: dictionary with the inputs of the model {"input_ids": ..., "attention_mask": ..., "pixel_values": ...}
            - attn_heads: list of dictionaries with the layer and head we want to extract something (e.g. attn pattern or values vectors...). If "all" is passed, it will extract all the heads of all the layers
            it expect a list of dictionaries with the keys "layer" and "head": [{"layer": 0, "head": 0}, {"layer": 1, "head": 1}, ...] # NICE-TO-HAVE: add a assert to check the format
            - extract_attn_pattern: bool to extract the attention pattern (i.e. the attention matrix)
            - extract_avg_attn_pattern: bool to extract the average attention pattern. It will extract the average of the attention pattern of the heads passed in attn_heads. The average is saved in the external_cache
            - extract_avg_values_vectors: bool to extract the average values vectors. It will extract the average of the values vectors of the heads passed in attn_heads. The average is saved in the external_cache. The computation will be
                                        alpha_ij * ||V_j|| where alpha_ij is the attention pattern and V_j is the values vectors for each element of the batch. The average is computed for each element of the batch. It return a matrix of shape [batch, seq_len, seq_len]
            - extract_intermediate_states: bool to extract the intermediate states of the model (i.e. the hiddden rappresentation between the attention and the MLP)
            - save_input_ids: bool to save the input_ids in the cache
            - extract_head_out: bool to extract the output of the heads. It will extract the output of the heads projected by the final W_O projection.
            - extract_values_vectors: bool to extract the values vectors. It will extract the values vectors projected by the final W_O projection. If X_i is the residual stream of the i layer, it will return W_OV * X_i
            - move_to_cpu: bool to move the activations to the cpu before returning the cache. Sometimes it's useful to move the activations to the cpu to avoid to fill the gpu memory, while sometimes it's better to keep the activations on the gpu to avoid to move them back and forth
            - ablation_queries: dataframe with the ablation queries. The user can configure the ablation through a json file passed to extract_activations.py
            - patching_queries: dataframe with the patching queries.
            - freeze_ablation: if true, the attention weights will be frozen during the ablation.
            - external_cache: dictionary with the activations of the model. If passed, the activations will be saved in this dictionary. This is useful if we want to save average activations of multiple batches
            - idx_batch: index of the batch. It's useful to save the activations in the external_cache or perform mean computation

        Returns:
            - cache: dictionary with the activations of the model. The keys are the names of the activations and the values are the activations themselve
                cache = {
                    "resid_out_0": tensor of shape [batch, seq_len, hidden_size] with the activations of the residual stream of layer 0
                    "resid_mid_0": tensor of shape [batch, seq_len, hidden_size] with the activations of the residual stream of layer 0
                    "pattern_L0": tensor of shape [num_heads, batch, seq_len, seq_len] with the attention pattern of layer 0,
                    "patten_L0H0": tensor of shape [batch, seq_len, seq_len] with the attention pattern of layer 0 and head 0,
                    "input_ids": tensor of shape [batch, seq_len] with the input_ids,
                    "head_out_0": tensor of shape [batch, seq_len, hidden_size] with the output of the heads of layer 0
                    ...
                    }
        """
        # set the model family
        #
        if extract_attn_pattern or extract_head_out or extract_values_vectors_projected:
            if (
                attn_heads is None
            ):  # attn_head must be passed if we want to extract the attention pattern or the output of the heads. If not, raise an error
                raise ValueError(
                    "attn_heads must be a list of dictionaries with the layer and head to extract the attention pattern or 'all' to extract all the heads of all the layers"
                )

        # process the token where we want the activation from

        # token_index, token_dict = TokenIndex(
        #     self.config.model_name, split_positions=split_positions
        # ).get_token_index(tokens=target_token_positions, string_tokens=string_tokens)

        # define a dynamic factory hook. It takes a function and the corresponding kwargs and returns a function that pyvene can use. This is necessary to use partial() in the hook function
        # but still be consistent with the type of the function that pyvene expects. It's basically a custom partial function that retuns a function of type FuncType

        hooks = []

        if extract_resid_out:
            # assert that the component exists in the model
            hooks += [
                {
                    "component": self.model_config.residual_stream_hook_name.format(i),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"resid_out_{i}",
                        token_index=token_index,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]
        if extract_resid_in:
            # assert that the component exists in the model
            hooks += [
                {
                    "component": self.model_config.residual_stream_input_hook_name.format(
                        i
                    ),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"resid_in_{i}",
                        token_index=token_index,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]

        if save_input_ids:
            hooks += [
                {
                    "component": self.model_config.embed_tokens,
                    "intervention": partial(
                        embed_hook,
                        cache=cache,
                        cache_key="input_ids",
                    ),
                }
            ]

        if extract_values:
            hooks += [
                {
                    "component": self.model_config.attn_value_hook_name.format(i),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"values_{i}",
                        token_index=token_index,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]

        if extract_attn_in:
            hooks += [
                {
                    "component": self.model_config.attn_in_hook_name.format(i),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"attn_in_{i}",
                        token_index=token_index,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]

        if extract_attn_out:
            hooks += [
                {
                    "component": self.model_config.attn_out_hook_name.format(i),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"attn_out_{i}",
                        token_index=token_index,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]

        if extract_avg:
            # Define a hook that saves the activations of the residual stream
            raise NotImplementedError(
                "The hook for the average is not working with token_index as a list"
            )

            # hooks.extend(
            #     [
            #         {
            #             "component": self.model_config.residual_stream_hook_name.format(
            #                 i
            #             ),
            #             "intervention": partial(
            #                 avg_hook,
            #                 cache=cache,
            #                 cache_key="resid_avg_{}".format(i),
            #                 last_image_idx=last_image_idxs, #type
            #                 end_image_idx=end_image_idxs,
            #             ),
            #         }
            #         for i in range(0, self.model_config.num_hidden_layers)
            #     ]
            # )
        if extract_resid_mid:
            hooks += [
                {
                    "component": self.model_config.intermediate_stream_hook_name.format(
                        i
                    ),
                    "intervention": partial(
                        save_resid_hook,
                        cache=cache,
                        cache_key=f"resid_mid_{i}",
                        token_index=token_index,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]

            # if we want to extract the output of the heads

        # PATCHING
        if patching_queries:
            token_to_pos = partial(
                map_token_to_pos,
                _get_token_index=token_dict,
                # string_tokens=string_tokens,
                hf_tokenizer=self.hf_tokenizer,
                inputs=inputs,
            )
            patching_queries = preprocess_patching_queries(
                patching_queries=patching_queries,
                map_token_to_pos=token_to_pos,
                model_config=self.model_config,
            )

            def make_patch_tokens_hook(patching_queries_subset):
                """
                Creates a hook function to patch the activations in the
                current forward pass.
                """

                def patch_tokens_hook(module, input, output):
                    if output is None:
                        if isinstance(input, tuple):
                            b = input[0]
                        else:
                            b = input
                    else:
                        if isinstance(output, tuple):
                            b = output[0]
                        else:
                            b = output
                    # Modify the tensor without affecting the computation graph
                    act_to_patch = b.detach().clone()
                    for pos, patch in zip(
                        patching_queries_subset["pos_token_to_patch"],
                        patching_queries_subset["patching_activations"],
                    ):
                        act_to_patch[0, pos, :] = patching_queries_subset[
                            "patching_activations"
                        ].values[0]

                    if output is None:
                        if isinstance(input, tuple):
                            return (act_to_patch, *input[1:])
                        elif input is not None:
                            return act_to_patch
                    else:
                        if isinstance(output, tuple):
                            return (act_to_patch, *output[1:])
                        elif output is not None:
                            return act_to_patch
                    raise ValueError("No output or input found")

                return patch_tokens_hook

            # Group the patching queries by 'layer' and 'act_type'
            grouped_queries = patching_queries.groupby(["layer", "activation_type"])

            for (layer, act_type), group in grouped_queries:
                hook_name_template = self.act_type_to_hook_name.get(
                    act_type[:-3]
                )  # -3 because we remove {}
                if not hook_name_template:
                    raise ValueError(f"Unknown activation type: {act_type}")
                    # continue  # Skip unknown activation types

                hook_name = hook_name_template.format(layer)
                hook_function = partial(make_patch_tokens_hook(group))

                hooks.append(
                    {
                        "component": hook_name,
                        "intervention": hook_function,
                    }
                )

        if ablation_queries is not None:
            # TODO: debug and test the ablation. Check with ale
            token_to_pos = partial(
                map_token_to_pos,
                _get_token_index=token_dict,
                # string_tokens=string_tokens,
                hf_tokenizer=self.hf_tokenizer,
                inputs=inputs,
            )
            if self.config.batch_size > 1:
                raise ValueError("Ablation is not supported with batch size > 1")
            ablation_manager = AblationManager(
                model_config=self.model_config,
                token_to_pos=token_to_pos,
                inputs=inputs,
                model_attn_type=self.config.attn_implementation,
                ablation_queries=pd.DataFrame(ablation_queries)
                if isinstance(ablation_queries, dict)
                else ablation_queries,
            )
            hooks.extend(ablation_manager.main())

        if extract_values_vectors_projected or extract_avg_values_vectors_projected:
            if attn_heads == "all":  # extract the output of all the heads
                hooks += [
                    {
                        "component": self.model_config.attn_value_hook_name.format(i),
                        "intervention": partial(
                            projected_value_vectors_head,
                            cache=cache,
                            layer=i,
                            num_attention_heads=self.model_config.num_attention_heads,
                            num_key_value_heads=self.model_config.num_key_value_heads,
                            hidden_size=self.model_config.hidden_size,
                            d_head=self.model_config.head_dim,
                            out_proj_weight=get_attribute_from_name(
                                self.hf_model,
                                f"{self.model_config.attn_out_proj_weight.format(i)}",
                            ),
                            out_proj_bias=get_attribute_from_name(
                                self.hf_model,
                                f"{self.model_config.attn_out_proj_bias.format(i)}",
                            ),
                            head="all",
                        ),
                    }
                    for i in range(0, self.model_config.num_hidden_layers)
                ]
            elif isinstance(attn_heads, list):
                for el in attn_heads:
                    head = el["head"]
                    layer = el["layer"]
                    hooks.append(
                        {
                            "component": self.model_config.attn_value_hook_name.format(
                                layer
                            ),
                            "intervention": partial(
                                projected_value_vectors_head,
                                cache=cache,
                                layer=layer,
                                num_attention_heads=self.model_config.num_attention_heads,
                                hidden_size=self.model_config.hidden_size,
                                out_proj_weight=self.hf_model.model.layers[
                                    layer
                                ].self_attn.o_proj.weight,  # (d_model, d_model)
                                out_proj_bias=self.hf_model.model.layers[
                                    layer
                                ].self_attn.o_proj.bias,  # (d_model)
                                head=head,
                            ),
                        }
                    )
        if extract_avg_attn_pattern:
            hooks += [
                {
                    "component": self.model_config.attn_matrix_hook_name.format(i),
                    "intervention": partial(
                        avg_attention_pattern_head,
                        layer=i,
                        attn_pattern_current_avg=external_cache,
                        batch_idx=batch_idx,
                        cache=cache,
                        extract_avg_value=extract_avg_values_vectors_projected,
                    ),
                }
                for i in range(0, self.model_config.num_hidden_layers)
            ]
        if extract_attn_pattern:
            if attn_heads == "all":
                hooks += [
                    {
                        "component": self.model_config.attn_matrix_hook_name.format(i),
                        "intervention": partial(
                            attention_pattern_head,
                            cache=cache,
                            layer=i,
                            head="all",
                        ),
                    }
                    for i in range(0, self.model_config.num_hidden_layers)
                ]
            else:
                hooks += [
                    {
                        "component": self.model_config.attn_matrix_hook_name.format(
                            el["layer"]
                        ),
                        "intervention": partial(
                            attention_pattern_head,
                            cache=cache,
                            layer=el["layer"],
                            head=el["head"],
                        ),
                    }
                    for el in attn_heads
                ]

            # if additional hooks are not empty, add them to the hooks list
        if self.additional_hooks:
            hooks += self.additional_hooks
        return hooks

    @torch.no_grad()
    def forward(
        self,
        inputs,
        target_token_positions: List[str] = ["last"],
        split_positions: Optional[List[int]] = None,
        extract_resid_in: bool = False,
        extract_resid_mid: bool = False,
        extract_resid_out: bool = False,
        extract_attn_pattern: bool = False,
        extract_avg_attn_pattern: bool = False,
        extract_values_vectors_projected: bool = False,
        extract_avg_values_vectors_projected: bool = False,
        extract_values: bool = False,
        extract_head_out: bool = False,
        extract_attn_out: bool = False,
        extract_attn_in: bool = False,
        save_input_ids: bool = False,
        extract_avg: bool = False,
        ablation_queries: Optional[pd.DataFrame | None] = None,
        patching_queries: Optional[pd.DataFrame | None] = None,
        external_cache: Optional[Dict] = None,
        attn_heads: Union[list[dict], Literal["all"]] = "all",
        batch_idx: Optional[int] = None,
        move_to_cpu: bool = False,
    ):
        """ """
        model_to_use = self.hf_language_model if self.use_language_model else self.hf_model
        assert model_to_use is not None, "Error: The model is not loaded"
        
        if target_token_positions is None and any(
            [
                extract_resid_in,
                extract_resid_mid,
                extract_resid_out,
                extract_attn_pattern,
                extract_avg_attn_pattern,
                extract_values_vectors_projected,
                extract_avg_values_vectors_projected,
                extract_values,
                extract_head_out,
                extract_attn_out,
                extract_attn_in,
                extract_avg,
                ablation_queries,
                patching_queries,
            ]
        ):
            raise ValueError(
                "target_token_positions must be passed if we want to extract the activations of the model"
            )
                
        cache = {}
        string_tokens = self.to_string_tokens(
            self.input_handler.get_input_ids(inputs).squeeze()
        )
        token_index, token_dict = TokenIndex(
            self.config.model_name, split_positions=split_positions
        ).get_token_index(tokens=target_token_positions, string_tokens=string_tokens, return_type="all")

        hooks = self.create_hooks(  # TODO: add **kwargs
            inputs=inputs,
            token_dict=token_dict,
            token_index=token_index,
            cache=cache,
            attn_heads=attn_heads,
            extract_attn_pattern=extract_attn_pattern,
            extract_attn_out=extract_attn_out,
            extract_attn_in=extract_attn_in,
            extract_avg_attn_pattern=extract_avg_attn_pattern,
            extract_avg_values_vectors_projected=extract_avg_values_vectors_projected,
            extract_resid_in=extract_resid_in,
            extract_resid_out=extract_resid_out,
            extract_values=extract_values,
            extract_resid_mid=extract_resid_mid,
            save_input_ids=save_input_ids,
            extract_head_out=extract_head_out,
            extract_values_vectors_projected=extract_values_vectors_projected,
            extract_avg=extract_avg,
            ablation_queries=ablation_queries,
            patching_queries=patching_queries,
            batch_idx=batch_idx,
            external_cache=external_cache,
        )
        # define the pyvene model, i.e. a wrapper around the huggingface model that allows to set hooks around the modules of the model

        # log_memory_usage("Before creating the model")
        hook_handlers = self.set_hooks(hooks)
        # pv_model = pv.IntervenableModel(hooks, model=self.hf_model)
        # log_memory_usage("After creating the model")
        inputs = self.input_handler.prepare_inputs(inputs, self.first_device, self.config.torch_dtype)
        # forward pass
        output = model_to_use(
            **inputs,
            # output_original_output=True,
            # output_attentions=extract_attn_pattern,
        )

        # log_memory_usage("After forward pass")

        cache["logits"] = output.logits[:, -1]
        # since attention_patterns are returned in the output, we need to adapt to the cache structure
        if move_to_cpu:
            for key, value in cache.items():
                if extract_avg_values_vectors_projected:
                    # remove the values vectors from the cache
                    if "values" in key:
                        del cache[key]
                cache[key] = value.detach().cpu()
            if external_cache is not None:
                for key, value in external_cache.items():
                    external_cache[key] = value.detach().cpu()

        mapping_index = {}
        current_index = 0
        for token in target_token_positions:
            mapping_index[token] = []
            if isinstance(token_dict, int):
                mapping_index[token].append(current_index)
                current_index += 1
            elif isinstance(token_dict, dict):
                for idx in range(len(token_dict[token])):
                    mapping_index[token].append(current_index)
                    current_index += 1
            elif isinstance(token_dict, list):
                for idx in range(len(token_dict)):
                    mapping_index[token].append(current_index)
                    current_index += 1
            else:
                raise ValueError("Token dict must be an int, a dict or a list")
        cache["mapping_index"] = mapping_index

        self.remove_hooks(hook_handlers)

        return cache

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)

    def predict(self, k=10, **kwargs):
        out = self.forward(**kwargs)
        logits = out["logits"]
        probs = torch.softmax(logits, dim=-1)
        probs = probs.squeeze()
        topk = torch.topk(probs, k)
        #return a dictionary with the topk tokens and their probabilities
        string_tokens = self.to_string_tokens(topk.indices)
        token_probs = {}
        for token,prob in zip(string_tokens, topk.values):
            if token not in token_probs:
                token_probs[token] = prob.item()
        return token_probs
        # return {
        #     token: prob.item() for token, prob in zip(string_tokens, topk.values)
        # }

    def get_module_from_string(self, component: str):
        return self.hf_model.retrieve_modules_from_names(component)

    def set_hooks(self, hooks: List[Dict[str, Any]]):
        # 1. Parsing the module path

        if len(hooks) == 0:
            return []

        hook_handlers = []
        for hook in hooks:
            component = hook["component"]
            hook_function = hook["intervention"]

            # get the last module string (.input or .output) and remove it from the component string
            last_module = component.split(".")[-1]
            # now remove the last module from the component string
            component = component[: -len(last_module) - 1]

            if last_module == "input":
                hook_handlers.append(
                    get_module_by_path(
                        self.hf_model, component
                    ).register_forward_pre_hook(partial(hook_function, output=None))
                )
            elif last_module == "output":
                hook_handlers.append(
                    get_module_by_path(self.hf_model, component).register_forward_hook(
                        hook_function
                    )
                )

        return hook_handlers

    def remove_hooks(self, hook_handlers):
        for hook_handler in hook_handlers:
            hook_handler.remove()

    @torch.no_grad()
    def generate(
        self,
        inputs,
        generation_config: Optional[GenerationConfig] = None,
        target_token_positions: Optional[List[str]] = None,
        return_text: bool = False,
        **kwargs,
    ):
        """
        Generate new tokens using the model and the inputs passed as argument
        Args:
            - inputs: dictionary with the inputs of the model {"input_ids": ..., "attention_mask": ..., "pixel_values": ...}
            - generation_config: original hf dataclass with the generation configuration
            - kwargs: additional arguments to control hooks generation (i.e. ablation_queries, patching_queries)
        Returns:
            - output: dictionary with the output of the model
        """
        # Initialize cache for logits
        # TODO FIX THIS. IT is not general and it is not working
        # raise NotImplementedError("This method is not working. It needs to be fixed")
        hook_handlers = None
        if target_token_positions is not None:
            string_tokens = self.to_string_tokens(
                self.input_handler.get_input_ids(inputs).squeeze()
            )
            token_index, token_dict = TokenIndex(
                self.config.model_name, split_positions=None
            ).get_token_index(tokens=[], string_tokens=string_tokens, return_type="all")
            hooks = self.create_hooks(
                inputs=inputs,
                token_dict=token_dict,
                token_index=token_index,
                cache={},
                extract_resid_out=False,
                **kwargs,
            )
            hook_handlers = self.set_hooks(hooks)
            
        inputs = self.input_handler.prepare_inputs(inputs, self.first_device)
        
        model_to_use = self.hf_language_model if self.use_language_model else self.hf_model
        assert model_to_use is not None, "Error: The model is not loaded"
        
        output = model_to_use.generate(
            **inputs, generation_config=generation_config, output_scores=False
        )
        if hook_handlers:
            self.remove_hooks(hook_handlers)
        if return_text:
            return self.hf_tokenizer.decode(output[0], skip_special_tokens=True)
        return output # type: ignore

    @torch.no_grad()
    def extract_cache(
        self,
        dataloader,
        target_token_positions: List[str],
        batch_saver: Callable = lambda x: None,
        move_to_cpu_after_forward: bool = True,
        # save_other_batch_elements: bool = False,
        **kwargs,
    ):
        """
        Method to extract the activations of the model. It will perform a forward pass on the dataloader and extract the activations of the model
        Args:
            - dataloader: dataloader with the dataset
            - batch_saver: function to save additional elements from the batch. It will be called after the forward pass and it will receive the batch as input and return a dictionary with the additional elements to save in the cache

        Returns:
            - final_cache: dictionary with the activations of the model. The keys are the names of the activations and the values are the activations themselve
        """

        self.logger.info("Extracting cache", std_out=True)

        # get the function to save in the cache the additional element from the batch sime

        self.logger.info("Forward pass started", std_out=True)
        all_cache = []  # a list of dictoionaries, each dictionary contains the activations of the model for a batch (so a dict of tensors)
        attn_pattern = {}  # Initialize the dictionary to hold running averages

        example_dict = {}
        n_batches = 0  # Initialize batch counter

        for batch in tqdm(dataloader, total=len(dataloader), desc="Extracting cache:"):
            # log_memory_usage("Extract cache - Before batch")
            # tokens, others = batch
            # inputs = {k: v.to(self.first_device) for k, v in tokens.items()}

            # get input_ids, attention_mask, and if available, pixel_values from batch (that is a dictionary)
            # then move them to the first device
            inputs = self.input_handler.prepare_inputs(batch, self.first_device)
            others = {k: v for k, v in batch.items() if k not in inputs}

            cache = self.forward(
                inputs,
                target_token_positions=target_token_positions,
                split_positions=batch.get("split_positions", None),
                external_cache=attn_pattern,
                batch_idx=n_batches,
                **kwargs,
            )
            # possible memory leak from here -___--------------->
            additional_dict = batch_saver(others)
            if additional_dict is not None:
                cache = {**cache, **additional_dict}

            if move_to_cpu_after_forward:
                for key, value in cache.items():
                    if isinstance(value, torch.Tensor):
                        cache[key] = value.detach().cpu()

            n_batches += 1  # Increment batch counter# Process and remove "pattern_" keys from cache
            # log_memory_usage("Extract cache - Before append")
            all_cache.append(cache)
            # log_memory_usage("Extract cache - After append")
            del cache
            inputs = {k: v.cpu() for k, v in inputs.items()}
            del inputs
            torch.cuda.empty_cache()
            # log_memory_usage("Extract cache - After empty cache")

        self.logger.info(
            "Forward pass finished - started to aggregate different batch", std_out=True
        )
        final_cache = aggregate_cache_efficient(all_cache)
        final_cache = {
            **final_cache,
            **attn_pattern,
        }  # Add the running averages to the final cache

        # add the example_dict to the final_cache as a sub-dictionary
        final_cache["example_dict"] = example_dict
        self.logger.info("Aggregation finished", std_out=True)

        torch.cuda.empty_cache()
        return final_cache

    @torch.no_grad()
    def compute_patching(
        self,
        target_token_positions: List[str],
        # counterfactual_dataset,
        base_dataloader,
        target_dataloader,
        base_dictonary_idxs: Optional[List[List[int]]] = None,
        target_dictonary_idxs: Optional[List[List[int]]] = None,
        patching_query=[
            {
                "patching_elem": "@end-image",
                "layers_to_patch": [1, 2, 3, 4],
                "activation_type": "resid_in_{}",
            }
        ],
        return_logit_diff: bool = False,
        batch_saver: Callable = lambda x: None,
        **kwargs,
    ) -> Dict:
        """
        Method to activation patching. It substitutes the activations of the model with the activations of the counterfactual dataset

        It will perform three forward passes:
        1. Forward pass on the base dataset to extract the activations of the model (cat)
        2. Forward pass on the target dataset to extract clean logits (dog) [to compare against the patched logits]
        3. Forward pass on the target dataset to patch (cat) into (dog) and extract the patched logits
        Args:


        """
        self.logger.info("Computing patching", std_out=True)

        self.logger.info("Forward pass started", std_out=True)
        self.logger.info(
            f"Patching elements: {[q['patching_elem'] for q in patching_query]} at {[query['activation_type'][:-3] for query in patching_query]}",
            std_out=True,
        )

        # get a random number in the range of the dataset to save a random batch
        all_cache = []
        # for each batch in the dataset
        for index, (base_batch, target_batch) in tqdm(
            enumerate(zip(base_dataloader, target_dataloader)),
            desc="Computing patching on the dataset:",
            total=len(base_dataloader),
        ):
            torch.cuda.empty_cache()
            inputs = self.input_handler.prepare_inputs(base_batch, self.first_device)

            # set the right arguments for extract the patching activations
            activ_type = [query["activation_type"][:-3] for query in patching_query]

            args = {
                "extract_resid_out": True,
                "extract_resid_in": False,
                "extract_resid_mid": False,
                "extract_attn_in": False,
                "extract_attn_out": False,
                "extract_values": False,
                "extract_head_out": False,
                "extract_avg_attn_pattern": False,
                "extract_avg_values_vectors_projected": False,
                "extract_values_vectors_projected": False,
                "extract_avg": False,
                "ablation_queries": None,
                "patching_queries": None,
                "external_cache": None,
                "attn_heads": "all",
                "batch_idx": None,
                "move_to_cpu": False,
            }

            if "resid_in" in activ_type:
                args["extract_resid_in"] = True
            if "resid_out" in activ_type:
                args["extract_resid_out"] = True
            if "resid_mid" in activ_type:
                args["extract_intermediate_states"] = True
            if "attn_in" in activ_type:
                args["extract_attn_in"] = True
            if "attn_out" in activ_type:
                args["extract_attn_out"] = True
            if "values" in activ_type:
                args["extract_values"] = True
            # other cases

            # first forward pass to extract the base activations
            base_cache = self.forward(
                inputs=inputs,
                target_token_positions=target_token_positions,
                split_positions=base_batch.get("split_positions", None),
                **args,
            )

            # extract the target activations
            target_inputs = self.input_handler.prepare_inputs(
                target_batch, self.first_device
            )

            requested_position_to_extract = []
            for query in patching_query:
                query["patching_activations"] = base_cache
                if (
                    query["patching_elem"].split("@")[1]
                    not in requested_position_to_extract
                ):
                    requested_position_to_extract.append(
                        query["patching_elem"].split("@")[1]
                    )
                query["base_activation_index"] = base_cache["mapping_index"][
                    query["patching_elem"].split("@")[1]
                ]

            # second forward pass to extract the clean logits
            target_clean_cache = self.forward(
                target_inputs,
                target_token_positions=requested_position_to_extract,
                split_positions=target_batch.get("split_positions", None),
                extract_resid_out=False,
                # move_to_cpu=True,
            )

            # merge requested_position_to_extract with extracted_token_positio
            # third forward pass to patch the activations
            target_patched_cache = self.forward(
                target_inputs,
                target_token_positions=list(
                    set(target_token_positions + requested_position_to_extract)
                ),
                split_positions=target_batch.get("split_positions", None),
                patching_queries=patching_query,
                **kwargs,
            )

            if return_logit_diff:
                if base_dictonary_idxs is None or target_dictonary_idxs is None:
                    raise ValueError(
                        "To compute the logit difference, you need to pass the base_dictonary_idxs and the target_dictonary_idxs"
                    )
                self.logger.info("Computing logit difference", std_out=True)
                # get the target tokens (" cat" and " dog")
                base_targets = base_dictonary_idxs[index]
                target_targets = target_dictonary_idxs[index]

                # compute the logit difference
                result_diff = logit_diff(
                    base_label_tokens=[s for s in base_targets],
                    target_label_tokens=[c for c in target_targets],
                    target_clean_logits=target_clean_cache["logits"],
                    target_patched_logits=target_patched_cache["logits"],
                )
                target_patched_cache["logit_diff_variation"] = result_diff[
                    "diff_variation"
                ]
                target_patched_cache["logit_diff_in_clean"] = result_diff[
                    "diff_in_clean"
                ]
                target_patched_cache["logit_diff_in_patched"] = result_diff[
                    "diff_in_patched"
                ]

            # compute the KL divergence
            result_kl = kl_divergence_diff(
                base_logits=base_cache["logits"],
                target_clean_logits=target_clean_cache["logits"],
                target_patched_logits=target_patched_cache["logits"],
            )
            for key, value in result_kl.items():
                target_patched_cache[key] = value

            target_patched_cache["base_logits"] = base_cache["logits"]
            target_patched_cache["target_clean_logits"] = target_clean_cache["logits"]
            # rename logits to target_patched_logits
            target_patched_cache["target_patched_logits"] = target_patched_cache[
                "logits"
            ]
            del target_patched_cache["logits"]

            # move to cpu
            for key, value in target_patched_cache.items():
                if isinstance(value, torch.Tensor):
                    target_patched_cache[key] = value.detach().cpu()

            all_cache.append(target_patched_cache)

        self.logger.info(
            "Forward pass finished - started to aggregate different batch", std_out=True
        )
        final_cache = aggregate_cache_efficient(all_cache)

        self.logger.info("Aggregation finished", std_out=True)
        return final_cache
