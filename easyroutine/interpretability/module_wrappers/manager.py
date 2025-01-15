from typing import Union, Type
import torch.nn as nn

from easyroutine.interpretability.module_wrappers.chameleon_attention import (
    ChameleonAttentionWrapper,
)
from easyroutine.interpretability.module_wrappers.llama_attention import (
    LlamaAttentionWrapper,
)
from easyroutine.interpretability.module_wrappers.T5_attention import T5AttentionWrapper
from easyroutine.logger import Logger


class AttentionWrapperFactory:
    """
    Maps a given model name to the correct attention wrapper class.
    """

    MODEL_NAME_TO_WRAPPER = {
        "facebook/chameleon-7b": ChameleonAttentionWrapper,
        "facebook/chameleon-30b": ChameleonAttentionWrapper,
        "mistral-community/pixtral-12b": LlamaAttentionWrapper,
        "llava-hf/llava-v1.6-mistral-7b-hf": LlamaAttentionWrapper,
        "hf-internal-testing/tiny-random-LlamaForCausalLM": LlamaAttentionWrapper,
        "ChoereForAI/aya-101": T5AttentionWrapper,
    }

    @staticmethod
    def get_wrapper_class(
        model_name: str,
    ) -> Union[
        Type[ChameleonAttentionWrapper],
        Type[LlamaAttentionWrapper],
        Type[T5AttentionWrapper],
    ]:
        """
        Returns the attention wrapper class for the specified model name.
        Raises a ValueError if the model is not supported.
        """
        if model_name in AttentionWrapperFactory.MODEL_NAME_TO_WRAPPER:
            return AttentionWrapperFactory.MODEL_NAME_TO_WRAPPER[model_name]

        raise ValueError(
            f"Model {model_name} not supported by AttentionWrapperFactory."
        )


class ModuleWrapperManager:
    """
    Handles the logic of replacing an original attention class within a given model
    with a custom attention wrapper, based on user-specified model_name.
    Also allows restoring the original modules if needed, using a single
    recursive function.
    """

    def __init__(self, model_name: str, log_level: str = "INFO"):
        """
        Initializes the manager with a given model name.
        """
        self.logger = Logger(logname="ModuleWrapperManager", level=log_level)

        # Fetch the appropriate wrapper class for the given model name
        self.attention_wrapper_class = AttentionWrapperFactory.get_wrapper_class(model_name) # TODO: extend to support multiple module type for model
        # The original attention class name is fetched via a class method or attribute in the wrapper
        self.target_module_name = self.attention_wrapper_class.original_name() # TODO: extend to support multiple module type for model

        # Dictionary to store submodule_path -> original attention module
        self.original_modules = {}

    def __contains__(self, module_name:str):
        return module_name == self.target_module_name # TODO: extend to support multiple module type for model

    def substitute_attention_module(self, model: nn.Module) -> None:
        """
        Public method that performs the substitution of attention modules in the model.
        Logs each replacement. This will replace *all* modules whose class name
        matches `self.target_module_name`.
        """
        self._traverse_and_modify(model, parent_path="", mode="substitute")

    def restore_original_attention_module(self, model: nn.Module) -> None:
        """
        Public method that restores the original attention modules in the model.
        Logs each restoration.
        """
        self._traverse_and_modify(model, parent_path="", mode="restore")

    def _traverse_and_modify(self, module: nn.Module, parent_path: str, mode: str) -> None:
        """
        Recursively traverses `module` and either substitutes or restores each matching
        submodule, depending on `mode`.

        - mode="substitute": Replaces the original module (with class name == self.target_module_name)
                            with the wrapper, storing the original in self.original_modules.
        - mode="restore": Replaces the wrapper submodule (class name == self.attention_wrapper_class.__name__)
                        with the original module from self.original_modules.

        Args:
            module (nn.Module): The current module to inspect.
            parent_path (str): A string that tracks the 'path' of this submodule in the overall model hierarchy.
            mode (str): Either "substitute" or "restore".
        """
        for name, child in list(module.named_children()):
            # Identify the submodule path (e.g. "encoder.layer.0.attention")
            submodule_path = f"{parent_path}.{name}" if parent_path else name

            if mode == "substitute":
                # Look for the original module class name
                if child.__class__.__name__ == self.target_module_name:
                    # Store the original
                    self.original_modules[submodule_path] = child
                    # Wrap it
                    wrapped_module = self.attention_wrapper_class(child)
                    setattr(module, name, wrapped_module)

                    self.logger.info(
                        f"Substituted '{submodule_path}' with wrapper for {self.target_module_name}."
                    )
                else:
                    # Recurse
                    self._traverse_and_modify(child, submodule_path, mode="substitute")

            elif mode == "restore":
                # Look for the wrapper class name
                if child.__class__.__name__ == self.attention_wrapper_class.__name__:
                    if submodule_path in self.original_modules:
                        original_module = self.original_modules[submodule_path]
                        setattr(module, name, original_module)
                        self.logger.info(
                            f"Restored '{submodule_path}' to original {self.target_module_name}."
                        )
                    else:
                        self.logger.warning(
                            f"Found a wrapped submodule '{submodule_path}' but no original stored. Skipping."
                        )
                else:
                    # Recurse
                    self._traverse_and_modify(child, submodule_path, mode="restore")
