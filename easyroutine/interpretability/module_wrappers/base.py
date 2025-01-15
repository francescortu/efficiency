import torch.nn as nn
import torch

class AttentionMatrixHookModule(nn.Module):
    """Computation of the attention matrix. *Note*: it has been added just for adding custom hooks."""
    
    def forward(
            self,
            attention_matrix: torch.Tensor,
    ):
        return attention_matrix

class BaseAttentionWrapper(nn.Module):
    """
    A base class for wrapping an original attention module.

    Provides:
        `_orig_module` to store the real (unwrapped) attention.
        A robust `__getattr__` that checks:
            1) self.__dict__
            2) self._modules
            3) the base class
            4) fallback to `_orig_module`
    """

    def __init__(self, original_module: nn.Module):
        super().__init__()
        # store the original module in a private attribute
        object.__setattr__(self, "_orig_module", original_module)

    def __getattr__(self, name: str):
        """
        If name is not in this wrapper, fall back to the original module.
        Also checks `self._modules` for submodules, because PyTorch
        automatically places them there.
        """
        # 1) get this wrapper's __dict__
        wrapper_dict = object.__getattribute__(self, "__dict__")

        # 2) if name is in our own instance dictionary, return it
        if name in wrapper_dict:
            return wrapper_dict[name]

        # 3) if name is in our submodules, return it
        modules_dict = wrapper_dict["_modules"]
        if name in modules_dict:
            return modules_dict[name]

        # 4) check if name is in our class (methods, etc.)
        cls = object.__getattribute__(self, "__class__")
        if hasattr(cls, name):
            return getattr(cls, name)

        # 5) fallback to _orig_module
        orig = wrapper_dict["_orig_module"]
        return getattr(orig, name)

    @staticmethod
    def original_name() -> str:
        """
        By default, you might override this in each derived class if you want
        your manager code to know which original class name this wrapper replaces.
        """
        return "BaseAttention"