# Module Wrapper 

## Introduction
Module Wrapper is the submodule that is responsible for managing the module wrappers. The module wrappers are essential to add custom hook where in the original transfomer codebase the hook is not available. For example, the `transformer` module does not have a hook to get the attention matrix of a head. The module wrapper is used to add this hook. The `module_wrapper`  submodel is composed of the following files:
    - `manager.py`: The manager file is responsible for managing the module wrappers. It is the standard interface to add the wrap around models.
    - `base.py`: The base file is the base class for the module wrapper. Implement a base form of a Wrapper class.
    - `model_name_attention.py`: The model name attention file is the module wrapper for the attention matrix of a single model. When add a new model, add a new file with the name `model_name_attention.py` and implement the `ModelNameAttention` class. It is basically a copy of the forward pass of the attention module with the addition of the hook to get the attention matrix. 

## Manager Wrappers and Abstract Base Class
### ::: easyroutine.interpretability.module_wrappers.manager 
### ::: easyroutine.interpretability.module_wrappers.base

## Specific Module Wrappers

### ::: easyroutine.interpretability.module_wrappers.llama_attention
### ::: easyroutine.interpretability.module_wrappers.chameleon_attention
### ::: easyroutine.interpretability.module_wrappers.T5_attention