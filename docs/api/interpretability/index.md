# API Introduction
`easyroutine.interpretability` is the module that implement code for extract the hidden rappresentations of HuggingFace LLMs and intervening on the forward pass.

## Simple Tutorial
```python
from easyroutine.interpretability import HookedModel, HookedModelConfig

config = HookedModelConfig(
    model_name="mistral-community/pixtral-12b",
    device_map = "auto"
)

hooked_model = HookedModel(config)
```
