# EasyRoutine

This is just a simple collection of routines that I use frequently. I have found that I often need to do the same things over and over again, so I have created this repository to store them. I hope you find them useful.

## Installation


## Interpretability
The interpretability module contains wrapper of huggingface LLM/VLM that help to perform interpretability tasks on the model. Currently, it supports:
- Extract activations of any component of the model
- Perform ablation study on the model during inference
- Perform activation patching on the model during inference

### Load the model
```python
from easyroutine.interpretability import HookedModel
```