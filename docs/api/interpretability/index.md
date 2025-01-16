# API Introduction
`easyroutine.interpretability` is the module that implement code for extract the hidden rappresentations of HuggingFace LLMs and intervening on the forward pass.

## Simple Tutorial

```python
# First we need to import the HookedModel and the config classes
from easyroutine.interpretability import HookedModel, ExtractionConfig

# Then we can create the hooked model
hooked_model = HookedModel.from_pretrained(model_name="mistral-community/pixtral-12b", device_map = "auto")

# Now let's define a simple dataset
dataset = [
    "This is a test",
    "This is another test"
]

tokenizer = hooked_model.get_tokenizer()

dataset = tokenizer(dataset, padding=True, truncation=True, return_tensors="pt") 

cache = hooked_model.extract_cache(
    dataset,
    target_token_positions = ["last"],
    extraction_config = ExtractionConfig(
        extract_resid_out = True
    )
)

```
