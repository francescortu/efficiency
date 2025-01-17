import torch
import torch.nn as nn
from typing import Union
from easyroutine.interpretability.activation_cache import ActivationCache
from easyroutine.interpretability.hooked_model import HookedModel
from easyroutine.logger import Logger
from tqdm import tqdm
from copy import deepcopy

class LogitLens:
    def __init__(self, unembedding_matrix, last_layer_norm: nn.Module, model_name: str):
        self.unembed = deepcopy(unembedding_matrix)
        self.norm = deepcopy(last_layer_norm)
        self.logger = Logger(
            logname = "LogitLens",
            level="INFO"
        )
        self.model_name = model_name
        
    def __repr__(self):
        return f"LogitLens({self.model_name})"
    @classmethod
    def from_model(cls, model: HookedModel) -> 'LogitLens':
        return cls(model.get_lm_head(), model.get_last_layernorm(), model.config.model_name)
    
    @classmethod
    def from_model_name(cls, model_name: str) -> 'LogitLens':
        model = HookedModel.from_model_name(model_name)
        cls = cls.from_model(model)
        del model
        torch.cuda.empty_cache()
        return cls
    
    def to(self, device: Union[str, torch.device]):
        # move 
        self.unembed = self.unembed.to(device)
        self.norm = self.norm.to(device)
        
    def device(self) -> torch.device:
        if self.unembed.device != self.norm.weight.device:
            self.unembed = self.unembed.to(self.norm.weight.device)
        return self.unembed.device
    
    def get_vocab_size(self):
        return self.unembed.shape[1]
    
    def get_keys(self, activations: ActivationCache, key: str):
        # check if is a format key ("resid_out_{i}")
        keys = []
        if "{i}" in key:
            # check if exists and how many starting from 0
            i = 0
            while activations.get(f"{key.format(i=i)}") is not None:
                keys.append(f"{key.format(i=i)}")
                i += 1
            if i == 0:
                raise KeyError(f"Key {key} not found in activations")
            else:
                self.logger.info(f"Key {key} found in activations with {i} elements")
                return keys
        else:
            if activations.get(key) is None:
                raise KeyError(f"Key {key} not found in activations")
            else:
                self.logger.info(f"Key {key} found in activations")
                return [key]

    def compute(
        self,
        activations: ActivationCache,
        target_key: str,
        apply_norm: bool = True,
        apply_softmax: bool = False,
    ) -> dict:
        """
        Compute the logit lens on the activations given at the target_key.
        
        Arguments:
            activations (ActivationCache): the activations store where to get the activations from
            target_key (str): the key where apply the logit lens. E.g. "resid_out_0" or "resid_out_{i}"
            apply_norm (bool): whether to apply the last layer norm before the unembedding matrix
            apply_softmax (bool): whether to apply the softmax after the unembedding matrix
            
        Returns:
            dict: a dictionary with the logit lens for each key found in the activations
                    
        """
        keys = self.get_keys(activations, target_key)
        
        logit_lens = {}
        for key in tqdm(keys, total=len(keys), desc=f"Computing Logit Lens of {target_key}"):
            act = activations.get(key).to(self.device())
            if apply_norm:
                act = self.norm(act)
            logits = torch.matmul(act, self.unembed.T)
            if apply_softmax:
                logits = torch.softmax(logits, dim=-1)
            logit_lens[f"logit_lens_{key}"] = logits
            
        return logit_lens
            
    


    