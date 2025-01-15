import re
from easyroutine.logger import Logger, LambdaLogger
import torch
from typing import List, Union
import contextlib

class ActivationCache():
    r"""
    Class to store and aggregate activation values from a model.
    It is a dictionary-like object with additional functionality to aggregate values.
    """
    
    def __init__(self):
        self.cache = {}
        self.logger = Logger(
            logname="ActivationCache",
            level="INFO",
        )
        
        self.valid_keys = (
            re.compile(r"resid_out_\d+"),
            re.compile(r"resid_in_\d+"),
            re.compile(r"resid_mid_\d+"),
            re.compile(r"attn_in_\d+"),
            re.compile(r"attn_out_\d+"),
            re.compile(r"avg_attn_pattern_L\dH\d+"),
            re.compile(r"pattern_L\dH\d+"),
            re.compile(r"values_\d+"),
            re.compile(r"input_ids"),
            re.compile(r"mapping_index")
        )
        
        self.aggregation_strategies = {}
        self.register_aggregation("mapping_index", lambda values: values[0])  # First value
        self.register_aggregation("pattern_", lambda values: values)  # Keep as list
        self.register_aggregation("input_ids", lambda values: values)  # Keep as list
        self.register_aggregation("offset", lambda values: [item for sublist in values for item in sublist])  # Flatten lists
        
        self.defferred_cache = False
        
    def __repr__(self) -> str:
        """
        Returns:
            str: A string representation of the ActivationCache object.
            
        Examples:
            >>> cache
            ActivationCache(resid_out_0, resid_in_0, resid_mid_0, attn_in_0, attn_out_0, avg_attn_pattern_L1H1, pattern_L1H1, values_L1H1)
        """
        return f"ActivationCache(`{', '.join(self.cache.keys())}`)"
    
    def __str__(self) -> str:
        """
        Returns:
            str: A string representation of the ActivationCache object.
            
        Examples:
            >>> print(cache)
            ActivationCache(resid_out_0: torch.Tensor([1, 2, 3, 4]), resid_in_0: torch.Tensor([1, 2, 3, 4]))
        """
        return f"ActivationCache({', '.join([f'{key}: {value}' for key, value in self.cache.items()])})"
        
    def __setitem__(self, key:str, value):
        """
        Set a key-value pair in the cache.
        
        Arguments:
            key (str): The key to store the value.
            value (Any): The value to store.
            
        Examples:
            >>> cache["resid_out_0"] = torch.randn(1, 3, 16)
        """
        if not any([pattern.match(key) for pattern in self.valid_keys]):
            self.logger.warning(f"Invalid key: {key}. Valid keys are: {self.valid_keys}. Could be a user-defined key.")
        self.cache[key] = value
        
    def __getitem__(self, key:str):
        """
        Get a value from the cache.
        
        Arguments:
            key (str): The key to retrieve the value.
            
        Examples:
            >>> cache["resid_out_0"]
            torch.Tensor([1, 2, 3, 4])
        """
        return self.cache[key]
    
    
    def __delitem__(self, key:str):
        """
        Remove a key-value pair from the cache.
        
        Arguments:
            key (str): The key to remove from the cache.
        """
        del self.cache[key]
        
    def __add__(self, other) -> "ActivationCache":
        """
        Overload the `+` operator to merge caches efficiently.
        Arguments:
            other (dict or ActivationCache): Another cache or dictionary to merge with.
        Returns:
            ActivationCache: A new ActivationCache object with merged data.
        """
        if not isinstance(other, (dict, ActivationCache)):
            raise TypeError("Can only add ActivationCache or dict objects.")

        new_cache = ActivationCache()
        new_cache.cache = {**self.cache, **(other.cache if isinstance(other, ActivationCache) else other)}
        return new_cache
    
    def __contains__(self, key):
        """
        Check if a key is present in the cache.
        Arguments:
            key (str): The key to check.
        Returns:
            bool: True if the key is present, False otherwise
        """
        return key in self.cache
    
    def items(self):
        """
        Just like the dictionary items method, returns a list of key-value pairs.
        """
        return self.cache.items()
    
    def keys(self):
        """
        Just like the dictionary keys method, returns a list of keys.
        """
        return self.cache.keys()
    
    def values(self):
        """
        Just like the dictionary values method, returns a list of values.
        """
        return self.cache.values()
    
    def update(self, other):
        """
        Updates the cache with values from an additional dictionary or ActivationCache object.
        Arguments:
            other (Union[dict, ActivationCache]): Dictionary or ActivationCache object to update with.
        """
        if isinstance(other, dict):
            self.cache.update(other)
        elif isinstance(other, ActivationCache):
            self.cache.update(other.cache)
        else:
            raise TypeError("Can only update with dict or ActivationCache objects.")
        

            
    def to(self, device: Union[str, torch.device]):
        """
        Moves the tensors in the cache to a specified device.
        
        Args:
            device (Union[str, torch.device]): The device to move the tensors to.
        """
        
        for key, value in self.cache.items():
            if hasattr(value, "to"):
                self.cache[key] = value.to(device)

    def cpu(self):
        """
        Moves the tensors in the cache to the CPU.
        """
        self.to("cpu")
        
    def cuda(self):
        """
        Moves the tensors in the cache to the GPU.
        """
        self.to("cuda")

        
    def register_aggregation(self, key_pattern, function):
        """
        Register a custom aggregation strategy for keys matching a pattern. In this way, you can define how to aggregate values for specific keys when merging caches.
        
        Arguments:
            key_pattern (str): The key or prefix to match.
            function (callable): The function to apply for aggregation.
            
        Examples:
            >>> cache.register_aggregation("values_", lambda values: torch.stack(values, dim=0))
        """
        self.aggregation_strategies[key_pattern] = function
    
    
    def default_aggregation(self, values):
        """
        Default aggregation strategy for keys without a custom strategy.
        Handles tensors, lists, and scalars.
        
        Arguments:
            values (List): List of values to aggregate.
            
        Returns:
            Union[torch.Tensor, List, Any]: The aggregated value.
        """
        if isinstance(values[0], torch.Tensor):
            try:
                return torch.cat(values, dim=0)
            except RuntimeError:
                return torch.stack(values, dim=0)
        elif isinstance(values[0], list):
            return [item for sublist in values for item in sublist]
        else:
            return values[0]  # Fallback to the first value
        
    @contextlib.contextmanager
    def deferred_mode(self):
        """
        Context manager to enable deferred aggregation.
        Collects all external caches in a list and aggregates them at the end of the context.
        This is most similar to the old way of using the `cat` method. It could (or could not) be more efficient.
        The main difference to direct calls to `cat` is that the cache is not updated until the end of the context, in this way the torch.cat, torch.stack and the other strategies are called only once.
        It will require more memory, but it could be more efficient.
        
        Examples:
            >>> with cache.deferred_mode():
            >>>     cache.cat(external_cache1)
            >>>     cache.cat(external_cache2)
        """
        self.deferred_cache = []
        try:
            yield self
            # Perform aggregation at the end of the context
            for external_cache in self.deferred_cache:
                self.cat(external_cache)
        finally:
            # Clear the deferred cache
            self.deferred_cache = None
        
    def cat(self, external_cache):
        """
        Merge the current cache with an external cache using aggregation strategies.
        
        Arguments:
            external_cache (ActivationCache): The external cache to merge with.
        
        
        Examples:
            >>> a, b = ActivationCache(), ActivationCache()
            >>> a["values_0"] = torch.tensor([1, 2])
            >>> b["values_0"] = torch.tensor([1, 4])
            >>> a.cat(b)
            >>> print(a["values_0"].shape)
            torch.Size([2,1])
            >>> print(a["values_0"])
            tensor([[2], [4]]
        """
        if not isinstance(external_cache, ActivationCache):
            raise TypeError("external_cache must be an instance of ActivationCache")

        # Case 1: Initialize self if it's empty
        if not self.cache and external_cache.cache:
            self.update(external_cache.cache)
            return

        # Case 2: Ensure both caches have the same keys
        self_keys = set(self.cache.keys())
        external_keys = set(external_cache.cache.keys())

        if self_keys != external_keys:
            raise ValueError(
                f"Key mismatch: self has {self_keys - external_keys}, "
                f"external has {external_keys - self_keys}"
            )

        # Case 3: Aggregate matching keys using registered strategies or default
        for key in self.cache:
            # Check for a custom aggregation strategy
            for pattern, strategy in self.aggregation_strategies.items():
                if key.startswith(pattern):
                    self.cache[key] = strategy([self.cache[key], external_cache[key]])
                    break
            else:
                # Use the default aggregation if no custom strategy matches
                self.cache[key] = self.default_aggregation(
                    [self.cache[key], external_cache[key]]
                )
                
                
                
                
                
                
        