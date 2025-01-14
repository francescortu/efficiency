import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from transformers import Cache
from easyroutine.interpretability.module_wrappers.base import BaseAttentionWrapper, AttentionMatrixHookModule


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids=None,
    unsqueeze_dim=1
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    (batch, num_key_value_heads, seq_len, head_dim)
        -> (batch, num_attention_heads, seq_len, head_dim)
    """
    bsz, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, slen, head_dim)


class LlamaAttentionWrapper(BaseAttentionWrapper):
    """
    A wrapper around the original LlamaAttention. It has:
    - The same named attributes (q_proj, k_proj, etc.), which are references
        to the original module's submodules/parameters.
    - A private reference (`_orig_attn`) to the entire original attention,
        for falling back if something isn't found on the wrapper itself.
    - An additional `attention_matrix_hook` for intercepting attention.
    """

    @staticmethod
    def original_name():
        return "LlamaAttention"

    def __init__(self, original_attention: nn.Module):
        """
        Store references to all relevant submodules so the wrapper
        "feels" the same. Also store a reference to the original module
        in a private attribute for fallback.
        """
        super().__init__(original_attention)

        # This is the private reference to the entire original attention.
        # We'll fallback to it for any attribute we haven't explicitly set.
        object.__setattr__(self, "_orig_attn", original_attention)

        # Now replicate the original attention's submodules as attributes of *this* wrapper.
        # These are direct references, not new modules:
        self.q_proj = original_attention.q_proj
        self.k_proj = original_attention.k_proj
        self.v_proj = original_attention.v_proj
        self.o_proj = original_attention.o_proj
        self.rotary_emb = original_attention.rotary_emb
        
        # Copy over any scalar attributes you need
        self.num_heads = original_attention.num_heads
        self.num_key_value_heads = original_attention.num_key_value_heads
        self.num_key_value_groups = original_attention.num_key_value_groups
        self.head_dim = original_attention.head_dim
        self.hidden_size = original_attention.hidden_size
        self.attention_dropout = original_attention.attention_dropout
        self.layer_idx = original_attention.layer_idx
        self.config = original_attention.config

        # Add your custom hook module
        self.attention_matrix_hook = AttentionMatrixHookModule()


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Re-implement the forward pass so we can intercept attention weights
        mid-forward. Because we have direct references to q_proj, k_proj, etc.
        in *this* object, we can just call them as usual.
        """

        bsz, q_len, _ = hidden_states.size()

        # If the original code handles pretraining_tp > 1, do that logic:
        if self.config.pretraining_tp > 1:
            raise NotImplementedError("Implement splitted logic if needed.")
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        # Apply RoPE
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # If we have a past_key_value cache
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Expand KV if needed
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Calculate attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Softmax in float32, cast back
        attn_weights = F.softmax(attn_weights.to(torch.float32), dim=-1).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # Intercept or modify the attention matrix here:
        attn_weights = self.attention_matrix_hook(attn_weights)

        # Multiply by values
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be {(bsz, self.num_heads, q_len, self.head_dim)}, got {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)

        # Final projection
        if self.config.pretraining_tp > 1:
            raise NotImplementedError("Implement splitted logic if needed.")
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value # type: ignore
