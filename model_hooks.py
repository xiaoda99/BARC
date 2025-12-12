from collections import OrderedDict
from dataclasses import dataclass, field

import torch
import einops
from einops import rearrange

from transformers.modeling_attn_mask_utils import AttentionMaskConverter


@dataclass
class Outputs:
    inputs_embeds: torch.FloatTensor = None
    position_embeds: torch.FloatTensor = None
    attn_outputs: tuple = ()
    values: tuple = ()
    attn_outs: tuple = ()
    head_inputs: tuple = ()
    head_outputs: tuple = ()
    mlp_pre_acts: tuple = ()
    mlp_gates: tuple = ()
    avg_mlp_gates: dict = field(default_factory=dict)
    mlp_outputs: tuple = ()
    ln_states: tuple = ()
    hidden_states: tuple = ()
    attentions: tuple = ()
    logits: torch.FloatTensor = None
    labels: torch.LongTensor = None
    loss: torch.FloatTensor = None
    attn_attr: OrderedDict = field(default_factory=OrderedDict)


def set_hooks(model):
    for block in model.model.layers:
        block.input_layernorm.variance = None
        block.self_attn.attn_out = None
        block.post_attention_layernorm.variance = None
        block.mlp_output = None
    model.model.norm.variance = None


def del_hooks(model):
    for block in model.model.layers:
        if hasattr(block.input_layernorm, 'variance'): delattr(block.input_layernorm, 'variance')
        if hasattr(block.self_attn, 'attn_out'): delattr(block.self_attn, 'attn_out')
        if hasattr(block.post_attention_layernorm, 'variance'): delattr(block.post_attention_layernorm, 'variance')
        if hasattr(block, 'mlp_output'): delattr(block, 'mlp_output')
    if hasattr(model.model.norm, 'variance'): delattr(model.model.norm, 'variance')


def _to_cpu_pinned(tensor):
    """Move tensor to CPU with pinned memory for faster GPU transfers."""
    return tensor.to('cpu').pin_memory()


def get_outputs(model, output, device='cpu', pin_memory=True):
    """
    Collect and offload model outputs to CPU.
    
    Args:
        pin_memory: If True and device='cpu', use pinned memory for ~2x faster 
                   GPU transfers when reloading (32ms vs 67ms for 486MB).
    """
    attn_outs, mlp_outputs, ln_states = [], [], []
    to_device = _to_cpu_pinned if (device == 'cpu' and pin_memory) else lambda t: t.to(device)
    
    for block in model.model.layers:
        ln_states.append((to_device(block.input_layernorm.variance),
                          to_device(block.post_attention_layernorm.variance)))
        delattr(block.input_layernorm, 'variance')
        delattr(block.post_attention_layernorm, 'variance')
        attn_outs.append(to_device(block.self_attn.attn_out))
        delattr(block.self_attn, 'attn_out')
        mlp_outputs.append(to_device(block.mlp_output))
        delattr(block, 'mlp_output')
    ln_states.append(to_device(model.model.norm.variance))
    delattr(model.model.norm, 'variance')
    hidden_states = [to_device(hs) for hs in output.hidden_states]
    return Outputs(hidden_states=hidden_states, ln_states=ln_states, attn_outs=attn_outs, mlp_outputs=mlp_outputs)


def to_gpu(tensor, device, pos_ids=None):
    """Load tensor from CPU (pinned) memory to GPU efficiently."""
    if pos_ids is not None: tensor = tensor[:, pos_ids]
    return tensor.to(device, non_blocking=True)


def is_remote(model):
    """Check if model is a remote client (has compute_node_gradient method)."""
    return hasattr(model, 'compute_node_gradient')


# =============================================================================
# Unified get_xxx functions - work with local model or remote client
# Usage: get_head_output(model, r, layer, head, pos_ids)
# =============================================================================

def get_hidden_states(model, r, layer, pos_ids=None, **_):
    if is_remote(model):
        return model.get_hidden_states(sample_id=r.index, layer=layer, pos_ids=pos_ids)
    hs = r.outputs.hidden_states[layer]
    if pos_ids is not None: hs = hs[:, pos_ids]
    return hs


def get_head_output(model, r, layer, head, pos_ids=None, **_):
    if is_remote(model):
        return model.get_head_output(sample_id=r.index, layer=layer, head=head, pos_ids=pos_ids)
    outputs = r.outputs
    if head == model.config.num_attention_heads:  # mlp
        return to_gpu(outputs.mlp_outputs[layer], model.device, pos_ids)
    attn_out = to_gpu(outputs.attn_outs[layer], model.device, pos_ids)
    if head is not None:
        m = torch.zeros(attn_out.shape[2]).to(model.device, dtype=model.dtype)
        m[head] = 1.
        masked_attn_out = torch.einsum('bind,n->bind', attn_out, m)
        masked_attn_out = rearrange(masked_attn_out, 'b i n d -> b i (n d)')
    else:  # all heads
        mask = torch.eye(model.config.num_attention_heads).to(model.device, dtype=attn_out.dtype) # n*n
        masked_attn_out = torch.einsum('bind,mn->bimnd', attn_out, mask)
        masked_attn_out = rearrange(masked_attn_out, 'b i m n d -> b i m (n d)')
    return model.model.layers[layer].self_attn.o_proj(masked_attn_out)  # bie for one head or bine for all heads


def get_attn_output(model, r, layer, pos_ids=None, **_):
    if is_remote(model):
        return model.get_attn_output(sample_id=r.index, layer=layer, pos_ids=pos_ids)
    outputs = r.outputs
    attn_out = to_gpu(outputs.attn_outs[layer], model.device, pos_ids)
    attn_out = rearrange(attn_out, 'b i n d -> b i (n d)')
    return model.model.layers[layer].self_attn.o_proj(attn_out)


def get_attn_kwargs(batch_size, seq_length, device, dtype, pos_ids=None):
    attn_mask_converter = AttentionMaskConverter(is_causal=True)
    attention_mask = attn_mask_converter.to_causal_4d(
        batch_size, seq_length, seq_length, dtype=dtype, device=device)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    q_position_ids = None
    if pos_ids is not None:
        q_position_ids = position_ids[:, pos_ids]  # [bsz(=1), seq_len[pos_ids]]
        attention_mask = attention_mask[:, :, pos_ids, :]  # [bsz, 1, q_seq_len[pos_ids], kv_seq_len]
    return {'attention_mask': attention_mask, 'position_ids': position_ids, 'q_position_ids': q_position_ids}


_attn_weights_cache = {} # Cache for get_attn_weights results (avoids SDPA non-determinism)

def get_attn_weights(model, r, layer, head, pos_ids=None, use_cache=True, **_):
    if is_remote(model):
        return model.get_attn_weights(sample_id=r.index, layer=layer, head=head, pos_ids=pos_ids)
    
    outputs = r.outputs
    head_key = head if isinstance(head, (int, type(None))) else tuple(head)
    pos_key = tuple(pos_ids) if pos_ids is not None else None
    cache_key = (layer, head_key, pos_key)
    
    if use_cache and cache_key in _attn_weights_cache:
        return _attn_weights_cache[cache_key]
    
    self = model.model.layers[layer]
    hidden_states = to_gpu(outputs.hidden_states[layer], model.device)
    hidden_states = self.input_layernorm(hidden_states)
    q_hidden_states = hidden_states[:, pos_ids] if pos_ids is not None else None

    kwargs = get_attn_kwargs(hidden_states.shape[0], hidden_states.shape[1], model.device, model.dtype, pos_ids=pos_ids)
    if head is None: head = list(range(model.config.num_attention_heads))
    result = self.self_attn(hidden_states=hidden_states, q_hidden_states=q_hidden_states, 
                            output_attentions=True, **kwargs)[1][:, head].to('cpu')
    
    if use_cache:
        _attn_weights_cache[cache_key] = result
    return result


def clear_attn_weights_cache():
    """Clear the attention weights cache."""
    global _attn_weights_cache
    _attn_weights_cache.clear()


def get_attn_weights_cache_info():
    """Return cache statistics."""
    total_bytes = sum(t.numel() * t.element_size() for t in _attn_weights_cache.values())
    return {
        'entries': len(_attn_weights_cache),
        'memory_mb': total_bytes / (1024 * 1024),
        'keys': list(_attn_weights_cache.keys())
    }

