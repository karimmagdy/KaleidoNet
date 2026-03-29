"""
Model surgery: export a trained KaleidoNet into a physically pruned model.

After training, elastic layers still carry dormant weights (zeroed by masks).
This module removes those weights entirely, producing a compact model for
inference benchmarking that does NOT rely on index_select at runtime.

Usage:
    from kaleidonet.export import export_pruned_state
    pruned = export_pruned_state(trained_model)
    # pruned is a dict with surgery summary and the compact state dict
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from kaleidonet.core.elastic import ElasticAttention, ElasticLinear


def _surgery_elastic_linear(
    module: ElasticLinear,
    prefix: str,
    state: OrderedDict,
    summary: list[dict],
) -> None:
    """Replace ElasticLinear state with a compact Linear state."""
    active = module._get_active_indices()
    kept = active.numel()

    # Compact weight: (kept, in_features)
    state[f"{prefix}.weight"] = module.weight.data.index_select(0, active).clone()
    if module.bias is not None:
        state[f"{prefix}.bias"] = module.bias.data.index_select(0, active).clone()

    summary.append({
        "layer": prefix,
        "type": "ElasticLinear",
        "original_out": module.out_features,
        "pruned_out": kept,
        "kept_fraction": kept / module.out_features,
    })


def _surgery_elastic_attention(
    module: ElasticAttention,
    prefix: str,
    state: OrderedDict,
    summary: list[dict],
) -> None:
    """Replace ElasticAttention state with compact QKV/proj weights for active heads only."""
    active = module._get_active_head_indices()
    h = active.numel()
    D = module.head_dim
    C = module.embed_dim

    # Build row indices for active heads in QKV weight (3*C, C)
    offsets = active * D
    per_head = torch.arange(D, device=active.device)
    head_rows = (offsets.unsqueeze(1) + per_head.unsqueeze(0)).reshape(-1)
    qkv_rows = torch.cat([head_rows, head_rows + C, head_rows + 2 * C])

    state[f"{prefix}.qkv.weight"] = module.qkv.weight.data.index_select(0, qkv_rows).clone()
    if module.qkv.bias is not None:
        state[f"{prefix}.qkv.bias"] = module.qkv.bias.data.index_select(0, qkv_rows).clone()

    # Proj: (C, C) → keep only active-head input columns
    state[f"{prefix}.proj.weight"] = module.proj.weight.data.index_select(1, head_rows).clone()
    if module.proj.bias is not None:
        state[f"{prefix}.proj.bias"] = module.proj.bias.data.clone()

    summary.append({
        "layer": prefix,
        "type": "ElasticAttention",
        "original_heads": module.num_heads,
        "pruned_heads": h,
        "kept_fraction": h / module.num_heads,
    })


def export_pruned_state(model: nn.Module) -> dict[str, Any]:
    """
    Walk a trained KaleidoNet and produce a surgery report + compact state dict.

    The returned state dict replaces elastic layers' weights with compact
    versions that have dormant neurons/heads physically removed.  Non-elastic
    parameters are copied verbatim.

    Returns:
        {
            "state_dict": OrderedDict — compact state dict,
            "surgery_summary": list — per-layer pruning info,
            "original_params": int,
            "pruned_params": int,
        }
    """
    summary: list[dict] = []
    compact_state = OrderedDict()

    # Collect elastic module prefixes for exclusion
    elastic_prefixes: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, ElasticLinear):
            _surgery_elastic_linear(module, name, compact_state, summary)
            elastic_prefixes.add(name)
        elif isinstance(module, ElasticAttention):
            _surgery_elastic_attention(module, name, compact_state, summary)
            elastic_prefixes.add(name)

    # Copy non-elastic parameters (skip mask_logits & parameters already handled)
    for name, param in model.named_parameters():
        # Skip mask logits entirely — they don't exist in the pruned model
        if "mask_logits" in name or "head_mask_logits" in name:
            continue
        # Skip parameters already handled by surgery
        skip = False
        for ep in elastic_prefixes:
            if name.startswith(ep + "."):
                skip = True
                break
        if skip:
            continue
        compact_state[name] = param.data.clone()

    # Copy non-elastic buffers
    for name, buf in model.named_buffers():
        if name not in compact_state:
            compact_state[name] = buf.clone()

    original_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(v.numel() for v in compact_state.values())

    return {
        "state_dict": compact_state,
        "surgery_summary": summary,
        "original_params": original_params,
        "pruned_params": pruned_params,
    }
