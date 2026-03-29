"""
Post-training pruning utility for KaleidoNet.

After training with soft Gumbel-sigmoid masks, this utility applies hard pruning
by setting mask_logits below a target percentile to -inf (permanently deactivated).

Usage:
    python3 run.py kaleidonet/pruning.py --checkpoint checkpoints/latest.pt --target 0.5
"""

from __future__ import annotations

import torch


def prune_by_percentile(model: torch.nn.Module, target_fraction: float) -> dict:
    """
    Prune mask_logits so that approximately `target_fraction` of neurons are kept.

    Neurons with the lowest mask_logits are set to -100 (masked off).
    Returns a dict with pruning statistics.

    Args:
        model: A KaleidoNet model with mask_logits parameters.
        target_fraction: Fraction of neurons to keep (0.5 = keep 50%).
    """
    # Collect all mask logits
    all_logits = []
    for name, param in model.named_parameters():
        if 'mask_logits' in name:
            all_logits.append(param.data.view(-1))

    if not all_logits:
        return {"pruned": 0, "total": 0, "kept_fraction": 1.0}

    flat = torch.cat(all_logits)
    total = flat.numel()

    # Find threshold: keep top `target_fraction` by logit value
    k = max(1, int(total * target_fraction))
    threshold = flat.topk(k).values[-1].item()

    # Apply pruning
    pruned = 0
    for name, param in model.named_parameters():
        if 'mask_logits' in name:
            below = param.data < threshold
            pruned += below.sum().item()
            param.data[below] = -100.0  # Hard off

    return {
        "pruned": pruned,
        "total": total,
        "kept": total - pruned,
        "kept_fraction": (total - pruned) / total,
        "threshold": threshold,
    }


def get_pruning_stats(model: torch.nn.Module) -> dict:
    """Get current pruning statistics from mask logits."""
    total = 0
    active = 0
    logit_values = []

    for name, param in model.named_parameters():
        if 'mask_logits' in name:
            n = param.numel()
            a = (param.data >= 0).sum().item()
            total += n
            active += a
            logit_values.append(param.data.view(-1))

    if not logit_values:
        return {"total": 0, "active": 0, "active_fraction": 1.0}

    flat = torch.cat(logit_values)
    return {
        "total": total,
        "active": active,
        "active_fraction": active / total if total > 0 else 1.0,
        "logits_mean": flat.mean().item(),
        "logits_std": flat.std().item(),
        "logits_min": flat.min().item(),
        "logits_max": flat.max().item(),
        "logits_negative_pct": (flat < 0).float().mean().item(),
    }


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, '.')

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("--target", type=float, default=0.5, help="Fraction of neurons to keep")
    parser.add_argument("--save", default=None, help="Save pruned checkpoint to this path")
    args = parser.parse_args()

    from kaleidonet.model import KaleidoNet

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))

    # Create model (infer config from checkpoint keys)
    model = KaleidoNet(
        embed_dim=192, num_blocks=4, num_heads=6, num_experts=4, top_k=1,
        num_classes=100, vocab_size=0, image_size=32, patch_size=4, elastic=True,
    )
    model.load_state_dict(model_state)

    print("=== Before pruning ===")
    stats = get_pruning_stats(model)
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\n=== Pruning to {args.target:.0%} ===")
    result = prune_by_percentile(model, args.target)
    for k, v in result.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== After pruning ===")
    stats = get_pruning_stats(model)
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Measure FLOPs after pruning
    model.eval()
    dummy = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out = model({"images": dummy})
    print(f"\n  Active FLOPs (post-prune): {out['active_flops']:,}")
    print(f"  vs Dense FLOPs:            682,444,800")
    print(f"  Compression ratio:         {out['active_flops']/682444800:.2%}")

    if args.save:
        ckpt["model_state"] = model.state_dict()
        torch.save(ckpt, args.save)
        print(f"\nPruned checkpoint saved to {args.save}")
