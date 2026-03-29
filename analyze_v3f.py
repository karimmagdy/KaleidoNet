"""Analyze v3f checkpoint: actual inference FLOPs with hard masks."""
import sys
sys.path.insert(0, ".")
import torch
from kaleidonet.model import KaleidoNet
from kaleidonet.metrics.flops import FLOPsCounter
from kaleidonet.core.elastic import ElasticLinear, ElasticAttention

# Build model with same config
model = KaleidoNet(
    embed_dim=192, num_blocks=4, num_heads=6, num_experts=4, top_k=1,
    num_classes=100, vocab_size=0, image_size=32, patch_size=4,
    elastic=True, drop_path_rate=0.1,
)

# Load checkpoint
ckpt = torch.load("checkpoints/latest.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()

print("=" * 60)
print("v3f Checkpoint Analysis")
print("=" * 60)

# Per-layer mask stats
print("\n--- Per-layer mask logit analysis ---")
for name, m in model.named_modules():
    if isinstance(m, ElasticLinear):
        logits = m.mask_logits.data
        active = (logits >= 0).sum().item()
        total = logits.numel()
        print(f"  {name}: {active}/{total} active ({active/total:.1%}) | "
              f"logit range [{logits.min():.2f}, {logits.max():.2f}]")
    elif isinstance(m, ElasticAttention):
        logits = m.head_mask_logits.data
        active = (logits >= 0).sum().item()
        total = logits.numel()
        print(f"  {name}: {active}/{total} heads ({active/total:.1%}) | "
              f"logit range [{logits.min():.2f}, {logits.max():.2f}]")

# FLOPs comparison
counter = FLOPsCounter()
seq_len = 64  # 32x32/4x4 = 64 patches

# Dense FLOPs
dense_flops = counter.count_dense(model, batch_size=1, seq_len=seq_len)

# Active FLOPs (using hard masks in eval mode)
active_info = counter.count(model, batch_size=1, seq_len=seq_len)
active_flops = active_info["total_active_flops"]

# Param counts
param_info = model.count_active_params()

print(f"\n--- FLOPs Summary ---")
print(f"Dense ViT baseline FLOPs:   236,733,640")
print(f"KaleidoNet dense FLOPs:     {dense_flops:,}")
print(f"KaleidoNet active FLOPs:    {active_flops:,}")
print(f"FLOPs reduction vs dense:   {active_flops/dense_flops:.1%} of KaleidoNet dense")
print(f"FLOPs vs Dense ViT:         {active_flops/236_733_640:.1%}")
print(f"Speedup vs Dense ViT:       {236_733_640/max(active_flops,1):.2f}x")

print(f"\n--- Params Summary ---")
print(f"Total params:    {param_info['total_params']:,}")
print(f"Active params:   {param_info['active_params']:,}")
print(f"Active fraction: {param_info['active_fraction']:.1%}")

print(f"\n--- Accuracy Summary ---")
print(f"Dense ViT:       36.11% val_acc")
print(f"KaleidoNet v2:   33.53% val_acc (no compression)")
print(f"KaleidoNet v3f:  31.24% val_acc (compressed)")
print(f"Accuracy gap:    {36.11 - 31.24:.2f}pp vs dense ViT")
print(f"Accuracy ratio:  {31.24/36.11:.1%} of dense ViT")
