"""Analyze the trained KaleidoNet model's compression state."""
from __future__ import annotations
import torch
from kaleidonet.model import KaleidoNet
from kaleidonet.core.elastic import ElasticLinear, ElasticAttention

# Load checkpoint
ckpt = torch.load("checkpoints/latest.pt", map_location="cpu", weights_only=False)
print(f"Checkpoint from step: {ckpt['global_step']}")

# Rebuild model
model = KaleidoNet(
    embed_dim=192, num_heads=6, num_experts=4, num_blocks=4,
    image_size=32, patch_size=4, num_classes=100, vocab_size=0,
    top_k=1, elastic=True, drop_path_rate=0.1,
)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Check mask logits distribution
print("\n=== Mask Logits Analysis ===")
total_neurons = 0
active_neurons = 0
for name, m in model.named_modules():
    if isinstance(m, ElasticLinear):
        logits = m.mask_logits.data
        sig = torch.sigmoid(logits)
        hard = (logits > 0).sum().item()
        total_neurons += logits.numel()
        active_neurons += hard
        print(f"  {name}: logits=[{logits.min():.2f}, {logits.max():.2f}], "
              f"sigmoid=[{sig.min():.3f}, {sig.max():.3f}], "
              f"hard_active={hard}/{logits.numel()} ({hard/logits.numel():.0%})")
    elif isinstance(m, ElasticAttention):
        logits = m.head_mask_logits.data
        sig = torch.sigmoid(logits)
        hard = (logits > 0).sum().item()
        total_neurons += logits.numel()
        active_neurons += hard
        print(f"  {name}: logits=[{logits.min():.2f}, {logits.max():.2f}], "
              f"sigmoid=[{sig.min():.3f}, {sig.max():.3f}], "
              f"hard_active={hard}/{logits.numel()} ({hard/logits.numel():.0%})")

print(f"\nOverall hard active: {active_neurons}/{total_neurons} ({active_neurons/total_neurons:.1%})")

# Compute actual FLOPs
batch = {"images": torch.randn(1, 3, 32, 32), "task": "classify"}
with torch.no_grad():
    out = model(batch)

print(f"\n=== FLOPs Comparison ===")
print(f"Active FLOPs (post-training): {out['active_flops']:,}")
print(f"Init active FLOPs:            239,516,160")
print(f"Dense FLOPs:                  682,444,800")
print(f"Post/Init ratio:              {out['active_flops']/239516160:.2%}")
print(f"Post/Dense ratio:             {out['active_flops']/682444800:.2%}")
print(f"\nDense ViT FLOPs:              236,733,640")
print(f"KaleidoNet speedup vs Dense:  {236733640/out['active_flops']:.2f}x")
