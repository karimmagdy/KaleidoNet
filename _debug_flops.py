"""Debug: check what active_flops returns with logits at 0."""
import torch
import sys
sys.path.insert(0, '.')
from kaleidonet.model import KaleidoNet
from kaleidonet.metrics.flops import FLOPsCounter

model = KaleidoNet(embed_dim=192, num_blocks=4, num_heads=6, num_experts=4, top_k=1,
                   num_classes=100, vocab_size=0, image_size=32, patch_size=4, elastic=True)
model.train()

# Forward pass
x = torch.randn(1, 3, 32, 32)
out = model({'images': x})

print(f"active_flops from forward: {out['active_flops']:,}")
print(f"diff_flops from forward:   {out['diff_flops']:.0f}")
print(f"budget would be 50% of init: {out['active_flops'] * 0.5:,.0f}")
print(f"violation = (actual - budget) / budget = {(out['active_flops'] - out['active_flops'] * 0.5) / (out['active_flops'] * 0.5):.3f}")

# Check individual layer active_flops
counter = FLOPsCounter()
info = counter.count(model, batch_size=1, seq_len=64)  # seq_len=64 for 32x32 / 4x4 patches
print(f"\nTotal active FLOPs (FLOPsCounter): {info['total_active_flops']:,}")
for name, flops in info['breakdown'].items():
    print(f"  {name}: {flops:,}")

# Check active_width for some elastic layers
for name, m in model.named_modules():
    if hasattr(m, 'active_width'):
        print(f"\n{name}: active_width={m.active_width}/{m.out_features}")
    elif hasattr(m, 'active_heads'):
        print(f"\n{name}: active_heads={m.active_heads}/{m.num_heads}")
