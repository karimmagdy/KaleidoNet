"""Quick sanity check for v3 init changes."""
import torch
import sys
sys.path.insert(0, '.')
from kaleidonet.model import KaleidoNet

model = KaleidoNet(embed_dim=192, num_blocks=4, num_heads=6, num_experts=4, top_k=1,
                   num_classes=100, vocab_size=0, image_size=32, patch_size=4, elastic=True)

# Check init mask logits
print("=== Mask logits at init ===")
for name, p in model.named_parameters():
    if 'mask_logits' in name:
        print(f'  {name}: mean={p.data.mean():.3f} min={p.data.min():.3f} max={p.data.max():.3f}')

# Forward pass
print("\n=== Forward pass ===")
x = torch.randn(1, 3, 32, 32)
out = model({'images': x})
print(f'  Active FLOPs: {out["active_flops"]:,}')
print(f'  Diff FLOPs: {out.get("diff_flops")}')

# Check soft active fraction
vals = []
for m in model.modules():
    if hasattr(m, 'mask_logits'):
        soft = torch.sigmoid(m.mask_logits).mean().item()
        vals.append(soft)
print(f'  Avg soft sigmoid(mask_logits): {sum(vals)/len(vals):.3f} (expected ~0.5)')

# Verify gradient flow
print("\n=== Gradient flow ===")
loss = out['logits'].sum()
diff_flops = out.get('diff_flops')
if diff_flops is not None:
    loss = loss + 0.01 * diff_flops
loss.backward()
grads = sum(1 for n, p in model.named_parameters() if 'mask_logits' in n and p.grad is not None and p.grad.abs().sum() > 0)
total = sum(1 for n, p in model.named_parameters() if 'mask_logits' in n)
print(f'  Mask logits with gradients: {grads}/{total}')
print("\nAll checks passed!" if grads == total else "\nWARNING: Some mask logits have no gradient!")
