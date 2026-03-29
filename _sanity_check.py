"""Quick sanity check: model construction + forward/backward pass."""
import torch
import torch.nn as nn
from kaleidonet.model import KaleidoNet

model = KaleidoNet(
    embed_dim=192, num_blocks=4, num_heads=6, num_experts=4,
    top_k=1, num_classes=100, image_size=32, patch_size=4, elastic=True,
)
param_info = model.count_active_params()
print(f"Total params: {param_info['total_params']:,}")
print(f"Active params: {param_info['active_params']:,}")
print(f"Active fraction: {param_info['active_fraction']:.1%}")

batch = {
    "images": torch.randn(4, 3, 32, 32),
    "targets": torch.randint(0, 100, (4,)),
    "task": "classify",
}
out = model(batch)
loss = nn.CrossEntropyLoss()(out["logits"], batch["targets"])
loss.backward()
print(f"Loss: {loss.item():.4f}")
print(f"Logits shape: {out['logits'].shape}")
print(f"Blocks used: {out['backbone_aux']['blocks_used']}")
print(f"Ponder cost: {out['backbone_aux']['ponder_cost']:.2f}")
print("Forward+backward OK!")
