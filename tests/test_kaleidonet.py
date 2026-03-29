"""
Smoke tests for KaleidoNet core components.

Run: pytest tests/ -v
"""

import torch
import pytest


def test_elastic_linear_forward():
    from kaleidonet.core.elastic import ElasticLinear
    layer = ElasticLinear(64, 128, min_width=8)
    x = torch.randn(2, 10, 64)
    out = layer(x)
    assert out.shape == (2, 10, 128)
    assert layer.active_width > 0
    assert layer.active_width <= 128


def test_elastic_linear_target_width():
    from kaleidonet.core.elastic import ElasticLinear
    layer = ElasticLinear(64, 128, min_width=8)
    x = torch.randn(2, 10, 64)
    out = layer(x, target_width=32)
    assert out.shape == (2, 10, 128)
    # With target_width=32, only 32 neurons should have nonzero output
    nonzero_cols = (out.abs().sum(dim=(0, 1)) > 0).sum().item()
    assert nonzero_cols == 32


def test_elastic_linear_eval_mask_is_deterministic():
    from kaleidonet.core.elastic import ElasticLinear
    layer = ElasticLinear(4, 6, min_width=2)
    layer.eval()
    with torch.no_grad():
        layer.mask_logits.copy_(torch.tensor([-2.0, 1.5, -1.0, 0.2, -0.5, 0.8]))

    mask_a = layer.get_mask()
    mask_b = layer.get_mask()

    expected = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    assert torch.equal(mask_a, mask_b)
    assert torch.equal(mask_a, expected)


def test_elastic_linear_eval_sparse_path_matches_dense_masked_output():
    from kaleidonet.core.elastic import ElasticLinear
    layer = ElasticLinear(4, 6, min_width=2)
    x = torch.randn(2, 3, 4)
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight))
        layer.bias.copy_(torch.randn_like(layer.bias))
        layer.mask_logits.copy_(torch.tensor([-2.0, 1.5, -1.0, 0.2, -0.5, 0.8]))

    layer.eval()
    out = layer(x)

    active_indices = torch.tensor([1, 3, 5])
    dense = torch.nn.functional.linear(x, layer.weight, layer.bias)
    expected = torch.zeros_like(dense)
    expected.index_copy_(-1, active_indices, dense.index_select(-1, active_indices))

    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)


def test_elastic_attention_forward():
    from kaleidonet.core.elastic import ElasticAttention
    attn = ElasticAttention(embed_dim=64, num_heads=4, min_heads=1)
    x = torch.randn(2, 10, 64)
    out = attn(x)
    assert out.shape == (2, 10, 64)


def test_elastic_attention_target_heads():
    from kaleidonet.core.elastic import ElasticAttention
    attn = ElasticAttention(embed_dim=64, num_heads=4, min_heads=1)
    x = torch.randn(2, 10, 64)
    out = attn(x, target_heads=2)
    assert out.shape == (2, 10, 64)


def test_elastic_attention_eval_mask_is_deterministic():
    from kaleidonet.core.elastic import ElasticAttention
    attn = ElasticAttention(embed_dim=64, num_heads=4, min_heads=1)
    attn.eval()
    with torch.no_grad():
        attn.head_mask_logits.copy_(torch.tensor([-1.0, 0.3, -2.0, 1.2]))

    mask_a = attn.get_head_mask()
    mask_b = attn.get_head_mask()

    expected = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert torch.equal(mask_a, mask_b)
    assert torch.equal(mask_a, expected)


def test_elastic_attention_sparse_path_matches_dense():
    """Sparse attention (active heads only) should match dense + head mask output."""
    from kaleidonet.core.elastic import ElasticAttention
    attn = ElasticAttention(embed_dim=64, num_heads=4, min_heads=1)
    with torch.no_grad():
        # Prune heads 0 and 2 (logits < 0)
        attn.head_mask_logits.copy_(torch.tensor([-1.0, 0.3, -2.0, 1.2]))
    attn.eval()

    x = torch.randn(2, 8, 64)
    out_sparse = attn(x)

    # Dense reference: full QKV, mask inactive heads, proj
    B, N, C = x.shape
    import torch.nn.functional as F
    qkv_full = attn.qkv(x).reshape(B, N, 3, 4, 16).permute(2, 0, 3, 1, 4)
    q, k, v = qkv_full.unbind(0)
    scores = (q @ k.transpose(-2, -1)) * (16 ** -0.5)
    weights = scores.softmax(dim=-1)
    head_mask = torch.tensor([0.0, 1.0, 0.0, 1.0]).view(1, 4, 1, 1)
    weights = weights * head_mask
    dense_out = (weights @ v).transpose(1, 2).reshape(B, N, C)
    out_dense = attn.proj(dense_out)

    assert torch.allclose(out_sparse, out_dense, atol=1e-5, rtol=1e-5)


def test_export_pruned_state():
    """Model surgery should produce a compact state dict with fewer params."""
    from kaleidonet.core.elastic import ElasticLinear
    from kaleidonet.export import export_pruned_state

    # Simple model with one elastic layer
    model = torch.nn.Sequential(
        ElasticLinear(8, 16, min_width=2),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    )
    # Prune: keep only 4 of 16 outputs
    with torch.no_grad():
        logits = torch.full((16,), -1.0)
        logits[[2, 5, 9, 14]] = 1.0
        model[0].mask_logits.copy_(logits)
    model.eval()

    result = export_pruned_state(model)
    assert result["pruned_params"] < result["original_params"]
    assert len(result["surgery_summary"]) == 1
    assert result["surgery_summary"][0]["pruned_out"] == 4
    # Check the weight shape in the compact state dict
    assert result["state_dict"]["0.weight"].shape == (4, 8)


def test_seed_reproducibility():
    """Setting the same seed should produce identical model weights."""
    from kaleidonet.training.trainer import set_seed
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_moe_layer_forward():
    from kaleidonet.routing.moe import MoELayer
    moe = MoELayer(embed_dim=64, num_experts=4, top_k=1)
    x = torch.randn(2, 10, 64)
    out, aux = moe(x)
    assert out.shape == (2, 10, 64)
    assert "load_balance_loss" in aux
    assert "expert_utilization" in aux


def test_moe_layer_elastic():
    from kaleidonet.routing.moe import MoELayer
    moe = MoELayer(embed_dim=64, num_experts=4, top_k=2, elastic_experts=True)
    x = torch.randn(2, 10, 64)
    out, aux = moe(x)
    assert out.shape == (2, 10, 64)


def test_morph_controller():
    from kaleidonet.morphing.controller import MorphController
    controller = MorphController(num_layers=4)
    stats = torch.randn(16)  # 4 layers * 4 features
    fractions = controller(stats)
    assert fractions.shape == (4,)
    assert (fractions >= 0).all() and (fractions <= 1).all()


def test_lagrangian_budget():
    from kaleidonet.morphing.lagrangian import LagrangianBudgetManager
    mgr = LagrangianBudgetManager(flops_budget=1_000_000)
    penalty = mgr.compute_penalty(2_000_000)
    assert penalty.item() > 0  # Over budget → positive penalty
    mgr.dual_step(2_000_000)
    assert mgr.lambda_val.item() > 0.01  # Lambda should increase


def test_fractal_block():
    from kaleidonet.growth.fractal import FractalBlock
    block = FractalBlock(dim=64, depth=2, elastic=True)
    x = torch.randn(2, 10, 64)
    out = block(x)
    assert out.shape == (2, 10, 64)


def test_fractal_net():
    from kaleidonet.growth.fractal import FractalNet
    net = FractalNet(dim=64, num_blocks=3, fractal_depth=2, elastic=True)
    x = torch.randn(2, 10, 64)
    out, aux_logits = net(x)
    assert out.shape == (2, 10, 64)
    assert len(aux_logits) == 0  # No aux heads added


def test_fractal_net_early_exit():
    from kaleidonet.growth.fractal import FractalNet
    net = FractalNet(dim=64, num_blocks=4, fractal_depth=2)
    net.add_aux_heads(num_classes=10)
    x = torch.randn(2, 10, 64)
    out, aux_logits = net(x, exit_after=2)
    assert out.shape == (2, 10, 64)
    assert len(aux_logits) == 2


def test_patch_tokenizer():
    from kaleidonet.tokenizers.vision import PatchTokenizer
    tok = PatchTokenizer(image_size=32, patch_size=4, embed_dim=64)
    images = torch.randn(2, 3, 32, 32)
    tokens = tok(images)
    assert tokens.shape == (2, 64, 64)  # 64 patches, 64 dim


def test_text_tokenizer():
    from kaleidonet.tokenizers.text import TextTokenizer
    tok = TextTokenizer(vocab_size=1000, max_seq_len=128, embed_dim=64)
    ids = torch.randint(0, 1000, (2, 20))
    tokens = tok(ids)
    assert tokens.shape == (2, 20, 64)


def test_universal_tokenizer_vision_only():
    from kaleidonet.tokenizers.universal import UniversalTokenizer
    tok = UniversalTokenizer(embed_dim=64, image_size=32, patch_size=4)
    images = torch.randn(2, 3, 32, 32)
    tokens, info = tok(images=images)
    assert tokens.shape[0] == 2
    assert tokens.shape[2] == 64
    assert "vision" in info["modality_ranges"]


def test_universal_tokenizer_multimodal():
    from kaleidonet.tokenizers.universal import UniversalTokenizer
    tok = UniversalTokenizer(embed_dim=64, image_size=32, patch_size=4, vocab_size=1000)
    images = torch.randn(2, 3, 32, 32)
    ids = torch.randint(0, 1000, (2, 20))
    tokens, info = tok(images=images, token_ids=ids)
    assert tokens.shape[0] == 2
    assert "vision" in info["modality_ranges"]
    assert "text" in info["modality_ranges"]
    # Total tokens = 64 patches + 1 sep + 20 text = 85
    assert tokens.shape[1] == 85


def test_backbone_forward():
    from kaleidonet.backbone.universal import UniversalBackbone
    backbone = UniversalBackbone(embed_dim=64, num_blocks=2, num_heads=4, num_experts=2, top_k=1)
    x = torch.randn(2, 10, 64)
    out, aux = backbone(x)
    assert out.shape == (2, 10, 64)
    assert "blocks_used" in aux
    assert aux["blocks_used"] >= 1


def test_kaleidonet_end_to_end():
    from kaleidonet.model import KaleidoNet
    model = KaleidoNet(
        embed_dim=64, num_blocks=2, num_heads=4, num_experts=2,
        top_k=1, num_classes=10, image_size=32, patch_size=4,
    )
    batch = {
        "images": torch.randn(2, 3, 32, 32),
        "targets": torch.randint(0, 10, (2,)),
        "task": "classify",
    }
    out = model(batch)
    assert out["logits"].shape == (2, 10)
    assert "backbone_aux" in out


def test_kaleidonet_backward():
    from kaleidonet.model import KaleidoNet
    model = KaleidoNet(
        embed_dim=64, num_blocks=2, num_heads=4, num_experts=2,
        top_k=1, num_classes=10, image_size=32, patch_size=4,
    )
    batch = {
        "images": torch.randn(2, 3, 32, 32),
        "targets": torch.randint(0, 10, (2,)),
        "task": "classify",
    }
    out = model(batch)
    loss = torch.nn.functional.cross_entropy(out["logits"], batch["targets"])
    loss.backward()
    # Check that gradients flow to elastic masks
    grads_exist = False
    for name, param in model.named_parameters():
        if "mask_logits" in name and param.grad is not None:
            grads_exist = True
            break
    assert grads_exist, "Gradients should flow to elastic mask logits"


def test_astar_router():
    from kaleidonet.routing.pathfinder import AStarRouter
    router = AStarRouter(embed_dim=64, num_nodes=8)
    x = torch.randn(20, 64)
    indices, weights, aux = router(x, top_k=2)
    assert indices.shape == (20, 2)
    assert weights.shape == (20, 2)
    assert "a_star_cost" in aux


def test_sinkhorn_router():
    from kaleidonet.routing.pathfinder import SinkhornRouter
    router = SinkhornRouter(embed_dim=64, num_experts=8)
    x = torch.randn(20, 64)
    indices, weights, aux = router(x, top_k=2)
    assert indices.shape == (20, 2)
    assert weights.shape == (20, 2)


def test_flops_counter():
    from kaleidonet.model import KaleidoNet
    from kaleidonet.metrics.flops import FLOPsCounter
    model = KaleidoNet(embed_dim=64, num_blocks=2, num_heads=4, num_experts=2, top_k=1, num_classes=10, image_size=32, patch_size=4)
    counter = FLOPsCounter()
    dense = counter.count_dense(model, batch_size=1, seq_len=64)
    assert dense > 0


def test_speedup_calculator():
    from kaleidonet.metrics.flops import SpeedupCalculator
    calc = SpeedupCalculator()
    calc.record_baseline(flops=1e12, accuracy=0.92)
    calc.record_method(flops=5e10, accuracy=0.92)
    s = calc.compute_speedup()
    assert s["flops_speedup"] == pytest.approx(20.0, rel=0.01)


def test_growth_scheduler():
    from kaleidonet.growth.scheduler import GrowthScheduler
    sched = GrowthScheduler(patience=3, min_improvement=0.01)
    # Simulate plateau
    assert not sched.step(1.0)   # improvement (vs inf), steps_since=0
    assert not sched.step(0.99)  # no improvement, steps_since=1
    assert not sched.step(0.99)  # steps_since=2
    assert sched.step(0.99)      # steps_since=3 == patience → trigger growth


def test_variance_transfer_widen():
    from kaleidonet.growth.scheduler import VarianceTransfer
    W = torch.randn(32, 64)
    b = torch.randn(32)
    W_new, b_new = VarianceTransfer.widen_linear(W, b, new_width=48)
    assert W_new.shape == (48, 64)
    assert b_new.shape == (48,)
    # Original weights preserved
    assert torch.allclose(W_new[:32], W)


def test_multi_objective_loss():
    from kaleidonet.training.loss import KaleidoNetLoss
    loss_fn = KaleidoNetLoss()
    task_loss = torch.tensor(2.0, requires_grad=True)
    backbone_aux = {
        "total_balance_loss": torch.tensor(0.5),
        "ponder_cost": 0.8,
        "blocks_used": 3,
        "mean_confidence": torch.tensor(0.9),
    }
    total, breakdown = loss_fn(task_loss, backbone_aux)
    assert total.item() > task_loss.item()  # Additional penalties should increase total
    assert "task_loss" in breakdown
    assert "balance_loss" in breakdown
