"""Summarize all CIFAR-10 results."""
import json
import glob
import statistics

print("=== CIFAR-10 Full Results ===\n")
for pattern, label in [
    ("results/dense_vit_cifar10_seed*.json", "Dense ViT"),
    ("results/kaleidonet_cifar10_seed*.json", "KaleidoNet"),
    ("results/magnitude_pruning_cifar10_seed*.json", "Magnitude Pruning"),
    ("results/random_pruning_cifar10_seed*.json", "Random Pruning"),
]:
    files = sorted(glob.glob(pattern))
    if not files:
        continue
    accs, flops = [], []
    for f in files:
        d = json.load(open(f))
        a = d.get("val_accuracy", d.get("best_val_acc", 0))
        fl = d.get("active_flops", d.get("flops_per_sample", 0))
        accs.append(a)
        flops.append(fl)
        seed = f.split("seed")[1].split(".")[0]
        print(f"  {label} seed{seed}: acc={a:.4f}, FLOPs={fl/1e6:.1f}M")
    if len(accs) > 1:
        m = statistics.mean(accs)
        s = statistics.stdev(accs)
        fm = statistics.mean(flops)
        print(f"  -> Mean: {m:.4f} +/- {s:.4f}, FLOPs={fm/1e6:.1f}M")
    print()
