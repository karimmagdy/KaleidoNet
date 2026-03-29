"""Summarize STL-10 results."""
import json
import statistics

print("=== STL-10 Full Results ===\n")
for pattern, label in [("dense_vit_stl10", "Dense ViT"), ("kaleidonet_stl10", "KaleidoNet")]:
    accs, flops = [], []
    for s in [1, 2, 3]:
        d = json.load(open(f"results/{pattern}_seed{s}.json"))
        a = d.get("val_accuracy", d.get("best_val_acc", 0))
        fl = d.get("active_flops", d.get("flops_per_sample", 0))
        accs.append(a)
        flops.append(fl)
        print(f"  {label} seed {s}: acc={a:.4f}, FLOPs={fl/1e6:.1f}M")
    m = statistics.mean(accs)
    sd = statistics.stdev(accs)
    fm = statistics.mean(flops)
    print(f"  -> Mean: {m:.4f} +/- {sd:.4f}, FLOPs={fm/1e6:.1f}M\n")

# Retention
dense_mean = statistics.mean([json.load(open(f"results/dense_vit_stl10_seed{s}.json")).get("best_val_acc") for s in [1,2,3]])
kaln_mean = statistics.mean([json.load(open(f"results/kaleidonet_stl10_seed{s}.json")).get("val_accuracy") for s in [1,2,3]])
print(f"Retention: {kaln_mean/dense_mean*100:.1f}%")
print(f"FLOPs ratio: {308.1/535.5:.2f}x")
