"""
Run multi-seed experiments for both KaleidoNet and Dense ViT.

Usage:
    python experiments/multi_seed_run.py                               # CIFAR-100, seeds 1,2,3
    python experiments/multi_seed_run.py --dataset tiny-imagenet       # Tiny-ImageNet
    python experiments/multi_seed_run.py --seeds 1 2 3 4 5             # custom seeds
    python experiments/multi_seed_run.py --model kaleidonet            # KaleidoNet only
    python experiments/multi_seed_run.py --model dense                 # Dense ViT only
    python experiments/multi_seed_run.py --steps 5000                  # custom step count
    python experiments/multi_seed_run.py --dataset tiny-imagenet --data-dir ./data/tiny-imagenet-200
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from typing import Optional


def run_experiment(
    script: str,
    seed: int,
    steps: int,
    label: str,
    use_launcher: bool = False,
    extra_args: Optional[list[str]] = None,
) -> dict:
    """Run a single experiment and return its results."""
    if use_launcher:
        cmd = [sys.executable, "-u", "run.py", script, "--seed", str(seed), "--steps", str(steps)]
    else:
        cmd = [sys.executable, "-u", script, "--seed", str(seed), "--steps", str(steps)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'='*60}")
    print(f"  {label} | seed={seed} | steps={steps}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=os.getcwd())
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode}) after {elapsed:.0f}s")
        return {"seed": seed, "status": "failed", "time_s": elapsed}

    print(f"\n  Completed in {elapsed:.0f}s")
    return {"seed": seed, "status": "ok", "time_s": elapsed}


def main():
    parser = argparse.ArgumentParser(description="Multi-seed experiment runner")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                        help="Seeds to run (default: 1 2 3)")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps per run")
    parser.add_argument("--model", choices=["both", "kaleidonet", "dense"], default="both",
                        help="Which model(s) to run")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "tiny-imagenet", "stl10"], default="cifar100",
                        help="Dataset to use")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Dataset root for datasets that require a local path (e.g. Tiny-ImageNet)")
    args = parser.parse_args()

    results_log = []

    DATASET_SCRIPTS = {
        "cifar10": {
            "kaleidonet": "experiments/baselines/train_cifar10.py",
            "dense": "experiments/baselines/dense_vit_cifar10.py",
            "extra_args": [],
        },
        "cifar100": {
            "kaleidonet": "experiments/baselines/train_cifar100.py",
            "dense": "experiments/baselines/dense_vit_baseline.py",
            "extra_args": [],
        },
        "tiny-imagenet": {
            "kaleidonet": "experiments/baselines/train_tiny_imagenet.py",
            "dense": "experiments/baselines/dense_vit_tiny_imagenet.py",
            "extra_args": ["--data-dir", "{data_dir}"],
        },
        "stl10": {
            "kaleidonet": "experiments/baselines/train_stl10.py",
            "dense": "experiments/baselines/dense_vit_stl10.py",
            "extra_args": [],
        },
    }

    ds_cfg = DATASET_SCRIPTS[args.dataset]
    kaleidonet_script = ds_cfg["kaleidonet"]
    dense_script = ds_cfg["dense"]
    extra_args: list[str] = [
        a.format(data_dir=args.data_dir or "./data/tiny-imagenet-200") for a in ds_cfg["extra_args"]
    ]

    if args.model in ("both", "kaleidonet"):
        for seed in args.seeds:
            r = run_experiment(
                kaleidonet_script,
                seed, args.steps, f"KaleidoNet {args.dataset} seed={seed}",
                use_launcher=True,
                extra_args=extra_args,
            )
            r["model"] = "KaleidoNet"
            results_log.append(r)

    if args.model in ("both", "dense"):
        for seed in args.seeds:
            r = run_experiment(
                dense_script,
                seed, args.steps, f"Dense ViT {args.dataset} seed={seed}",
                extra_args=extra_args,
            )
            r["model"] = "DenseViT"
            results_log.append(r)

    # Summary
    print("\n" + "=" * 60)
    print("Multi-Seed Run Summary")
    print("=" * 60)
    total_time = sum(r["time_s"] for r in results_log)
    ok = sum(1 for r in results_log if r["status"] == "ok")
    failed = sum(1 for r in results_log if r["status"] == "failed")
    print(f"  Total runs: {len(results_log)} ({ok} ok, {failed} failed)")
    print(f"  Total time: {total_time/60:.1f} min")
    for r in results_log:
        status = "OK" if r["status"] == "ok" else "FAILED"
        print(f"    {r['model']:12s} seed={r['seed']} -> {status} ({r['time_s']:.0f}s)")


if __name__ == "__main__":
    main()
