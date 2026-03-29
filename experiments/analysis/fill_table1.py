"""Auto-fill experiments.tex Table 1 from result JSONs.

Reads all seed-level result files, computes mean ± std per
(dataset, method) pair, and rewrites paper/sections/experiments.tex
with actual numbers replacing 'results pending' placeholders.

Run:
    python experiments/analysis/fill_table1.py           # preview only
    python experiments/analysis/fill_table1.py --write   # update .tex file
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(ROOT, "results")
TEX_PATH = os.path.join(ROOT, "paper", "sections", "experiments.tex")

# Dataset mapping: internal slug → LaTeX block name in table
DATASETS = ["cifar100", "tiny_imagenet", "cifar10", "stl10"]
DS_LABELS = {
    "cifar100": "CIFAR-100",
    "tiny_imagenet": "Tiny-ImageNet",
    "cifar10": "CIFAR-10",
    "stl10": "STL-10",
}

# Method mapping: model field → LaTeX method name in table
METHODS = {
    "DenseViT": "Dense ViT",
    "MagnitudePruning": "Magnitude",
    "RandomPruning": "Random",
    "LinearSchedule": "Linear Schedule",
    "KaleidoNet": "KaleidoNet (cubic)",
}


def load_results() -> dict:
    """Load all individual seed JSON files grouped by (dataset, model)."""
    groups = {}
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        base = os.path.basename(f)
        if "summary" in base or "sweep" in base or "ablation" in base or "inference" in base:
            continue
        # Only include seed-numbered results for multi-seed stats
        if not re.search(r"seed\d+\.json$", base):
            continue
        try:
            with open(f) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, KeyError):
            continue

        model = data.get("model")
        dataset = data.get("dataset")
        if not model:
            continue

        # Infer dataset from filename if not in JSON
        if not dataset:
            fname = base.lower()
            for slug in sorted(DATASETS, key=len, reverse=True):
                if slug in fname:
                    dataset = slug
                    break
            if not dataset:
                continue

        key = (dataset, model)
        groups.setdefault(key, []).append(data)

    return groups


def get_acc(r: dict) -> float | None:
    acc = r.get("val_accuracy") or r.get("best_val_acc")
    if acc is not None:
        v = float(acc)
        return v * 100 if v < 1 else v
    return None


def get_flops_m(r: dict) -> float | None:
    flops = r.get("active_flops") or r.get("flops_per_sample")
    if flops is not None:
        return float(flops) / 1e6
    return None


def compute_stats(records: list[dict]):
    accs = [a for a in (get_acc(r) for r in records) if a is not None]
    flops = [f for f in (get_flops_m(r) for r in records) if f is not None]
    if not accs:
        return None
    acc_mean = np.mean(accs)
    acc_std = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
    flops_mean = np.mean(flops) if flops else None
    return {
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "flops_mean": flops_mean,
        "n_seeds": len(accs),
    }


def format_table_row(stats: dict, dense_acc: float | None, is_best_pruned: bool) -> str:
    """Format a LaTeX table row for one method's results."""
    acc_str = f"${stats['acc_mean']:.2f} \\pm {stats['acc_std']:.2f}$"
    flops_str = f"{stats['flops_mean']:.1f}" if stats['flops_mean'] else "---"
    if dense_acc and dense_acc > 0 and stats['acc_mean'] > 0:
        retention = (stats['acc_mean'] / dense_acc) * 100
        ret_str = f"{retention:.1f}"
    else:
        ret_str = "---"

    if is_best_pruned:
        acc_str = f"$\\mathbf{{{stats['acc_mean']:.2f} \\pm {stats['acc_std']:.2f}}}$"
        flops_str = f"\\textbf{{{flops_str}}}" if flops_str != "---" else "---"
        ret_str = f"\\textbf{{{ret_str}}}" if ret_str != "---" else "---"

    return f"{acc_str} & {flops_str} & {ret_str}"


def build_table(groups: dict) -> str:
    """Build the full LaTeX table body."""
    lines = []
    for ds in DATASETS:
        ds_label = DS_LABELS[ds]
        methods_order = ["DenseViT", "MagnitudePruning", "RandomPruning", "LinearSchedule", "KaleidoNet"]

        # Collect stats for all methods
        all_stats = {}
        for model_key in methods_order:
            key = (ds, model_key)
            if key in groups:
                s = compute_stats(groups[key])
                if s:
                    all_stats[model_key] = s

        if not all_stats:
            continue

        # Find dense acc for retention
        dense_acc = all_stats.get("DenseViT", {}).get("acc_mean")

        # Find best pruned method
        pruned_methods = ["MagnitudePruning", "RandomPruning", "LinearSchedule", "KaleidoNet"]
        best_acc = -1
        best_method = None
        for m in pruned_methods:
            if m in all_stats and all_stats[m]["acc_mean"] > best_acc:
                best_acc = all_stats[m]["acc_mean"]
                best_method = m

        lines.append(f"\\multirow{{5}}{{*}}{{{ds_label}}}")
        for i, model_key in enumerate(methods_order):
            method_name = METHODS[model_key]
            prefix = "& " if i > 0 else "& "
            # First row needs \multirow, rest are continuation
            if i == 0:
                prefix = ""

            if model_key in all_stats:
                is_best = (model_key == best_method)
                row_data = format_table_row(all_stats[model_key], dense_acc, is_best)
                if model_key == "DenseViT":
                    # Dense ViT: no retention
                    s = all_stats[model_key]
                    acc_str = f"${s['acc_mean']:.2f} \\pm {s['acc_std']:.2f}$"
                    flops_str = f"{s['flops_mean']:.1f}" if s['flops_mean'] else "---"
                    row_data = f"{acc_str} & {flops_str} & ---"
                if is_best:
                    method_name = f"\\textbf{{{method_name}}}"
                line = f"& {method_name} & {row_data} \\\\"
            else:
                line = f"& {method_name} & \\multicolumn{{3}}{{c}}{{\\emph{{results pending}}}} \\\\"

            lines.append(line)

        lines.append("\\midrule")

    # Remove last \midrule and replace with \bottomrule
    if lines and lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"

    return "\n".join(lines)


def update_tex(table_body: str, write: bool = False):
    """Replace the table body in experiments.tex."""
    with open(TEX_PATH) as f:
        content = f.read()

    # Find the table between \midrule (after header) and \bottomrule
    pattern = r"(\\midrule\n)(\\multirow.*?)(\\bottomrule)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        print("Could not find table body to replace in experiments.tex")
        print("Table body would be:")
        print(table_body)
        return

    new_content = content[:match.start(2)] + table_body + "\n" + content[match.start(3):]

    if write:
        with open(TEX_PATH, "w") as f:
            f.write(new_content)
        print(f"Updated {TEX_PATH}")
    else:
        print("Preview of table body (use --write to apply):")
        print(table_body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Write changes to .tex file")
    args = parser.parse_args()

    groups = load_results()
    print(f"Loaded results for {len(groups)} (dataset, model) combinations:")
    for (ds, model), records in sorted(groups.items()):
        stats = compute_stats(records)
        if stats:
            print(f"  {ds:20s} {model:20s} — {stats['n_seeds']} seeds, acc={stats['acc_mean']:.2f}±{stats['acc_std']:.2f}")

    print()
    table_body = build_table(groups)
    update_tex(table_body, write=args.write)


if __name__ == "__main__":
    main()
