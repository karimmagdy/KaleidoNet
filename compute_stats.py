#!/usr/bin/env python3
"""
compute_stats.py — Aggregate convergence-run results and compute proper statistical evidence
for Table 1 (main comparison) and Table 3 (schedule add-on ablation).

Produces:
  - Per (method, dataset): mean, std, 95% bootstrap CI on the mean
  - Paired t-test p-value vs reference method (default: kaleidonet)
  - Cohen's d effect size vs reference method
  - LaTeX table fragments ready to drop into paper sections

Usage:
  python compute_stats.py                       # Auto-detect all methods, all datasets
  python compute_stats.py --reference dense     # Compare against dense baseline
  python compute_stats.py --table main          # Emit Table 1 LaTeX
  python compute_stats.py --table ablation      # Emit Table 3 LaTeX
  python compute_stats.py --json out.json       # Dump stats as JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats  # type: ignore

# --- paths ---
ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "convergence"

# --- canonical orderings ---
DATASETS = ["cifar10", "cifar100", "tiny_imagenet"]
DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "tiny_imagenet": "Tiny-ImageNet",
}

MAIN_METHODS = ["dense", "magnitude", "random", "linear", "lagrangian", "kaleidonet"]
ABLATION_METHODS = ["cubic_only", "masking_only", "dual_only", "linear", "kaleidonet"]

METHOD_LABELS = {
    "dense": "Dense ViT (reference)",
    "magnitude": "Magnitude + fine-tune",
    "random": "Random mask",
    "linear": "Linear schedule",
    "lagrangian": "Lagrangian (per-neuron)",
    "kaleidonet": "KaleidoNet (cubic + masking + dual-rate)",
    "cubic_only": "Cubic schedule only",
    "masking_only": "Cubic + gradient masking",
    "dual_only": "Cubic + dual-rate",
}


def load_results(method: str, dataset: str) -> list[dict]:
    """Load all per-seed JSONs for a (method, dataset)."""
    seeds = []
    for seed in (1, 2, 3):
        p = RESULTS_DIR / f"{method}_{dataset}_seed{seed}.json"
        if p.exists():
            with open(p) as f:
                seeds.append(json.load(f))
    return seeds


def get_accuracy(results: list[dict]) -> np.ndarray:
    """Extract val_accuracy across seeds (in percent)."""
    accs = []
    for r in results:
        a = r.get("val_accuracy") or r.get("accuracy") or r.get("final_accuracy")
        if a is None:
            continue
        # already in [0,1]? Convert to percent
        accs.append(a * 100 if a < 1.5 else a)
    return np.array(accs, dtype=float)


def bootstrap_ci(x: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI on the mean."""
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = x[idx].mean()
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d between two paired-seed samples (using pooled std)."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def paired_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Paired t-test: returns (t_stat, p_value)."""
    if len(a) != len(b) or len(a) < 2:
        return (float("nan"), float("nan"))
    res = sp_stats.ttest_rel(a, b)
    return (float(res.statistic), float(res.pvalue))


def aggregate(methods: list[str], datasets: list[str], reference: str = "kaleidonet") -> dict:
    """Aggregate results into a stats dict: {dataset: {method: {...stats...}}}."""
    out: dict = {}
    for ds in datasets:
        out[ds] = {}
        ref_acc = get_accuracy(load_results(reference, ds))
        for m in methods:
            seeds_acc = get_accuracy(load_results(m, ds))
            entry: dict = {
                "n_seeds": int(len(seeds_acc)),
                "mean": float(seeds_acc.mean()) if len(seeds_acc) else float("nan"),
                "std": float(seeds_acc.std(ddof=1)) if len(seeds_acc) > 1 else float("nan"),
                "raw": seeds_acc.tolist(),
            }
            if len(seeds_acc) > 0:
                lo, hi = bootstrap_ci(seeds_acc, n_boot=10_000, seed=42)
                entry["ci95_lo"] = lo
                entry["ci95_hi"] = hi
            if m != reference and len(seeds_acc) == len(ref_acc) and len(ref_acc) > 1:
                t, p = paired_t(seeds_acc, ref_acc)
                entry["paired_t_vs_" + reference] = t
                entry["p_value_vs_" + reference] = p
                entry["cohens_d_vs_" + reference] = cohens_d(seeds_acc, ref_acc)
            out[ds][m] = entry
    return out


def latex_table(stats: dict, methods: list[str], datasets: list[str], caption: str, label: str, reference: str = "kaleidonet") -> str:
    """Format stats as a LaTeX table with mean ± std and p-value vs reference."""
    n_ds = len(datasets)
    col_spec = "l" + "c" * (n_ds * 2)  # mean and p-value per dataset
    head = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
    )
    # Header rows
    head += "\\textbf{Method}"
    for ds in datasets:
        head += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{DATASET_LABELS[ds]}}}}}"
    head += " \\\\\n"
    cmidrules = "".join([f"\\cmidrule(lr){{{2 + 2*i}-{3 + 2*i}}}" for i in range(n_ds)])
    head += cmidrules + "\n"
    head += " "
    for _ in datasets:
        head += " & Acc.\\ (\\%) & $p$ vs. ref."
    head += " \\\\\n\\midrule\n"

    body = ""
    for m in methods:
        row = METHOD_LABELS.get(m, m)
        for ds in datasets:
            entry = stats.get(ds, {}).get(m, {})
            mean = entry.get("mean", float("nan"))
            std = entry.get("std", float("nan"))
            p = entry.get("p_value_vs_" + reference)
            mean_cell = "---" if np.isnan(mean) else f"${mean:.2f}\\pm{std:.2f}$"
            if m == reference:
                p_cell = "---"  # reference vs itself
            elif p is None or np.isnan(p):
                p_cell = "---"
            elif p < 0.001:
                p_cell = "$<\\!0.001$"
            else:
                p_cell = f"${p:.3f}$"
            row += f" & {mean_cell} & {p_cell}"
        body += row + " \\\\\n"

    foot = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    return head + body + foot


def print_table_summary(stats: dict, methods: list[str], datasets: list[str], reference: str):
    print(f"\nReference method (paired t-test): {reference}")
    print(f"{'Method':<35} | " + " | ".join([f"{ds:<25}" for ds in datasets]))
    print("-" * (35 + 28 * len(datasets)))
    for m in methods:
        cells = []
        for ds in datasets:
            e = stats.get(ds, {}).get(m, {})
            mean = e.get("mean", float("nan"))
            std = e.get("std", float("nan"))
            p = e.get("p_value_vs_" + reference)
            n = e.get("n_seeds", 0)
            if np.isnan(mean):
                cell = f"{'(no data)':<25}"
            elif m == reference:
                cell = f"{mean:5.2f} ± {std:.2f} (n={n})    "
            else:
                if p is None or np.isnan(p):
                    p_str = "p=?"
                elif p < 0.001:
                    p_str = "p<0.001"
                else:
                    p_str = f"p={p:.3f}"
                cell = f"{mean:5.2f} ± {std:.2f} {p_str}"
            cells.append(f"{cell:<25}")
        print(f"{METHOD_LABELS.get(m, m):<35} | " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=["main", "ablation", "both"], default="both",
                    help="Which table to emit")
    ap.add_argument("--reference", default="kaleidonet",
                    help="Reference method for paired t-test (default: kaleidonet)")
    ap.add_argument("--json", help="Dump full stats as JSON to this path")
    ap.add_argument("--latex-out", default=str(ROOT / "paper" / "stats_tables.tex"),
                    help="Output LaTeX fragment file")
    args = ap.parse_args()

    targets = []
    if args.table in ("main", "both"):
        targets.append(("main", MAIN_METHODS,
                        "Main comparison with statistical significance (50{,}000 steps, 3 seeds, "
                        "mean $\\pm$ std). $p$ values are paired $t$-tests vs.\\ KaleidoNet across the "
                        "matched seeds. With 3 seeds the test is underpowered; we report $p<0.05$ as "
                        "indicative rather than confirmatory.",
                        "tab:main_stats"))
    if args.table in ("ablation", "both"):
        targets.append(("ablation", ABLATION_METHODS,
                        "Schedule add-on ablation at convergence (50{,}000 steps, 3 seeds, "
                        "mean $\\pm$ std). $p$ values are paired $t$-tests vs.\\ KaleidoNet (full).",
                        "tab:ablation_schedule"))

    all_stats = {}
    fragments = []
    for name, methods, caption, label in targets:
        s = aggregate(methods, DATASETS, reference=args.reference)
        all_stats[name] = s
        print(f"\n=== {name.upper()} ===")
        print_table_summary(s, methods, DATASETS, args.reference)
        frag = latex_table(s, methods, DATASETS, caption, label, args.reference)
        fragments.append(f"% --- Table {name} ---\n" + frag)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"\n[saved] {args.json}")

    out_path = Path(args.latex_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n\n".join(fragments) + "\n")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
