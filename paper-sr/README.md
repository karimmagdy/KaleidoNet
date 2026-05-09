# KaleidoNet -- Scientific Reports Submission

This directory contains the Scientific Reports (SR) version of the
KaleidoNet manuscript. The NeurIPS / TMLR LaTeX version lives at
`/Users/kmagdy-ma-eg/Workspace/Research/KaleidoNet/` and is not
modified.

## Files

| File | Purpose |
|------|---------|
| `main.tex` | Complete SR-format manuscript. `\documentclass[fleqn,10pt]{article}`. Nature section order: Introduction -> Results -> Discussion -> Methods. |
| `references.bib` | Cleaned bibliography. Orphan entries removed; MoE-Pruner venue corrected; 2024-2025 MoE pruning and ViT compression references added. |
| `SUBMISSION_NOTES.md` | Cover letter draft, subject areas, suggested reviewers, data/code availability, pre-submission TODOs, SPRO-item crosswalk, and framing rationale. |
| `README.md` | This file -- index + diff notes vs. NeurIPS/TMLR version. |

Figures are referenced from the original paper tree via relative path:
`\includegraphics{../../KaleidoNet/paper/figures/FILENAME.pdf}`. Do
not edit the original figures; the original manuscript must remain
untouched.

## Compiling

```bash
cd /Users/kmagdy-ma-eg/Workspace/Research/ScientificReports_Submissions/KaleidoNet_SR
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Required bibliography style: `naturemag.bst` (shipped with most TeX
distributions; Nature's submission system accepts this style).

## What this manuscript is

An **analysis paper** that:

1. Diagnoses why a widely used neural-network pruning recipe
   (Lagrangian FLOPs penalty) fails when applied at per-neuron
   granularity inside Mixture-of-Experts (MoE) experts. The failure
   is a gradient-scale mismatch: per-neuron penalty gradients are
   ~10^-4, two orders of magnitude below the task gradients ~10^-2
   on the same variables.
2. Empirically validates the failure mode on CIFAR-10, CIFAR-100,
   and Tiny-ImageNet, where a tuned Lagrangian baseline reaches the
   target FLOPs budget but collapses to near-random accuracy
   (36.4% / 10.4% / 4.7%).
3. Proposes a schedule-based fix (**KaleidoNet**): deterministic
   cubic sparsity schedule + irreversible gradient masking +
   dual-rate optimisation. Trained to convergence (50,000 steps,
   three seeds), KaleidoNet achieves a 1.80x FLOPs reduction,
   retains 98.3-99.2% of dense accuracy, and outperforms every
   schedule-based pruning baseline on every dataset.

The paper's remaining limitation is wall-clock: at the small model
scale used here, MoE routing overhead makes KaleidoNet ~18x slower
than the dense baseline, an artefact expected to shrink
substantially at larger model scale.

## Diff notes vs. NeurIPS / TMLR version

Both the NeurIPS / TMLR LaTeX at `KaleidoNet/paper/main.tex` and this
SR submission now report the same 50,000-step converged numbers
(CIFAR-10 82.0%, CIFAR-100 59.1%, Tiny-ImageNet 40.1%; dense 83.1% /
59.6% / 40.8%). All numerical claims in the SR `main.tex` are
numerically locked to the source paper's `experiments.tex` and
`appendix.tex` per-seed tables. Earlier drafts of this submission
used obsolete 5,000-step preliminary numbers from a legacy
`REPORT.md`; those have been replaced in a numerical-sync pass.

### Structural changes

- **Section order**: NeurIPS uses *Introduction -> Related Work ->
  Method -> Experiments -> Discussion*; SR uses the Nature order
  *Introduction -> Results -> Discussion -> Methods -> References ->
  Acknowledgements -> Author Contributions -> Competing Interests ->
  Data Availability -> Code Availability*.
- **Related work**: folded into the Introduction and Discussion
  narratives rather than kept as a standalone section, per Nature's
  convention.
- **NeurIPS checklist**: removed.
- **Appendix**: removed. Per-seed numbers are shipped as JSON with
  the companion code repository.

### Content changes (driven by SPRO/REVISION_PLAN and numerical-sync pass)

- **Abstract, intro, Results, and Discussion rewritten** to match the
  source paper's 50,000-step canonical data. Now states: KaleidoNet
  retains 98.3-99.2% of dense accuracy at 1.80x FLOPs reduction and
  outperforms all schedule-based pruning baselines; Lagrangian
  baseline collapses (36.4% / 10.4% / 4.7%).
- **Table 1 expanded to three datasets (CIFAR-10, CIFAR-100,
  Tiny-ImageNet)**; all entries drawn verbatim from
  `KaleidoNet/paper/sections/experiments.tex`.
- **Wall-clock latency promoted to abstract and intro** (REVISION_PLAN
  P1.5). Not hidden in a late section; part of the contribution's
  self-description.
- **Table 1 bolding fixed** (REVISION_PLAN W15). Bold is used only for
  the best *pruned* result, not the dense reference.
- **Pareto figure (fig1_pareto.pdf) included explicitly**
  (REVISION_PLAN P2.3) as Figure 2.
- **Lagrangian sensitivity analysis** added as a narrative subsection
  in the Discussion (REVISION_PLAN W5).
- **Bibliography updated** with 2024-2025 MoE pruning references
  (REVISION_PLAN P2.5): SlimMoE, STUN, MoNE, MaskLLM, MoE-Pruner,
  MoNE, Li et al.\ 2024 Merge-then-Compress, Liu et al.\ 2024
  Efficient Expert Pruning, He et al.\ 2025 Upcycling, MCNC.
- **Tiny-ImageNet cited properly** via `le2015tiny` (SPRO P1.1).
- **MoE-Pruner citation** fixed to arXiv preprint (SPRO P1.2; paper
  was withdrawn from ICLR 2025).
- **Orphan bib entries removed**: `coates2011analysis`, `he2018soft`,
  `hinton2012neural`, `smith2019super` (SPRO P1.3).

### Style changes for SR audience

- **Mixture-of-Experts, pruning, Lagrangian penalty explained
  accessibly** in the first two paragraphs of the Introduction for a
  broader interdisciplinary audience.
- **Gradient-scale intuition precedes the mathematics**. The math
  appears in full only in the Methods section (Eq.\ 3).
- **Less ML jargon**: paragraph breaks at conceptual boundaries
  rather than at technical ones; terms like "uniform mask collapse"
  are named and then described in plain language.

## Hard rule

Do not edit any file under
`/Users/kmagdy-ma-eg/Workspace/Research/KaleidoNet/`. The SR
submission references figures in `KaleidoNet/paper/figures/` via
relative paths only; the originals remain untouched.
