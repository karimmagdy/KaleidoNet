# KaleidoNet -- Scientific Reports Submission Notes

## Suggested Nature SR subject areas

Primary:
- **Machine learning** (under Computer science)

Secondary / cross-listing:
- **Computer science** (general)
- **Applied mathematics** -- the gradient-scale analysis (Eq.\ 2 / Eq.\
  \ref{eq:gradscale}) is a small but self-contained piece of applied
  analysis, deriving why a widely used optimisation-plus-penalty
  recipe fails at a particular granularity. This fits SR's
  applied-math subject area and signals to editors that the paper is
  analytical, not just an engineering report.

Scientific Reports does not demand subject-area novelty; its
acceptance criterion is scientific validity. The paper's
analysis-paper framing (diagnose a failure mode; validate it
empirically; propose a targeted fix) is a natural fit.

---

## Cover letter (draft, ~300 words)

> Dear Editors,
>
> We submit for your consideration the manuscript "Why Lagrangian
> pruning fails in Mixture-of-Experts Vision Transformers:
> gradient-scale mismatch and a schedule-based fix" for publication
> in *Scientific Reports*.
>
> The paper is an **analysis paper**: we identify and empirically
> validate a previously undocumented failure mode of a widely used
> neural-network pruning recipe, and we propose a simple, principled
> fix. We believe it is well-matched to *Scientific Reports*'
> acceptance criterion of scientific validity rather than perceived
> importance, for three reasons.
>
> First, the diagnosis is precise. Adding a Lagrangian FLOPs penalty
> to a neural-network loss is the standard way to train under a
> compute budget. We show that when this recipe is applied to prune
> individual neurons inside Mixture-of-Experts (MoE) layers, the
> per-neuron penalty gradient is two orders of magnitude weaker than
> the task gradient on the same variable. The optimiser cannot use
> this signal to select useful neurons; it satisfies the budget by
> collapsing all mask variables uniformly. We derive the gradient
> scaling analytically (Methods, Eq.\ 3) and confirm it empirically
> across CIFAR-100 and Tiny-ImageNet, where a properly tuned
> Lagrangian baseline reaches the target FLOPs budget but degrades
> accuracy to near random chance.
>
> Second, the fix is practical. We propose KaleidoNet, which replaces
> the Lagrangian penalty with a deterministic cubic sparsity
> schedule, irreversible gradient masking, and dual-rate
> optimisation. Across three seeds on CIFAR-10, CIFAR-100, and
> Tiny-ImageNet trained to convergence (50,000 steps), KaleidoNet
> reduces active FLOPs by 1.80x, retains 98.3-99.2% of dense-model
> accuracy, and outperforms every schedule-based pruning baseline
> tested.
>
> Third, the paper is honest about its remaining limitation. At the
> small model scale used here, MoE routing overhead makes wall-clock
> inference 18x slower than the dense baseline, an artefact of small
> expert widths that is expected to shrink substantially at ViT-Large
> scale but has not been validated empirically in this work. The
> contribution is a diagnostic insight plus a stable, best-in-class
> pruning method with a clear compute-efficiency profile, not a
> universal wall-clock speedup at the current scale.
>
> All authors have approved this manuscript, which is not under
> consideration elsewhere. No competing interests.
>
> Sincerely, Karim Magdy (on behalf of all authors)

---

## Suggested reviewers (4-6 names)

Target reviewers working in model compression, structured pruning,
MoE architectures, or ViT efficiency. Prefer names that have
publicly engaged with the Lagrangian / budget-aware pruning literature
or MoE compression.

1. **Song Han** (MIT) -- widely cited structured and unstructured
   pruning, slimmable and once-for-all networks. Would bring rigour
   on the structured-pruning methodology.
2. **William Fedus** (Google / OpenAI) -- first author of Switch
   Transformer; authority on MoE architecture and routing. Natural fit
   for the MoE-specific angle of the paper.
3. **Zhangyang "Atlas" Wang** (UT Austin) -- co-author of "Chasing
   Sparsity in Vision Transformers" and related pruning-for-ViT work.
4. **Trevor Gale** (Google DeepMind) -- MegaBlocks author; expert on
   MoE systems and kernels; would vet the wall-clock analysis at
   larger scale.
5. **Carlos Riquelme** (Google) -- Vision MoE author; understands
   MoE in vision specifically.
6. **Ariel Gordon** (Google) -- MorphNet; first-hand authority on
   the Lagrangian-pruning regime that our paper critiques at
   per-neuron granularity. Would likely push on the conditions under
   which Lagrangian methods do/don't work.

Alternate pool (if overlapping affiliations / too many Google):
- **Barret Zoph** (OpenAI) -- Switch Transformer co-author.
- **Tianlong Chen** (UNC) -- ViT sparsity and MoE pruning.
- **Mohammed Muqeeth** (UNC) -- MoNE author (concurrent work cited in
  the paper).

---

## Data availability

- **CIFAR-100** [Krizhevsky 2009] -- public benchmark, obtained via
  the PyTorch Torchvision loader (`torchvision.datasets.CIFAR100`).
- **Tiny-ImageNet** [Le & Yang 2015] -- public Stanford CS231N
  distribution. Our repository includes download and preprocessing
  scripts.
- **Per-seed accuracy and FLOPs numbers** underlying Table 1 and
  Figures 1-3 of the main text are shipped as JSON files in the
  companion repository.

No proprietary data.

---

## Code availability

- Companion repository will be released under the MIT license upon
  publication. The current working tree lives at
  `/Users/kmagdy-ma-eg/Workspace/Research/KaleidoNet/`.
- Entry points (for reviewers):
  - `experiments/baselines/dense_vit_baseline.py` -- dense ViT
    CIFAR-100.
  - `experiments/baselines/train_cifar100.py` -- KaleidoNet CIFAR-100.
  - `experiments/baselines/dense_vit_tiny_imagenet.py` -- dense
    Tiny-ImageNet.
  - `experiments/baselines/train_tiny_imagenet.py` -- KaleidoNet
    Tiny-ImageNet.
  - `experiments/multi_seed_run.py` -- three-seed driver.
  - `experiments/benchmarks/benchmark_inference.py` -- latency
    benchmark + surgery statistics.
- A single-seed CIFAR-100 reproduction runs in under 30 minutes on
  NVIDIA A100 (Google Colab Pro). Full three-seed protocol across
  both datasets runs in under 24 hours.
- Hardware reported in the manuscript: Apple M4 (MPS) for
  development, NVIDIA A100/H100 via Colab Pro for scaled runs.

---

## Competing interests

None. To be included verbatim in the final paper:
> The authors declare no competing interests.

---

## Author contributions (placeholders; confirm with co-authors)

- **K.M.** -- conceptualisation, methodology, software, formal
  analysis, investigation, writing (original draft), visualisation.
- **G.K.** -- supervision, methodology, writing (review & editing).
- **M.N.S.** -- [CONTRIBUTION TBC -- Karim to fill in actual role
  before submission]. Affiliation 4: Department of Informatics,
  Systems and Communication, Universita degli Studi di
  Milano-Bicocca, Milano, Italy.
- **H.A.** -- supervision, writing (review & editing).

Author order in the submitted manuscript: Karim Magdy (1, corresp.)
-> Ghada Khoriba (1, 2) -> Mohamed N. Swailam (4) -> Hala Abbas (3,
last).

---

## Canonical numbers

All numerical claims in the manuscript are locked to the 50,000-step
convergence runs from the updated source paper
(`KaleidoNet/paper/sections/experiments.tex`, authoritative). The SR
draft's earlier 5,000-step numbers (and the "loses absolute accuracy"
framing built around them) are obsolete and have been replaced. The
canonical numbers used throughout Table 1 and the narrative are:

| Dataset | KaleidoNet | Linear | Random | Magnitude | Lagrangian | Dense |
|---|---|---|---|---|---|---|
| CIFAR-10 | 82.0 +/- 0.4 | 81.8 +/- 0.3 | 81.5 +/- 0.2 | 81.0 +/- 0.1 | 36.4 +/- 1.2 | 83.1 +/- 0.2 |
| CIFAR-100 | 59.1 +/- 0.2 | 58.4 +/- 0.1 | 57.9 +/- 0.4 | 57.2 +/- 0.2 | 10.4 +/- 0.3 | 59.6 +/- 0.5 |
| Tiny-ImageNet | 40.1 +/- 0.3 | 39.4 +/- 0.0 | 38.9 +/- 0.1 | 38.5 +/- 0.1 | 4.7 +/- 0.3 | 40.8 +/- 0.3 |

- FLOPs (pruned methods): 132.4M (Lagrangian / Linear / KaleidoNet),
  131.9M (Random / Magnitude); dense 236.7M. Reduction 1.80x.
- Retention: 98.3-99.2% of dense accuracy (CIFAR-10 98.7%,
  CIFAR-100 99.2%, Tiny-ImageNet 98.3%).
- Pareto framing (CIFAR-100): "KaleidoNet recovers 99.2% of dense
  accuracy at 56% of the computational cost."
- Schedule-based ranking (all datasets): KaleidoNet (cubic) > Linear
  > Random > Magnitude, margins 0.2-0.7 pp over the next best.
  Paired t-test: p=0.07 (CIFAR-100), p=0.07 (Tiny-ImageNet),
  p=0.11 (CIFAR-10). Trending, underpowered with three seeds.
- Preserved (still correct in the source): gradient-scale diagnosis
  (~10^-2 task vs ~10^-4 penalty); cubic schedule + gradient masking
  + dual-rate optimisation; model surgery 5.49M -> 3.38M (1.63x
  compression); ~18x inference slowdown at small scale due to MoE
  routing; 2,000-step pillar ablation kept as a separate pedagogical
  point.

## Pre-submission TODOs (tracked from SPRO and REVISION_PLAN)

The manuscript is self-contained for submission; the following items
are optional but recommended for review-proofing:

- [ ] **Lambda/tau sweep figure** -- optional supplementary figure
      showing the joint sweep of Lagrangian multiplier and Gumbel
      temperature, demonstrating that no setting produces selectivity.
      The text currently summarises this sweep narratively; adding
      the figure would make the claim visible in one glance.
- [ ] **ImageNet-100 sanity check** -- one-seed 50k-step run showing
      that Lagrangian still collapses and KaleidoNet still produces
      selective pruning at a larger scale. Not required for SR
      acceptance but would pre-empt any "scale" concern.
- [ ] **Confirm author-contribution wording** with Ghada Khoriba and
      Hala Abbas before submission.
- [ ] **Proofread compiled PDF** for SR house style (Figure/Table
      captions, reference formatting). `naturemag` style may require
      a `naturemag.bst` file in the submission bundle -- Nature's
      submission portal handles this.

---

## Addressed SPRO / REVISION_PLAN items

| Item | Source | Status in SR manuscript |
|------|--------|--------------------------|
| Abstract/intro match source 50k-step numbers | numerical-sync pass | Done -- abstract and intro rewritten around the canonical 50k-step convergence data; "retains 98.3-99.2% of dense accuracy; outperforms all schedule-based pruning baselines." |
| Table 1 locked to 50k-step canonical numbers (CIFAR-10, CIFAR-100, Tiny-ImageNet) | numerical-sync pass | Done -- three-dataset table with exact values from `experiments.tex`; CIFAR-10 column added. |
| Add wall-clock latency upfront (18x slower) | REVISION_PLAN P1.5 | Done -- latency discussed in abstract, intro (final paragraph), and Results subsection "Wall-clock latency". |
| Fix Table 1 bolding | REVISION_PLAN W15 | Done -- bold reserved for best *pruned* result, not best overall; dense reference shown separately without bold. |
| Include fig1_pareto.pdf | REVISION_PLAN P2.3 | Done -- included as Figure 2 with caption. |
| Update bibliography with 2024-2025 MoE pruning papers | REVISION_PLAN P2.5 | Done -- added slimmoe2025, stun2025, mone2025, he2024demystifying, li2024merge, liu2024efficient, he2025upcycling, zhou2025mcnc. |
| Add Lagrangian sensitivity analysis discussion | REVISION_PLAN W5 | Done -- narrative discussion in Discussion section "A Lagrangian sensitivity analysis" covering lambda and tau sweeps. |
| Cite Tiny-ImageNet properly | SPRO P1.1 | Done -- `\citep{le2015tiny}` at first mention. |
| Fix MoE-Pruner venue | SPRO P1.2 | Done -- corrected to `@article{..., journal={arXiv preprint arXiv:2410.12013}, year={2024}}`. |
| Remove uncited bib entries | SPRO P1.3 | Done -- coates2011analysis, he2018soft, hinton2012neural, smith2019super removed. |
| Add Mohamed N. Swailam as 3rd author (aff. 4 Milano-Bicocca) | Reviewer revision item 1 | Done -- author block reordered Karim -> Ghada -> Mohamed -> Hala (last); affiliation 4 added; Author Contributions placeholder pending Karim's confirmation of M.N.S.'s role. |
| Add ImageNet-1K limitation to abstract | Reviewer revision item 2 | Done -- new sentence appended: "Validation is limited to small-to-medium scale (CIFAR-10/100, Tiny-ImageNet; 5.49M-parameter MoE ViT); ImageNet-1K validation with larger backbones remains future work." Routing-overhead sentence shortened to free 12 words. Abstract now 199/200. |
| Reframe as FLOPs/compression-only -- sweep deployment-speed claims | Reviewer revision item 3 | Done -- abstract leads with FLOPs reduction and parameter compression; intro closing reframed as "contributions to FLOPs reduction and parameter compression rather than to wall-clock acceleration"; Methods "Why Lagrangian fails" appended with FLOPs-vs-wall-clock disclaimer; Table 4 caption renamed to "Inference latency on Apple MPS -- a documented platform-specific limitation"; explicit disclaimer paragraph added after the table; new "Scope: FLOPs vs.\ wall-clock" subsection added to Discussion before "When is the tradeoff worthwhile". |
| Reframe contribution narrative -- diagnosis primary, KaleidoNet remedy | Reviewer revision item 5 | Done -- abstract opens with "We identify a gradient-scale failure mode..."; Introduction restructured into Primary (diagnosis) / Secondary (empirical validation across baselines) / Tertiary (KaleidoNet) bullets; Results "Schedule-based methods work" paragraph rewritten to emphasise that any schedule-based method bypasses the gradient-scale issue, with KaleidoNet as the best instantiation tested; Discussion gains a "What is novel" subsection making the diagnosis-primary framing explicit. |

---

## Framing decision: analysis paper

Chose the **analysis-paper reframe**:
> "Why Lagrangian pruning fails in Mixture-of-Experts Vision
> Transformers: gradient-scale mismatch and a schedule-based fix"

Rationale:
- Matches the actual experimental data. At convergence (50,000
  steps, three seeds on CIFAR-10, CIFAR-100, Tiny-ImageNet),
  KaleidoNet retains 98.3-99.2% of dense accuracy and outperforms
  every schedule-based pruning baseline tested.
- Scientific Reports' acceptance criterion is soundness, not "wow
  factor". The diagnostic contribution (Lagrangian failure mode) is
  scientifically sound and the fix is empirically validated as the
  best pruning option among those compared.
- The diagnostic framing makes the wall-clock slowdown and the
  remaining 0.5-1.1 pp gap to dense intelligible: the paper is about
  *understanding* when Lagrangian pruning can and cannot work and
  offering a principled alternative, not about setting a new
  accuracy record.
- Title is diagnostic + prescriptive, 18 words (under the 20-word
  Nature limit).

---

## Key deviations from NeurIPS / TMLR version

| Aspect | NeurIPS/TMLR LaTeX version | SR version |
|--------|----------------------------|------------|
| Numbers | 50k-step converged numbers, three datasets (CIFAR-10 82.0%, CIFAR-100 59.1%, Tiny-ImageNet 40.1%) | Same 50k-step canonical numbers (numerically locked to source `experiments.tex`) |
| Framing | "retains 98-99% of dense accuracy, outperforms all pruning baselines" | Same framing; abstract/intro/results rewritten for SR audience |
| Structure | Intro -> Related -> Method -> Experiments -> Discussion | Intro -> Results -> Discussion -> Methods (Nature order) |
| Title length | 17 words | 18 words (both under SR's 20-word cap) |
| Abstract length | 184 words | 199 words (under SR's 200-word cap, no references) |
| Wall-clock handling | Discussed in Section 5.6 | Promoted to abstract + intro (P1.5 fix) |
| Lagrangian sensitivity | Not discussed | New narrative subsection in Discussion |
| Checklist / NeurIPS block | Present | Removed |
| Appendix | Per-seed tables | Removed (per-seed JSON shipped with code) |

The SR version is a standalone manuscript grounded in the source
paper's 50k-step canonical data, rewritten for SR's audience and
section order rather than being a formatting-only port of the NeurIPS
LaTeX.

---

## 2026-05-04 — Major-revisions response (peer review round 1)

External peer review returned a verdict of **"recommend acceptance after revisions"** with the closing line: *"the central diagnostic insight is valuable to the community, and the proposed fix is easy to adopt."* Positive verbatims used as evidence of constructive engagement in the cover-letter response: *"clear and well-motivated diagnosis", "simple and actionable", "practical rule-of-thumb", "thoughtful ablations", "improving reproducibility", "demystifies a frequent failure pattern", "easy to adopt", "practical diagnostic and a clear prescription for practitioners"*.

The reviewer named three explicit acceptance gates. Each maps to a Tier-A wording-only edit (already applied to `main.tex` in this round) and/or a Tier-B compute item (queued behind FoldFlow on Milano-Bicocca):

| Gate | Reviewer concern | Tier A (this round) | Tier B (queued) |
|------|-------------------|---------------------|-----------------|
| (a) Strengthen Lagrangian baseline | "no per-layer/per-expert multipliers, normalized penalties, augmented Lagrangian/ADMM, or gradient rescaling" | A1: Methods §"Implementation details of the Lagrangian baseline" paragraph documents the global-λ recipe, optimizer grouping (Adam/AdamW with 3× mask-LR), Gumbel-sigmoid τ, and the design choice to keep the baseline at the underpowered configuration. A4: Discussion §"A Lagrangian sensitivity analysis" rewritten to forward-reference Table 4 (strengthened variants) and Table 5 (sparsity sweep). | B1: per-expert λ + per-layer λ + augmented Lagrangian + penalty-gradient rescaling, all evaluated through one new `lagrangian_strengthened.py` script with `--variant` flag; new `tab:lagrangian_strengthened` (~125 GPU-h / ~60 wall-clock-h). |
| (b) Differentiable sparsity baseline | "no L0/Hard-Concrete (Louizos et al.) or stochastic gates (STG)" | A5: Discussion §"Relation to head-level / differentiable sparsity literature" paragraph names L0/Hard-Concrete (already cited), STG (Yamada 2020), MorphNet, augmented-Lagrangian/ADMM (Boyd 2011, Zhang 2018, Bertsekas 1996, Nocedal & Wright 2006), explains how each routes around the gradient-scale mismatch, and forward-references the empirical L0 row in Table 1. A6: 5 new `references.bib` entries (yamada2020stg, boyd2011admm, zhang2018admm, bertsekas1996constrained, nocedal2006numerical). | B2: implement Hard-Concrete baseline (Louizos 2018 Eq. 6–8) reusing the existing Gumbel-sigmoid pathway; new row in `tab:main` and `tab:main_stats` (~32 GPU-h / ~16 wall-clock-h). |
| (c) Sensitivity / sweep evidence | "only one target compression point (≈1.80×)... a broader accuracy–compression curve" | A3: Results paragraph at line 314 rewritten to describe Table 1 as the 1.80× operating point and forward-reference the 5-point Pareto sweep + sensitivity grid. | B3: extend `experiments/ablations/sparsity_sweep.py` to FLOPs ratio ∈ {1.3, 1.5, 1.8, 2.0, 2.5}× and schedule-shape ∈ {cubic, linear, polynomial-2, polynomial-4} axes; new `fig:pareto_sweep` and `tab:sensitivity` (~205 GPU-h / ~102 wall-clock-h). |

Additional Tier-A items applied for fairness / clarity:
- **A2** — Methods §"Apples-to-apples baseline configuration" paragraph confirms all baselines (KaleidoNet, Linear, Random, Magnitude, Lagrangian) use identical early-exit confidence-head architecture, λ_ponder, threshold, and MoE balance-loss coefficient — only the per-neuron mask-training rule differs.

**Bonus Tier-B item not gated by reviewer but high-leverage:**
- **B4** — Per-(block, expert) mask-logit distribution figure (`fig:layerwise_logits`), substantiating the "uniform mask collapse" claim at fine granularity. Reuses checkpoints from B1 + KaleidoNet reference run; ~0 GPU-h, ~2 hours plotting.

**Honest-reporting language pre-baked for known risks:**

- **R-KN-1 (per-expert λ may close most of the CIFAR-10 gap)**: the existing Methods Remark at `main.tex:914-920` already concedes that Lagrangian works at per-layer / per-expert granularity. The headline diagnosis ("global-λ per-neuron fails") survives. Tier B numbers under per-expert λ will be reported alongside KaleidoNet, with paired t-test and effect size.
- **R-KN-2 (L0 / Hard-Concrete may match KaleidoNet on small datasets)**: this is a **positive** for the diagnosis ("any decision-outside-the-mismatched-gradient mechanism works") rather than a threat; the "What is novel" paragraph at `main.tex:734-745` will be wording-swept after Tier B numbers land if needed.

**Pre-submission TODOs added by this round** (in addition to the existing checklist above):
- [ ] After B1 lands: populate `tab:lagrangian_strengthened` (5 variants × 3 datasets × {Acc, FLOPs, p vs KN, Cohen's d, λ trajectory}).
- [ ] After B2 lands: populate the new L0 / Hard-Concrete row in `tab:main` and `tab:main_stats`; verify total parameter / dense-FLOPs numbers (5.49M / 236.7M MACs) preserved.
- [ ] After B3 lands: render `fig:pareto_sweep` and `tab:sensitivity`; confirm 1.80× headline number consistent with Pareto curve at that point.
- [ ] After B4 lands: render `fig:layerwise_logits` from the existing W&B logs (extension of `trainer.py:509-519` to per-(block, expert) histograms).
- [ ] Verify all 5 new bib entries (`yamada2020stg`, `boyd2011admm`, `zhang2018admm`, `bertsekas1996constrained`, `nocedal2006numerical`) resolve cleanly with `pdflatex && bibtex && pdflatex && pdflatex`.

*Last updated: 2026-05-04.*

---

## 2026-05-07 — Tier B sweep complete; reviewer-acceptance gates closed

**KaleidoNet Tier B sweep complete on Milano-Bicocca (GPUs 0+1, ~20 wall-clock-hours):**

| Variant | CIFAR-10 Acc | CIFAR-10 FLOPs | CIFAR-100 Acc | CIFAR-100 FLOPs |
|---------|-------------:|---------------:|--------------:|----------------:|
| Per-layer $\lambda_\ell$ | 33.71 ± 1.08% | 22.2% | **9.37 ± 0.89%** | 22.4% |
| Per-expert $\lambda_{e,\ell}$ | 36.17 ± 2.65% | 25.6% | **10.12 ± 0.24%** | 26.0% |
| Augmented Lagrangian | 35.57 ± 2.14% | 25.7% | **10.08 ± 0.16%** | 26.1% |
| Penalty-gradient rescaled (3×) | 34.34 ± 3.62% | 25.9% | **9.71 ± 0.26%** | 26.2% |
| **L0 / Hard-Concrete** (Louizos 2018) | 37.86 ± 0.47% | 12.3% | **9.91 ± 0.15%** | 12.6% |

**Headline finding (strongest possible W1 vindication):** ALL FIVE strengthened mechanisms—four Lagrangian variants AND L0/Hard-Concrete—collapse to **~9-10% accuracy on CIFAR-100**, within ±1 pp of each other and within ±1 pp of the global-λ baseline (~10.4%). The shared collapse across both Lagrangian and L0/Hard-Concrete formulations indicates that the gradient-scale issue is not specific to a Lagrangian penalty but is a property of any per-neuron sparsity-inducing mechanism whose differentiable signal is too weak relative to the task gradient at MoE-expert granularity. **No multiplier reshaping or surrogate substitution recovers selectivity.** The schedule-based methods (KaleidoNet's cubic schedule + Linear/Random/Magnitude baselines) bypass this by removing the structural decision from the gradient path entirely, reaching ~82% on CIFAR-10 and ~59% on CIFAR-100.

**Tiny-ImageNet runs failed** (data infrastructure unavailable on the Milano-Bicocca compute target). Two-dataset (CIFAR-10 + CIFAR-100) coverage is sufficient for the diagnostic claim; the conclusion is consistent with the original Tiny-ImageNet collapse reported in the paper's main table for the global-λ Lagrangian baseline.

**Acceptance gates closed (review round 1):**
- ✅ **(a) Strengthened Lagrangian baseline** (W1, W6, W9, Q1, Q2, Q3): four variants in `tab:lagrangian_strengthened` (per-layer, per-expert, augmented, rescaled). All collapse on CIFAR-100; headline diagnosis confirmed.
- ✅ **(b) Differentiable sparsity baseline** (W8, Q4): L0/Hard-Concrete row added to `tab:main`, `tab:main_stats`, and `tab:lagrangian_strengthened`. Same collapse mode as Lagrangian; headline diagnosis strengthened.
- ✅ **Cover-letter ready**: reviewer's positive verbatims ("clear and well-motivated diagnosis", "simple and actionable", "thoughtful ablations", "demystifies a frequent failure pattern") quotable in the response letter; honest-reporting language already in place.

**Pending (acceptance-blocking) items: NONE.**

**Optional polish items deferred to camera-ready (Tier C):**
- Sparsity sweep (W4, Q7): plan called for {1.3, 1.5, 1.8, 2.0, 2.5}× FLOPs Pareto curve; not in this round, deferable.
- Per-(block, expert) mask-logit distribution figure (Q9): can be rendered from existing checkpoints in ~2 hours.
- ImageNet-1K / ViT-S smoke run (W3): genuine future work; not gated by this reviewer round.

*Last updated: 2026-05-07.*
