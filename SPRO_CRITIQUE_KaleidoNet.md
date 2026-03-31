# SPRO Critique: KaleidoNet

**Paper**: KaleidoNet: Stable Hard Pruning in Elastic MoE Vision Transformers via Cubic Sparsity Scheduling
**Target Venue**: NeurIPS 2026
**Review Date**: 2026-03-30
**Reviewer**: SPRO (Lead Scientific Paper Review Officer)

---

## 1. Overall Submission Readiness Score

**Score: 7 / 10 (Major Revision Required)**

The paper has a well-articulated motivation and a clean methodological narrative, but suffers from a critical experimental deficiency: the proposed method (KaleidoNet with cubic schedule) does not outperform any of its three baselines on any of the four datasets. This single fact undermines the entire contribution claim and would result in immediate desk rejection at NeurIPS.

---

## 2. Executive Punchline

KaleidoNet presents a well-motivated analysis of why Lagrangian FLOPs penalties fail at per-neuron granularity in MoE architectures, and proposes a clean cubic-schedule alternative -- but the experimental results fatally contradict the contribution claims because KaleidoNet is outperformed by random pruning and magnitude pruning on every single benchmark. The paper is not submittable in its current form; the authors must either (a) demonstrate conditions under which the cubic schedule genuinely outperforms baselines (e.g., longer training, larger scale) or (b) reframe the contribution entirely as a negative result / analysis paper. The writing quality and mathematical exposition are above average, which means this work can become publishable with substantial experimental revision.

---

## 3. Editor's First Impression

**Positive signals**: The gradient-scale analysis (Section 3.2, Eq. 2-4) is the strongest part of the paper -- it is a genuine, clearly explained insight. The paper is well-structured, the algorithm is precisely specified, reproducibility details are thorough, and the ablation design is methodical. The 4-figure set is appropriate.

**Red flags**: A careful editor will immediately check Table 1 and notice that on CIFAR-100, Random pruning (31.65%) beats KaleidoNet (31.21%); on Tiny-ImageNet, Magnitude pruning (17.36%) beats KaleidoNet (16.85%); on CIFAR-10, Random (62.87%) beats KaleidoNet (62.12%); on STL-10, Random (62.16%) beats KaleidoNet (61.63%). The bold entries in Table 1 highlight baselines, not KaleidoNet. This is a paper that argues its method is better but whose own data says otherwise. Any competent reviewer will catch this in under 60 seconds.

---

## 4. Major Weaknesses

### 4.1 Title and Abstract
- **W1 (Misleading claims)**: The abstract states KaleidoNet achieves "85-93% accuracy retention" and "1.8x FLOPs reduction," implying these are favorable outcomes. However, random pruning achieves equal or better retention at the same FLOPs. The abstract omits the critical fact that KaleidoNet does not outperform baselines.
- **W2 (Overclaimed contribution)**: "cubic schedule being the critical enabler of stable convergence" -- Table 1 shows the linear schedule performs comparably (within 0.2pp on most datasets), undermining this claim.

### 4.2 Introduction
- **W3 (Contribution bullet 3)**: States "we demonstrate that KaleidoNet achieves a consistent 1.8x FLOPs reduction with 85-87% accuracy retention." This is technically true but deceptively incomplete -- every baseline achieves similar or better retention at the same FLOPs target.
- **W4 (Figure 1 reference)**: Figure 1 (fig_motivation.pdf) is referenced as showing uniform collapse under Lagrangian penalty vs. selective pruning under KaleidoNet. This is the strongest visual argument, but without seeing the actual figure content, it is unclear whether this qualitative comparison is sufficiently rigorous.

### 4.3 Methods
- **W5 (Gradient analysis scope)**: The gradient-scale analysis (Section 3.2) is compelling but limited to a single set of hyperparameters. No sensitivity analysis is provided for lambda, tau, or batch size. The claim that Lagrangian control "fails" is based on one configuration.
- **W6 (No Lagrangian baseline)**: The paper identifies Lagrangian failure as the core motivation but does not include an actual Lagrangian baseline in the experimental comparison. Magnitude, random, and linear schedule are compared, but none of these use the Lagrangian penalty the paper critiques.

### 4.4 Results
- **W7 (FATAL -- KaleidoNet loses to all baselines)**: Table 1 shows KaleidoNet (cubic) is the worst or second-worst method on every dataset. On CIFAR-100: Random 31.65% vs. KaleidoNet 31.21%. On Tiny-ImageNet: Magnitude 17.36% vs. KaleidoNet 16.85%. On CIFAR-10: Random 62.87% vs. KaleidoNet 62.12%. On STL-10: Random 62.16% vs. KaleidoNet 61.63%. The bold markers in Table 1 correctly highlight baselines as winners, not KaleidoNet.
- **W8 (Absolute accuracy is very low)**: Dense ViT achieves only 36% on CIFAR-100 and 20% on Tiny-ImageNet at 5000 steps. Published ViT-Tiny results reach ~75% on CIFAR-100 with proper training. These numbers suggest extreme under-training, making it impossible to draw conclusions about pruning effectiveness.
- **W9 (STL-10 claim is misleading)**: The discussion claims "+3.9pp over dense baseline" on STL-10, but ALL pruned methods exceed the dense baseline on STL-10 (magnitude: +3.69, random: +4.46, linear: +4.25, cubic: +3.93). This is not a KaleidoNet-specific benefit; it reflects the MoE architecture helping at higher resolutions.
- **W10 (No wall-clock improvement)**: REPORT.md reveals KaleidoNet is 18x SLOWER than Dense ViT on MPS. This is not mentioned in the paper. For a pruning paper, this is a critical omission.

### 4.5 Ablation Study
- **W11 (Short ablation horizon)**: The ablation uses 2000 steps where elastic pruning has not yet completed. The paper acknowledges this but still draws conclusions about component contributions from an incomplete pruning trajectory.
- **W12 (Single seed)**: The ablation uses a single seed, which is insufficient to draw reliable conclusions about 0.4-0.7pp differences between configurations.

### 4.6 Discussion / Limitations
- **W13 (Training budget acknowledged but not addressed)**: The paper correctly identifies 5000-step training as a limitation but does not run even a single longer experiment (e.g., 20K steps on CIFAR-100) to validate the claim that "the accuracy gap would likely narrow."
- **W14 (No ImageNet)**: For a NeurIPS submission in 2026, evaluation only on CIFAR-10/100, Tiny-ImageNet, and STL-10 is below the expected standard.

### 4.7 Figures and Tables
- **W15 (Table 1 bold marking)**: Bold marks highlight baselines beating KaleidoNet, which a reader may initially misinterpret as KaleidoNet winning. This creates confusion rather than clarity.
- **W16 (Missing Pareto figure in main text)**: fig1_pareto.pdf exists in the figures directory but is not referenced in the paper. A Pareto efficiency plot would be the natural way to present accuracy-FLOPs tradeoffs.

### 4.8 References
- **W17 (Missing recent work)**: No citations to 2024-2026 MoE pruning work (e.g., "Not All Experts are Equal" by Lu et al., 2024; expert pruning work at ICLR 2025). The most recent MoE pruning citation is Kim et al. 2023.

### 4.9 Ethics / Broader Impact
- Minor. The broader impact statement is thin but adequate for a methods paper.

---

## 5. Fatal Flaws

**Fatal Flaw 1: The proposed method does not outperform any baseline on any dataset (Table 1).** This is not a matter of interpretation. The paper's own data shows random pruning and magnitude pruning consistently beat the cubic schedule. A method paper that cannot demonstrate superiority over random baseline selection will be rejected at any top venue.

**Fatal Flaw 2: Extreme under-training invalidates conclusions.** At 36% CIFAR-100 accuracy (vs. ~75% with proper training), the network has not learned meaningful feature representations. Pruning decisions made during this early phase may be entirely different from those made during converged training. No evidence is provided that the cubic schedule remains beneficial when training reaches competitive accuracy levels.

**Fatal Flaw 3: The paper does not include the Lagrangian baseline it critiques.** The entire motivation is that Lagrangian FLOPs penalties fail at per-neuron granularity, but the experimental section compares against magnitude, random, and linear schedule -- none of which use Lagrangian control. Without this baseline, the gradient-scale analysis remains theoretical.

---

## 6. Actionable Revision Plan

### Priority 1: Must Fix (blocks submission)

| # | Action | Effort | Section |
|---|--------|--------|---------|
| P1.1 | **Train to convergence** (minimum 50K steps / 300 epochs on CIFAR-100) and re-run all comparisons. If KaleidoNet still loses, reframe the paper. | 1-2 weeks | Sec 4 |
| P1.2 | **Add Lagrangian baseline**: implement the Lagrangian FLOPs penalty (Eq. 1) as a pruning method and include it in Table 1. This is the method the paper argues against. | 3-5 days | Sec 4.1 |
| P1.3 | **Add ImageNet-1K experiments** or at minimum ImageNet-100. Small-dataset-only evaluation is below NeurIPS standard for 2026. | 1-2 weeks | Sec 4 |
| P1.4 | **Rewrite claims to match data** -- if after P1.1 KaleidoNet still does not outperform, reframe as an analysis/negative-result contribution or find the regime where it wins. | 2-3 days | Abstract, Intro, Discussion |
| P1.5 | **Report wall-clock latency** in the main paper. A pruning method that is 18x slower must be transparently disclosed. | 1 day | Sec 4 |

### Priority 2: Strongly Recommended

| # | Action | Effort | Section |
|---|--------|--------|---------|
| P2.1 | Run ablation with 3 seeds and at full training length (5000+ steps). Single-seed 2000-step ablation is not reliable. | 3-5 days | Sec 4.3 |
| P2.2 | Add Lagrangian sensitivity analysis (sweep lambda, tau) to support the gradient-scale argument more rigorously. | 2-3 days | Sec 3.2 |
| P2.3 | Include fig1_pareto.pdf in the main text as an accuracy-FLOPs Pareto plot. | 1 hour | Sec 4 |
| P2.4 | Compare against L0 regularization (Louizos et al., 2018) as a baseline -- it is cited in related work but not compared against. | 3-5 days | Sec 4.1 |
| P2.5 | Update bibliography with 2024-2025 MoE pruning papers. | 1 day | References |

### Priority 3: Nice to Improve

| # | Action | Effort | Section |
|---|--------|--------|---------|
| P3.1 | Extend cubic schedule to prune attention heads (acknowledged as future work). | 1 week | Sec 3.3 |
| P3.2 | Test on a second hardware platform (CUDA GPU) to validate that wall-clock gains exist somewhere. | 2-3 days | Sec 4 |
| P3.3 | Add statistical significance tests (e.g., paired t-test) between KaleidoNet and baselines. | 1 day | Sec 4.2 |
| P3.4 | Discuss relationship to recent token pruning methods for ViTs. | 1 day | Sec 2 |

---

## 7. Journal / Venue Recommendation Matrix

Given the paper's current state requires major revision, I recommend against the NeurIPS 2026 May deadline. After revisions, consider the following:

| Rank | Venue | Fit Score /10 | Acceptance Likelihood | Speed to First Decision | Quartile / Indexing | APC | Why It Fits | Label |
|------|-------|---------------|----------------------|------------------------|--------------------|----|-------------|-------|
| 1 | **TMLR** (Transactions on Machine Learning Research) | 8/10 | Medium (after fixes) | ~9 weeks | Q1-equivalent, DBLP-indexed | Free (open access) | Rolling submissions, no deadline pressure. Ideal for methodological contributions with thorough analysis. J2C track allows later conference presentation at NeurIPS/ICML/ICLR. | **Top Choice** |
| 2 | **ECCV 2026** (Sep 8-13, Malmo) | 7/10 | Medium-Low | ~4 months (deadline passed: Feb 26, 2026) | Top-tier CV conference | Free | Strong vision venue; MoE+ViT pruning fits well. Deadline already passed for 2026 -- target ECCV 2028 or CVPR 2027. | **Stretch** |
| 3 | **NeurIPS 2026** (Dec 6-12) | 7/10 | Low-Medium | ~5 months (deadline: May 4-6, 2026) | Top-tier ML conference | Free | Original target. Only viable if P1.1-P1.5 are completed before May 4. Extremely tight timeline. | **Stretch** |
| 4 | **ICLR 2027** | 8/10 | Medium (after fixes) | ~5 months | Top-tier ML conference | Free | Best fit for methodological ML contributions. Gives 6+ months for thorough revision including ImageNet experiments. | **Safest Top-Tier** |
| 5 | **IEEE TPAMI** | 6/10 | Medium | 3-6 months first review | Q1, IF ~24 | ~$2,500 OA optional | For an extended version with comprehensive experiments. Requires significantly more empirical depth. | **Journal Option** |
| 6 | **Neural Networks** (Elsevier) | 7/10 | Medium-High | 2-4 months | Q1, IF ~7.8 | ~$3,390 OA | Good fit for architecture + pruning papers. More forgiving on scale of experiments if analysis is strong. | **Fastest** |

**Note on NeurIPS 2026 deadline (May 4-6)**: This is 35 days away. Completing P1.1 (train to convergence), P1.2 (Lagrangian baseline), and P1.3 (ImageNet) in 35 days is possible but risky. I recommend TMLR or ICLR 2027 for a higher-quality submission.

---

## 8. Cover Letter Advice

If submitting after revisions, the cover letter should:

- **Lead with the gradient-scale analysis** as the primary intellectual contribution -- this is the most novel and defensible part of the paper. Frame it as: "We identify a previously uncharacterized failure mode of Lagrangian FLOPs penalties at per-neuron granularity in MoE architectures."
- **Explicitly state that the cubic schedule is one solution, not necessarily the optimal one** -- this pre-empts the reviewer concern that the method does not outperform all baselines, if that remains the case.
- **Highlight the Lagrangian baseline comparison** (once added) as direct empirical validation of the theoretical analysis.
- **Acknowledge the scale limitation upfront** and frame small-scale experiments as a controlled study, with ImageNet results (once obtained) demonstrating generality.
- **Emphasize reproducibility**: all code, configs, and seeds are public; results are repeatable on consumer hardware (Apple MPS).
- **Suggest 2-3 qualified reviewers** who work at the intersection of MoE architectures and structured pruning (e.g., authors of Switch Transformer pruning work, Once-for-All, or ViT pruning papers).
- **Do NOT claim state-of-the-art** unless post-revision results genuinely demonstrate it. Overclaiming is the fastest path to desk rejection.

---

## 9. Final Recommendation

**Revise substantially (4-8 weeks), then submit to TMLR or ICLR 2027.**

The paper contains a genuine insight (gradient-scale mismatch in Lagrangian per-neuron pruning) buried under experimental results that contradict its claims. In its current form, it would receive a confident reject at NeurIPS (scores in the 3-4 range). However, the core analysis is sound, the writing is clear, and the codebase appears well-engineered. With (a) converged training runs, (b) a Lagrangian baseline, (c) ImageNet-scale validation, and (d) honest reframing of results, this can become a solid contribution.

Do NOT submit to NeurIPS 2026 (May 4 deadline) unless all Priority 1 items are completed and results are favorable. Submitting with the current data would waste a review cycle and potentially damage the authors' reputation with reviewers.

---

## 10. Summary Box

**One-line verdict**: A well-written paper with a genuine theoretical insight that is fatally undermined by experimental results showing the proposed method loses to every baseline including random pruning.

**Top 5 mandatory fixes**:
1. Train all methods to convergence (300+ epochs) and re-evaluate -- current 5000-step results are meaningless for pruning comparisons
2. Add the Lagrangian FLOPs penalty as an explicit baseline (it is the method being critiqued)
3. Add ImageNet-1K or ImageNet-100 experiments
4. Report wall-clock latency honestly (currently 18x slower, hidden in REPORT.md)
5. Rewrite abstract/intro claims to match actual experimental outcomes

**Top 4 venue options**:
1. TMLR (rolling, free, ~9 weeks to decision, J2C track to NeurIPS/ICML/ICLR)
2. ICLR 2027 (best top-tier fit, ~9 months runway)
3. NeurIPS 2026 (only if all fixes done by May 4 -- high risk)
4. Neural Networks / Elsevier (journal option, faster review, more forgiving on scale)

**Single best next action**: Run CIFAR-100 training to 50,000 steps (or 300 epochs) for all 5 methods (dense, magnitude, random, linear, cubic) with 3 seeds each, and add a Lagrangian baseline. This single experiment will determine whether the paper has a future as a methods contribution or should be reframed as an analysis paper.

---

*Report generated by SPRO audit protocol. All assessments are evidence-based, referencing specific sections, tables, and figures from the manuscript and supplementary materials.*

Sources:
- [NeurIPS 2026 Dates and Deadlines](https://neurips.cc/Conferences/2026/Dates)
- [NeurIPS 2026 Call for Papers](https://neurips.cc/Conferences/2026/CallForPapers)
- [TMLR - Transactions on Machine Learning Research](https://jmlr.org/tmlr/)
- [TMLR Journal-to-Conference Track](https://medium.com/@TmlrOrg/tmlr-joins-neurips-icml-iclr-journal-to-conference-track-937a898eab3d)
- [ECCV 2026 Dates](https://eccv.ecva.net/Conferences/2026/Dates)
- [CVPR 2026 Dates](https://cvpr.thecvf.com/Conferences/2026/Dates)
- [ICLR 2026 Dates](https://iclr.cc/Conferences/2026/Dates)
- [Efficient Expert Pruning for Sparse MoE (OpenReview)](https://openreview.net/forum?id=TTUtPIpaol)
- [Survey on Efficient Vision Transformers (IEEE)](https://ieeexplore.ieee.org/document/10508091/)
