# SPRO CRITIQUE -- Round 7 (FINAL)

**Paper**: "Why Lagrangian Pruning Fails in Mixture-of-Experts: Gradient-Scale Mismatch and a Schedule-Based Fix"  
**Authors**: Karim Magdy, Ghada Khoriba, Hala Abbas  
**Target venue**: NeurIPS 2026 (primary) / TMLR (fallback)  
**Review date**: 2026-04-03  
**Score trajectory**: R1:7.0 --> R2:4.5 --> R3:5.5 --> R4:6.5 --> R5:7.0 --> R6:7.5 --> **R7:8.0**

---

## 1. Overall Submission Readiness Score

**8.0 / 10** (up from 7.5)

All eight R6 P1 items have been credibly addressed. The paper is now a coherent, honest, well-scoped contribution ready for top-venue review. The remaining issues are polish-level (P2/P3) and do not block submission.

---

## 2. Executive Punchline

The paper identifies a genuine, previously undocumented failure mode -- per-neuron Lagrangian gradient-scale mismatch in MoE architectures -- and validates it with catastrophic empirical evidence on three datasets, which is the strongest element of this work. The fix (KaleidoNet's cubic schedule) is clean and effective at the tested scale, though the 18x inference overhead and small-model-only evidence remain disclosed limitations rather than fatal flaws. This paper is submittable to NeurIPS 2026 (deadline: May 6) with minor polish, and TMLR is a strong fallback where the honest scope and reproducibility emphasis align well with editorial criteria.

---

## 3. Editor's First Impression

**Positive signals:**
- Clear diagnostic contribution: the gradient-scale mismatch analysis (Eq. 2, Section 3.2) is the kind of "why does X fail?" insight that reviewers remember.
- Honest framing: scale limitations, inference overhead (18x), statistical power limitations (p=0.07-0.11) are all disclosed upfront rather than buried.
- Concurrent work paragraph (Section 2, final paragraph) positions this paper correctly: the *diagnosis* is the novelty, not the schedule itself.
- Complete experimental protocol: 3 datasets, 3 seeds, 6 methods, 50k steps to convergence, per-seed tables in appendix.
- NeurIPS checklist is present and properly filled.

**Concerns a desk editor would flag:**
- 5.49M-parameter ViT on 32x32 images -- scale is acknowledged but still invites "toy experiment" perception at a top venue.
- Cubic vs. linear gap is 0.2-0.7 pp with p-values of 0.07-0.11 -- the schedule *shape* is not statistically significant; the paper correctly reframes this, but some reviewers will still push.
- 18x inference slowdown is a hard sell for a "compute efficiency" paper, even with the ViT-Large extrapolation argument.

**Overall**: A well-crafted diagnostic paper with an effective fix. The narrative is honest and the writing is strong. Likely to receive mixed-positive reviews at NeurIPS (borderline accept) and a clean accept at TMLR.

---

## 4. Major Weaknesses (Referenced by Section)

### W1. Scale gap remains the elephant in the room (Section 5.1, Section 6)
The conjecture that gradient-scale mismatch generalizes to larger models (Section 5.1, "Scale context" paragraph) is now properly hedged as a conjecture rather than a claim. However, the paper still lacks any evidence beyond the 5.49M ViT. The ViT-Large routing overhead estimate (Section 5.6) is a helpful back-of-envelope calculation, but reviewers will distinguish between "we predict it would work" and "we showed it works." **Risk**: NeurIPS Reviewer 2 will ask for at least one ImageNet-1K run. **Mitigation**: The honest framing limits damage; TMLR would not block on this.

### W2. Cubic vs. Linear is not statistically significant (Section 5.2)
The paired t-test p-values (0.07, 0.07, 0.11) are correctly reported as "trending but underpowered." The paper's reframing -- that the diagnostic matters more than the specific schedule shape -- is the right strategy. However, the title says "Schedule-Based Fix" which implies the schedule details matter. **Risk**: A reviewer may argue the contribution reduces to "just use any schedule instead of Lagrangian," which is a less novel claim. **Mitigation**: The 0.7 pp gap on CIFAR-100 is consistent across all 3 seeds (58.2, 58.6, 58.4 vs. 58.9, 59.3, 59.1), suggesting a real effect that could reach significance with 5+ seeds.

### W3. No wall-clock speedup at any tested scale (Section 5.6)
The 18x inference slowdown is fully disclosed but remains a weakness. The ViT-Large extrapolation (18x -> 2-3x) is plausible but unvalidated. For a paper about "FLOPs reduction," the absence of any latency benefit is a rhetorical liability. The paper handles this well by framing the contribution as "accuracy-FLOPs tradeoff" rather than "speedup," but some reviewers will still find this unsatisfying.

### W4. Ablation at 2,000 steps is not at convergence (Section 5.4)
The ablation study (Table 3) uses 2,000 steps where elastic pruning actively hurts. The paper explains why (pruning is still in progress), but a reviewer may ask: "Why not ablate at 50k steps?" Running the 8-configuration ablation at convergence would cost ~8x more compute but would be much more convincing. As written, the ablation's key message is weakened by the short training budget.

### W5. Five uncited bibliography entries (references.bib)
The following bib entries are defined but never cited: `coates2011analysis`, `he2018soft`, `hinton2012neural`, `le2015tiny`, `smith2019super`. Notably, `le2015tiny` is the Tiny-ImageNet reference that *should* be cited when introducing the dataset (Section 5.1). The others are likely orphaned from earlier drafts. This is a minor cleanliness issue that could trigger a desk reviewer's "sloppy" perception.

### W6. MoE-Pruner citation venue is incorrect (references.bib, line 186)
MoE-Pruner (moepruner2025) is cited as "ICLR 2025" but was actually a *withdrawn* submission from ICLR 2025 per OpenReview. It should be cited as an arXiv preprint (arXiv:2410.12013, 2024). Citing a withdrawn paper as an accepted venue publication is a factual error that could undermine credibility if a reviewer notices.

---

## 5. Fatal Flaws

**None.** All previously identified fatal flaws (R1-R6) have been addressed:

| ID | Original flaw | Status |
|----|--------------|--------|
| R6-P1.1 | No concurrent work discussion | FIXED -- SlimMoE, STUN, MoNE, MoE-Pruner added |
| R6-P1.2 | Pareto figure FLOPs mismatch | FIXED -- Dense=236.7M matches Table 1 |
| R6-P1.3 | No statistical tests | FIXED -- p-values reported with honest "underpowered" caveat |
| R6-P1.4 | Scale claim too strong | FIXED -- downgraded to "conjecture" |
| R6-P1.5 | No routing overhead analysis at scale | FIXED -- ViT-Large estimate added |
| R6-P1.6 | Active vs total param confusion | FIXED -- footnote added in appendix |
| R6-P1.7 | Bengio bib type wrong | FIXED -- now @article |
| R6-P1.8 | STL reference error | FIXED -- corrected to Tiny-ImageNet throughout |

The paper has no remaining fatal flaws that would warrant desk rejection.

---

## 6. Actionable Revision Plan

### Priority 1 (Must-fix before submission, ~2 hours)

**P1.1. Cite Tiny-ImageNet properly.**  
Section 5.1 mentions Tiny-ImageNet without a citation. Add `\citep{le2015tiny}` after "Tiny-ImageNet (200 classes, 64x64 downsampled to 32x32)." The bib entry already exists.

**P1.2. Fix MoE-Pruner venue.**  
Change `moepruner2025` from `@inproceedings{..., booktitle={ICLR}, year={2025}}` to `@article{..., journal={arXiv preprint arXiv:2410.12013}, year={2024}}`. The paper was withdrawn from ICLR 2025 and remains an arXiv preprint.

**P1.3. Remove uncited bib entries.**  
Delete `coates2011analysis`, `he2018soft`, `hinton2012neural`, `smith2019super` from references.bib. These are orphaned and add noise.

**P1.4. Verify page count fits NeurIPS 9-page limit.**  
The body (abstract through Section 6) contains ~332 non-empty content lines plus 4 figures and 5 tables. This is likely at the 9-page boundary. Compile and verify. If over 9 pages, compress the related work (currently ~1.5 pages with the concurrent work paragraph) or reduce vertical spacing around tables. The appendix and checklist do not count toward the limit.

### Priority 2 (Strongly recommended, ~4 hours)

**P2.1. Run ablation at convergence.**  
The 2,000-step ablation (Table 3) actively contradicts the paper's message. If compute allows, run at least the "All pillars" and "MoE + early exit" configurations at 50k steps to show the crossover. Even 2 configurations at convergence would strengthen Section 5.4 significantly.

**P2.2. Add 2 more seeds to tighten p-values.**  
The cubic-vs-linear comparison has p=0.07-0.11 with 3 seeds. With 5 seeds, a consistent 0.7 pp gap would likely reach p<0.05 on at least CIFAR-100. This would convert a "trending" result into a statistically significant one. Cost: ~10 GPU-hours on CIFAR-100.

**P2.3. Add a "Lagrangian with 10x lambda" experiment.**  
A reviewer will ask: "What if you just increase lambda to fix the gradient scale?" Show that large lambda causes the opposite failure (over-pruning to zero) or oscillation. This would strengthen the diagnostic argument that the problem is structural, not just a hyperparameter issue.

### Priority 3 (Nice-to-have, post-submission or camera-ready)

**P3.1. One ImageNet-1K sanity check.**  
Even a single-seed, 50k-step run on ImageNet-1K (not to convergence) showing that Lagrangian still collapses while the cubic schedule does not would dramatically strengthen the scale argument. Use a pretrained ViT-S/16 backbone with MoE adapter layers to reduce training cost.

**P3.2. Latency on GPU with MegaBlocks.**  
Running inference with the MegaBlocks sparse kernel (already cited) would give a more realistic latency number than the Apple MPS benchmark. Even showing 3-5x overhead instead of 18x would help.

**P3.3. Tighten discussion of "irreversible" pruning.**  
The gradient masking (Section 3.3) makes pruning decisions permanent. A reviewer may ask whether this leads to error accumulation. A brief remark about the monotonic schedule preventing oscillation would preempt this.

---

## 7. Journal Recommendation Matrix

| Venue | Fit | Deadline | Pros | Cons | Recommendation |
|-------|-----|----------|------|------|---------------|
| **NeurIPS 2026** | 7/10 | May 6, 2026 (abstract May 4) | High visibility; diagnostic papers valued; 9-page limit fits | Scale concerns may draw R2 rejection; cubic-vs-linear not significant; 18x overhead | **Primary target** if P1 items fixed and ideally P2.1-P2.2 done |
| **TMLR** | 9/10 | Rolling | Values soundness over novelty; honest scope rewarded; no "impact" bar; rolling timeline; Featured Certification possible | Lower visibility than NeurIPS; no conference presentation | **Strongest fit** -- submit here if NeurIPS rejected or if prefer certainty |
| **ICLR 2027** | 7/10 | ~Oct 2026 | Similar audience to NeurIPS; diagnostic contributions valued | 6 months away; paper may age vs. concurrent work | **Secondary** if NeurIPS rejected and want conference venue |
| **ECCV 2026** | 5/10 | TBD | Vision venue | Less ML theory focus; may not value the diagnostic angle | Not recommended |
| **NeurIPS 2026 Workshop** | 8/10 | ~Sep 2026 | Lower bar; good for visibility; can later submit full version to TMLR | Limited archival impact | **Fallback** if main track rejected |

**Recommended strategy**: Submit to NeurIPS 2026 main track (deadline May 6). Execute P1 fixes immediately and P2.1-P2.2 if time allows before May 4 abstract deadline. If rejected, revise with reviewer feedback and submit to TMLR (rolling). The paper's honesty and completeness are well-suited to TMLR's acceptance criteria (soundness + audience interest).

---

## 8. Cover Letter Advice

- **Lead with the diagnosis, not the fix.** Frame the paper as "we identify and empirically validate a fundamental failure mode of Lagrangian pruning in MoE architectures." The schedule fix is secondary. This positions the paper as a contribution to understanding, not just another pruning method.
- **Explicitly state what reviewers should NOT expect.** "This paper does not claim wall-clock speedup at the tested scale; it identifies why a widely-used pruning paradigm fails in MoE settings and proposes a principled alternative." Pre-empting the "no speedup" objection disarms it.
- **Highlight the Lagrangian collapse results.** The numbers (10.4% on CIFAR-100 = near random chance for 100 classes) are striking and memorable. These are the paper's signature finding.
- **Acknowledge the scale limitation proactively.** "Our experiments span three datasets at 32x32 resolution. We conjecture the gradient-scale mismatch generalizes to larger models but leave validation at ImageNet-1K scale to future work." Honesty builds trust with area chairs.
- **Suggest reviewers with MoE compression expertise.** The concurrent work paragraph (SlimMoE, STUN, MoNE, MoE-Pruner) establishes the paper in an active research area. Request area chairs familiar with efficient deep learning or MoE architectures.
- **Mention the complete reproducibility package.** Open-source code, all hyperparameters in appendix, per-seed results -- this signals seriousness. NeurIPS values reproducibility.
- **Note the NeurIPS checklist is complete.** All 15 items are answered with justifications. This reduces friction during initial screening.

---

## 9. Final Recommendation

**SUBMIT to NeurIPS 2026 after executing P1 fixes (estimated 2 hours of work).**

The paper has matured substantially across seven revision rounds. The core contribution -- diagnosing *why* Lagrangian pruning fails at per-neuron granularity in MoE, with compelling empirical validation -- is a genuine insight that will interest the efficient ML community. The fix (cubic schedule + gradient masking + dual-rate optimization) is simple, effective, and well-ablated at the tested scale.

The remaining weaknesses (scale, statistical power, no wall-clock speedup) are real but are *disclosed* weaknesses, not hidden ones. At NeurIPS, I estimate a 40-50% acceptance probability -- the diagnostic novelty and experimental honesty will attract champions, but the scale limitation will draw at least one skeptical review. At TMLR, I estimate 80-90% acceptance probability -- the paper clearly meets both acceptance criteria (supported claims + audience interest).

**Expected review outcome at NeurIPS**: Split decision (one champion who values the diagnostic, one skeptic on scale, one in the middle). Success depends on the champion's ability to argue the diagnostic alone justifies acceptance. The P2 items (convergence ablation, more seeds, lambda sweep) would meaningfully improve the odds.

---

## 10. Summary Box

```
+------------------------------------------------------------------+
|  SPRO FINAL AUDIT -- Round 7                                     |
|                                                                   |
|  Score: 8.0 / 10  (trajectory: 7>4.5>5.5>6.5>7>7.5>8)           |
|                                                                   |
|  Verdict: READY TO SUBMIT (with P1 fixes)                        |
|                                                                   |
|  Primary target:   NeurIPS 2026 (deadline May 6)                  |
|  Fallback target:  TMLR (rolling, highest confidence)             |
|                                                                   |
|  Fatal flaws:      0 (all 8 R6 items resolved)                   |
|  P1 fixes needed:  4 (cite TinyImageNet, fix MoE-Pruner venue,   |
|                       remove orphaned bib entries, verify pages)  |
|  P2 recommended:   3 (convergence ablation, more seeds,          |
|                       lambda sweep experiment)                    |
|                                                                   |
|  Strongest element:  Lagrangian collapse diagnosis + empirics     |
|  Weakest element:    Scale (5.49M ViT, 32x32 only)               |
|                                                                   |
|  NeurIPS acceptance odds:  40-50%                                 |
|  TMLR acceptance odds:     80-90%                                 |
+------------------------------------------------------------------+
```

---

## 9-Pillar Evaluation Summary

| Pillar | Score (1-10) | Notes |
|--------|-------------|-------|
| **Significance** | 7.5 | Genuine insight into Lagrangian failure in MoE; limited by small scale |
| **Novelty** | 7.0 | Diagnostic is novel; schedule fix uses known techniques (Zhu & Gupta 2018) |
| **Methodological Rigor** | 8.0 | Clean gradient analysis; 3 datasets x 3 seeds x 6 methods; honest statistics |
| **Results Integrity** | 8.5 | Per-seed appendix; p-values reported; FLOPs now consistent; no cherry-picking |
| **Interpretation** | 8.5 | Honest about limitations; scale caveat is a conjecture not a claim; overhead disclosed |
| **Writing Quality** | 8.0 | Clear, well-structured; title is informative; figures support narrative |
| **Ethics & Reproducibility** | 9.0 | Full hyperparams; open-source code; NeurIPS checklist complete; broader impact discussed |
| **Journal Fit** | 7.5 | NeurIPS fit is good but scale is a risk; TMLR fit is excellent |
| **Acceptance Readiness** | 8.0 | P1 fixes are minor (2 hours); paper is substantively complete |

---

*Audit performed under SPRO v7 protocol. No hallucinated claims. All journal details verified via web search (NeurIPS 2026: 9-page limit, deadline May 6; TMLR: rolling review, soundness-based acceptance; MoE-Pruner: withdrawn from ICLR 2025). Evidence-first reasoning throughout.*
