# Cover Letter — KaleidoNet

**Paste this into the SR portal "Cover Letter" field.**

---

Dear Editors of *Scientific Reports*,

We submit for your consideration the manuscript "**Why Lagrangian Pruning Fails in Mixture-of-Experts: Gradient-Scale Mismatch and a Schedule-Based Fix**" as a research article.

The paper is an **analysis paper**: we identify and empirically validate a previously-undocumented failure mode of a widely-used neural-network pruning recipe, and we propose a simple, principled fix. We believe it is well-matched to *Scientific Reports*' acceptance criterion of scientific validity rather than perceived importance, for three reasons.

**First, the diagnosis is precise.** Adding a Lagrangian FLOPs penalty to a neural-network loss is the standard way to train under a compute budget. We show that when this recipe is applied to prune individual neurons inside Mixture-of-Experts (MoE) layers, the per-neuron penalty gradient is approximately two orders of magnitude weaker than the task gradient on the same variable. The optimiser cannot use this signal to select useful neurons; it satisfies the budget by collapsing all mask logits uniformly. We derive the gradient scaling analytically (Methods, Eq. 3) and confirm it empirically across CIFAR-10, CIFAR-100, and Tiny-ImageNet, where a properly tuned global-λ Lagrangian baseline reaches the target FLOPs budget but degrades CIFAR-100 accuracy to ≈10% — barely above random.

**Second, the fix is practical.** We propose KaleidoNet, which replaces the Lagrangian penalty with a deterministic cubic sparsity schedule, irreversible gradient masking, and dual-rate optimisation. Across three seeds trained to convergence (50,000 steps), KaleidoNet reduces active FLOPs by 1.80×, retains 98.3–99.2% of dense-model accuracy, and outperforms every schedule-based pruning baseline tested.

**Third, we strengthen the diagnosis with five additional sparsity-inducing variants** (per-layer λ, per-(block, expert) λ, augmented Lagrangian, FLOPs-gradient rescaling, and L0 / Hard-Concrete differentiable sparsity, the last being the closest non-Lagrangian alternative). All five collapse to ≈9–10% on CIFAR-100 — within ±1 pp of the global-λ baseline — confirming that the gradient-scale issue is not specific to a Lagrangian penalty but is a property of any per-neuron sparsity-inducing mechanism whose differentiable signal is too weak relative to the task gradient at MoE-expert granularity. The schedule-based methods bypass this by removing the structural decision from the gradient path entirely.

We are honest about the remaining limitation: at the small model scale used here, MoE routing overhead makes wall-clock inference slower than the dense baseline, an artefact of small expert widths that is expected to shrink at ViT-Large scale but has not been validated empirically. The contribution is a diagnostic insight plus a stable, best-in-class pruning method with a clear FLOPs-efficiency profile.

The manuscript was internally pre-reviewed against major-revisions peer-review criteria before submission; all five strengthened-Lagrangian variants and the L0/Hard-Concrete baseline were added in direct response to that pre-review.

All authors have approved the manuscript, which is not under consideration elsewhere. No competing interests apply.

Sincerely,
Karim Magdy
(On behalf of all authors)
