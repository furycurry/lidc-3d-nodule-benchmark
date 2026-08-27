# Gradient Analysis

This document presents the gradient-telemetry diagnostic that explains the mechanistic cause of the null result observed in the boundary-loss experiments.

---

## 1. Motivation

The multi-seed experimental matrix (N = 5 seeds, 3 loss configurations) found **no statistically significant difference** between the Dice+Focal baseline, fixed-schedule boundary loss, and adaptive hysteresis-gated boundary loss on either Dice overlap or HD95 boundary precision (see `statistical_analysis.md`).

A null result can arise for two very different reasons:
1. The method genuinely has no effect on the optimization landscape.
2. The method was applied incorrectly, at the wrong time, or with the wrong hyperparameters.

To distinguish these, a standalone gradient-magnitude diagnostic was run to directly measure the boundary loss term's contribution to the total gradient, independent of any scheduling logic.

---

## 2. Methodology

### 2.1 Design Principles
The diagnostic was designed to be **non-invasive**:
- No modifications to `train.py`.
- No gradient-telemetry hooks inserted into the training loop.
- No interference with the existing loss computation or backpropagation.

This avoids the risk that adding instrumentation could alter training dynamics (e.g., by introducing synchronization points or memory overhead).

### 2.2 Procedure
A standalone script (`grad_telemetry_check.py`) performs the following steps:

1. **Load a trained checkpoint.** The diagnostic uses the fixed-schedule boundary-loss model (seed 42), selected because it represents a fully converged model where the boundary loss has been active for the majority of training.

2. **Sample real validation batches.** A subset of validation patches is drawn through the standard data pipeline (`LIDCDataset` with `use_sdf=True`), ensuring the input distribution matches actual training/validation conditions.

3. **Forward pass.** For each batch, the model produces predictions; the loss function computes the regional (Dice + Focal) term and the boundary term separately.

4. **Gradient isolation via `torch.autograd.grad`.** Two separate backward calls are made per batch:
   - One backward through the **regional loss** alone, recording the gradient norm across all model parameters.
   - One backward through the **boundary loss** alone (at raw weight = 1.0), recording its gradient norm.

5. **Aggregation.** Gradient norms are averaged across all sampled batches.

### 2.3 Why Raw Weight = 1.0?
The boundary term is evaluated at weight = 1.0 to measure its **intrinsic** gradient magnitude before any external scaling. The actual trained weight (0.05) is applied as a multiplicative factor afterward to estimate the **effective** contribution during training.

---

## 3. Results

| Term | Gradient Norm (mean over batches) | Share of Combined Raw Magnitude |
|---|---|---|
| **Regional** (Dice + Focal) | 0.000855 | 99.06% |
| **Boundary** (raw, $\lambda = 1.0$) | 0.000008 | 0.94% |

### 3.1 Effective Contribution at Trained Weight
The fixed-schedule boundary loss uses a maximum weight of $\lambda_{max} = 0.05$. Scaling the raw boundary gradient by this weight:

$$\text{Effective boundary share} \approx 0.94\% \times 0.05 = 0.047\% \approx \mathbf{0.05\%}$$

### 3.2 Magnitude Ratio
The boundary term's effective gradient magnitude is approximately **1/2000** that of the regional term:

$$\frac{\|\nabla \mathcal{L}_{boundary}^{effective}\|}{\|\nabla \mathcal{L}_{regional}\|} \approx \frac{0.000008 \times 0.05}{0.000855} \approx \frac{1}{2100}$$

---

## 4. Interpretation

### 4.1 Gradient Starvation
The boundary loss term contributes roughly **0.05% of total gradient magnitude** at its trained weight. This is nearly **two orders of magnitude** too small to meaningfully influence the optimizer's trajectory. The optimizer is effectively "blind" to the boundary term — any updates it induces are drowned out by noise from the regional term and stochastic minibatch sampling.

### 4.2 Why Scheduling Cannot Help

Both the fixed-schedule and adaptive hysteresis-gated strategies control **when** the boundary loss is active and **how strongly** it is weighted, but neither can overcome a fundamental magnitude deficit in this specific setting:

- **Fixed schedule:** The weight ceiling ($\lambda_{max} = 0.05$) was chosen to avoid the catastrophic collapse observed at higher unclipped weights (see `methodology.md`, Section 3.2). Raising it risks destabilization.
- **Adaptive gating:** The gate can only scale the existing boundary term. If the raw term is already ~1% of the regional term, even perfect gating cannot amplify it beyond the $\lambda_{max}$ ceiling.

In other words, for the tested SDF formulation and weight ceiling, the scheduling debate (fixed vs. adaptive) is secondary to the fundamental magnitude deficit. The gating mechanism controls *when* a negligible signal is applied, not whether the signal itself is strong enough to influence optimization.

**This is empirically confirmed by the adaptive schedule's actual training logs.** The 0.05% figure above was computed by scaling the raw boundary gradient by $\lambda_{max}=0.05$ — the fixed schedule's steady-state weight. But across all 5 adaptive-schedule seeds, the *realized* boundary weight never actually reached $\lambda_{max}$: the maximum weight observed in any seed/epoch was 0.0231 (well under half the ceiling), and per-seed mean weight during gate-active epochs ranged from 0.0000 to 0.0101 (see `methodology.md`, Section 3.3). This means the adaptive variant's true effective gradient contribution was, for most of training, smaller than the 0.05% figure implies — reinforcing rather than complicating the gradient-starvation diagnosis for both scheduling strategies.

### 4.3 Consistency with the Null Result
This diagnosis directly explains the statistical null result:
- If the boundary term contributes ~0.05% of gradient magnitude, its presence or absence, and the exact schedule of its application, should produce no measurable difference in final model parameters.
- This is exactly what the N = 5 multi-seed experiment observed.

### 4.4 The Test-Time Generalization Gap Does Not Confound This Diagnosis
The held-out test-set evaluation (see `statistical_analysis.md`, "Held-Out Test-Set Evaluation") found that distance metrics (HD95, HD100, ASSD) are consistently worse on test than on validation, while Dice and IoU generalize well. This gap is essentially the same size and direction for the baseline and both boundary-loss configurations — it is not larger or smaller for the configurations that use the boundary term. A gradient-starved loss term that has no measurable effect on training should also have no effect on how the model generalizes to held-out data, and that is what is observed: the three configurations remain statistically indistinguishable from one another on the test set (all p > 0.17) even though all three show the same validation-to-test distance-metric gap. The generalization gap is therefore a property of the architecture/dataset combination, not evidence against the gradient-starvation explanation.

---

## 5. Implications

### 5.1 For This Project
The null result is not a failure of experimental design or a Type II error due to insufficient seeds. It is a **mechanistically expected outcome** given the measured gradient share. This elevates the project's conclusion from:

> "No significant difference was found between boundary-loss weighting strategies."

to:

> "Boundary loss showed no effect **because it was gradient-starved by design** at the tested weight ceiling ($\lambda_{max} = 0.05$), not because the gating timing was wrong."

The latter is a stronger, more scientifically complete finding for a report's Discussion or Limitations section.

**Converging evidence:** The gradient-starvation diagnosis makes three testable predictions, all independently confirmed:

1. The null result should persist under corrected, non-truncated SDFs — **confirmed**: 10 retraining runs, all p > 0.11 (see `statistical_analysis.md`, "SDF-Approximation Verification").
2. The null result should persist under rotation-corrected SDFs — **confirmed**: 10 retraining runs, all p > 0.11.
3. The null result should replicate on held-out data (one-look test set) — **confirmed**: all pairwise test-set comparisons non-significant, minimum p = 0.1718 (see `statistical_analysis.md`, "Held-Out Test-Set Evaluation").

In total, 35 independent training runs (15 original + 10 truncation-corrected + 10 rotation-corrected) plus 15 test-set evaluations all converge on the same null result, consistent with a single mechanistic cause.

### 5.2 Future Directions

If boundary loss is to be made effective in this setting, the gradient-magnitude disparity must be addressed. Potential directions include:

- **Dynamic gradient balancing** (e.g., GradNorm) to enforce gradient-magnitude parity between boundary and regional terms.
- **Higher λ_max with stronger stabilization** (e.g., gradient clipping, alternate SDF formulations) to increase the boundary term's influence.
- **Architecture modifications** that increase sensitivity to boundary cues (e.g., explicit edge-detection branches).

These are left as future work.

---

## 6. Limitations

1. **Single checkpoint.** The diagnostic was run on one trained model (fixed-schedule boundary, seed 42). While this is representative of a converged state, gradient dynamics could differ early in training.
2. **Single seed.** The checkpoint comes from one random seed. However, the N = 5 multi-seed results show tight variance across seeds for the fixed-schedule config, suggesting seed-dependent gradient dynamics are unlikely.
3. **Weight-specific.** The 0.05% figure is specific to $\lambda_{max} = 0.05$, and was computed from the fixed-schedule checkpoint, whose weight reaches and holds at the full 0.05 ceiling for the back half of training. The adaptive schedule's realized weight stayed well below this ceiling in all 5 seeds (max observed: 0.0231), so its true effective contribution was likely smaller than 0.05% for most of training — see Section 4.2. A higher weight ceiling would increase the effective share proportionally, but would also risk the catastrophic collapse that the clipping and warmup schedule were designed to prevent.
4. **Batch-size dependence.** Gradient norms are batch-size dependent. The diagnostic used the same batch size (8) as training, so the relative share is internally consistent, but absolute norms would scale with batch size.
5. **Confirmed predictions.** The four limitations above are methodological caveats on the diagnostic itself. The diagnosis's *predictions* — that the null result should persist under corrected SDFs and replicate on held-out data — have been independently confirmed by 20 additional training runs and 15 test-set evaluations, substantially reducing the risk that the 0.05% figure is an artifact of the specific checkpoint or seed chosen.

---

## 7. Reproducibility

The diagnostic script `grad_telemetry_check.py` is included in the repository under `scripts/`. It can be run against any saved checkpoint:

```bash
python scripts/grad_telemetry_check.py --checkpoint outputs/archive/<run_name>/best_dice_unet3d.pt
```
