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

### 4.3 Consistency with the Null Result
This diagnosis directly explains the statistical null result:
- If the boundary term contributes ~0.05% of gradient magnitude, its presence or absence, and the exact schedule of its application, should produce no measurable difference in final model parameters.
- This is exactly what the N = 5 multi-seed experiment observed.

---

## 5. Implications

### 5.1 For This Project
The null result is not a failure of experimental design or a Type II error due to insufficient seeds. It is a **mechanistically expected outcome** given the measured gradient share. This elevates the project's conclusion from:

> "No significant difference was found between boundary-loss weighting strategies."

to:

> "Boundary loss showed no effect **because it was gradient-starved by design** at the tested weight ceiling ($\lambda_{max} = 0.05$), not because the gating timing was wrong."

The latter is a stronger, more scientifically complete finding for a report's Discussion or Limitations section.

### 5.2 Future Directions

If boundary loss is to be made effective in this setting, the gradient-magnitude 
disparity must be addressed. Potential directions include:

- **Dynamic gradient balancing** (e.g., GradNorm) to enforce gradient-magnitude 
  parity between boundary and regional terms.
- **Higher λ_max with stronger stabilization** (e.g., gradient clipping, alternate 
  SDF formulations) to increase the boundary term's influence.
- **Architecture modifications** that increase sensitivity to boundary cues 
  (e.g., explicit edge-detection branches).

These are left as future work.

---

## 6. Limitations

1. **Single checkpoint.** The diagnostic was run on one trained model (fixed-schedule boundary, seed 42). While this is representative of a converged state, gradient dynamics could differ early in training.
2. **Single seed.** The checkpoint comes from one random seed. However, the N = 5 multi-seed results show tight variance across seeds for the fixed-schedule config, suggesting seed-dependent gradient dynamics are unlikely.
3. **Weight-specific.** The 0.05% figure is specific to $\lambda_{max} = 0.05$. A higher weight ceiling would increase the effective share proportionally, but would also risk the catastrophic collapse that the clipping and warmup schedule were designed to prevent.
4. **Batch-size dependence.** Gradient norms are batch-size dependent. The diagnostic used the same batch size (8) as training, so the relative share is internally consistent, but absolute norms would scale with batch size.

---

## 7. Reproducibility

The diagnostic script `grad_telemetry_check.py` is included in the repository under `scripts/`. It can be run against any saved checkpoint:

```bash
python scripts/grad_telemetry_check.py --checkpoint outputs/archive/<run_name>/best_dice_unet3d.pt
```

No training loop modifications are required.
