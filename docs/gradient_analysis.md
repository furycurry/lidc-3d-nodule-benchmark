# Gradient Analysis: Mechanistically Explaining the Null Result

`statistical_analysis.md` shows that boundary-aware loss terms produce no statistically significant improvement over the Dice+Focal baseline, on either architecture, on validation or held-out test data. This document asks why. It uses gradient-magnitude telemetry (`grad_telemetry_check.py`) to measure the boundary term's influence at the optimization level rather than the metric level, then checks how that measurement relates to the behavior of the adaptive gate and to a tenfold weight increase.

## 1. Method

`grad_telemetry_check.py` is a standalone, non-invasive diagnostic. It loads a trained checkpoint, runs several real validation batches through it, and uses `torch.autograd.grad` to measure the gradient norm contributed by each loss component separately:

- The **regional loss** (Dice + Focal) gradient norm.
- The **boundary loss** gradient norm at its raw, unweighted magnitude (weight = 1.0), which isolates how large the boundary signal is before any scaling.

Both norms are averaged over batches. The script reports the boundary term's **raw share** of the combined gradient norm at weight 1.0 (`boundary_raw_share_pct`, boundary norm divided by the sum of the two norms), and its **effective share** after scaling by the trained weight (λ = 0.05).

The script does not modify `train.py`. It takes `--architecture` and `--seed` arguments and reads the checkpoint `boundary_{architecture}_seed{seed}/best_dice_unet3d.pt`. Fixed-schedule checkpoints are used deliberately. The fixed schedule ramps to λ = 0.05 regardless of seed, whereas an adaptive checkpoint from a run whose gate never opened (Section 4) would trivially show about 0% boundary contribution and add no information.

## 2. Result: UNet3DPaper

Fixed-schedule boundary checkpoint, seed 42:

```
Regional (Dice+Focal) gradient norm, mean over batches: 0.000855
Boundary (raw, weight=1.0) gradient norm, mean over batches: 0.000008
Boundary share of combined raw gradient magnitude: 0.94%
At the actual trained weight (0.05), boundary's effective contribution is scaled down further: ~0.05%
```

The boundary term's raw gradient norm (8.0 × 10⁻⁶) is roughly two orders of magnitude below the regional term's (8.55 × 10⁻⁴) before any weighting. This is a raw share of about 0.94%. The printed norms are rounded to six decimals, so the exact share cannot be reconstructed from them (the UNETR JSON, Section 3, shows the same field computed from full-precision norms). Scaling by λ = 0.05 brings the effective contribution to about 0.05%.

## 3. Result: UNETR

The same script, adapted only in model instantiation and checkpoint path (the loss and gradient-measurement logic is unchanged), was run on the fixed-schedule UNETR checkpoint, seed 42:

| Architecture | Raw share (weight = 1.0) | Effective share (λ = 0.05) |
|---|---|---|
| UNet3DPaper (19.07M params) | ≈ 0.94% | ≈ 0.05% |
| UNETR (92.69M params) | 0.64% | 0.03% |

The boundary term is more starved on the transformer than on the CNN, in both raw and effective terms. The mechanism that explains the null result on UNet3DPaper therefore carries over to UNETR rather than being a property of one architecture.

The UNETR telemetry (`outputs/test_results/grad_telemetry_unetr.json`) gives full-precision norms: regional 9.93 × 10⁻⁴, boundary (raw, weight 1.0) 6.44 × 10⁻⁶, raw share 0.644%, effective share at λ = 0.05 of 0.0324%. The regional norm is similar in size to the CNN's (8.55 × 10⁻⁴) and the boundary norm is about 20% smaller, which together account for the lower share.

## 4. Adaptive gate behavior

The adaptive variant (PCG-BW) opens a Schmitt-trigger gate when validation-loss convergence velocity stays below `GATE_TAU_ENTER = 0.005` for `GATE_PATIENCE = 3` consecutive epochs. The realized boundary weight is read directly from the per-epoch `history` in each results JSON (`gate_active`, `boundary_weight`, `velocity`).

### 4.1 UNet3DPaper

All five seeds opened the gate at least once, but late (first active epoch 33–42, in the last quarter of training), and the realized weight was small:

| Seed | First active epoch | Pattern | Max weight | Mean weight, whole run |
|---|---|---|---|---|
| 42 | 42 | one activation, holds to epoch 50 | 0.0053 | 0.0003 |
| 123 | 42 | one deactivation then reactivation | 0.0001 | 0.0000 |
| 456 | 33 | one deactivation then reactivation | 0.0149 | 0.0010 |
| 789 | 35 | one activation, holds to epoch 50 | 0.0231 | 0.0032 |
| 2024 | 42 | one activation, holds to epoch 50 | 0.0227 | 0.0018 |

The maximum across all seeds and epochs was **0.0231**, under half of λ_max = 0.05. Two seeds (42 and 123) never reached a weight of 0.01, so "the gate opened" overstates the boundary term's role in those runs. The fixed schedule, by contrast, holds the full 0.05 from epoch 26 to epoch 50. The UNet3DPaper adaptive configuration therefore tests a much weaker boundary signal than the fixed one, which is consistent with its results being almost identical to the baseline.

### 4.2 UNETR

Only one of five seeds opened the gate:

| Seed | Pattern | Max weight | Min velocity | Qualifying epochs | Longest streak (3 needed) |
|---|---|---|---|---|---|
| 42 | never activated | 0.0000 | 0.00183 | 1/47 | 1 |
| 123 | never activated | 0.0000 | 0.00262 | 2/47 | 1 |
| 456 | never activated | 0.0000 | 0.00290 | 3/47 | 2 |
| 789 | never activated | 0.0000 | 0.00354 | 1/47 | 1 |
| 2024 | one clean activation | 0.0264 | −0.00010 | 5/47 | 3 (triggered) |

"Qualifying" means velocity below the entry threshold. Velocity dipped below 0.005 in every seed, so the gate is reachable. The qualifying epochs were sparse and mostly isolated, and only seed 2024 reached the required streak of three, at the minimum. UNETR's validation loss converges less smoothly than the CNN's, and the patience requirement was calibrated on the CNN.

Two consequences follow:

1. For four of five seeds, the UNETR "boundary_adaptive" configuration was functionally the baseline: the boundary term was zero for the whole run. The UNETR adaptive comparisons in `statistical_analysis.md` should be read with that in mind.
2. This fits the gradient result. A boundary term with 0.03% effective share perturbs validation loss too little to move the velocity signal that drives the gate.

## 5. Interpretation

The null result does not show that boundary-aware supervision is ineffective for this task. At the weights explored (λ ≤ 0.05), the boundary term's gradient is too small relative to the regional loss to change the direction of optimization. A term contributing 0.03–0.05% of the effective gradient magnitude has almost no room to move parameters differently than the regional loss alone would, whatever the schedule (fixed or gated) that controls when the small signal is applied.

### 5.1 Causal probe

A tenfold increase of the target boundary weight (λ = 0.50, an effective share of about 0.5%) was run on UNet3DPaper, seed 42, under the identical warmup schedule, to test whether raising the weight alone would produce surface refinement. The run was numerically stable. Last-10-epoch validation Dice was 0.7406 ± 0.0032 and HD95 was 5.2741 ± 0.2199 mm (mean ± SD over the final 10 epochs), against 0.7402 and 5.3617 mm for the same seed at λ = 0.05. Both differences are small relative to the epoch-to-epoch SD, so the extra weight produced no measurable change. This is consistent with the boundary signal being uninformative beyond what Dice + Focal already supply, rather than merely under-weighted, but a single-seed probe cannot establish that.


## 6. Scope and limitations

- Each architecture was measured at **one checkpoint** (fixed-schedule, seed 42) and one snapshot, not tracked across the training trajectory. A per-epoch gradient trajectory was deliberately not pursued. The roughly 100× raw gap is large enough that it is unlikely to close during training, but this document reports a diagnostic snapshot, not a full characterization of gradient dynamics.
- Gradient norms are measured on validation batches of a trained model, not on training batches during optimization.
- The adaptive-gate analysis (Section 4) uses data already logged in the results JSONs. No retraining was done for it.
- The probe in Section 5.1 tests one higher weight on one seed. It does not rule out effects at other weights, schedules, or normalizations.

## 7. Implication

The finding points to gradient-magnitude-aware balancing, for example GradNorm-style rescaling of the boundary term to a magnitude comparable to the regional loss, as a more informative next experiment than further tuning of the gating schedule. This is listed as future work in the main README and is not attempted here.
