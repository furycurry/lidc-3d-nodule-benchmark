# Statistical Analysis

This document reports every statistical comparison in the benchmark: boundary-loss variants against the Dice+Focal baseline (validation and held-out test), the two SDF-approximation audits, and the UNet3DPaper vs. UNETR architecture comparison. Numbers come from the per-run JSONs in `outputs/results/` and `outputs/test_results/`, via `aggregate_results.py`, `statistical_test.py`, `analyze_test_results.py`, `compare_architectures.py`, and the audit comparison scripts.

## 0. Conventions

- **Configurations:** `baseline` (Dice+Focal), `boundary_fixed` (fixed-schedule boundary loss), `boundary_adaptive` (convergence-gated boundary loss).
- **Seeds:** 42, 123, 456, 789, 2024 (N = 5), identical across configurations and architectures. Seeds are the pairing unit.
- **Tests:** paired t-test and Wilcoxon signed-rank test, both reported. α = 0.05. No claim rests on a single seed.
- **Spread:** all ± values are **population SD** (`np.std`, ddof = 0). Cohen's d for paired comparisons uses the population SD of the differences.
- **Wilcoxon floor:** with 5 pairs, the smallest attainable two-sided p is 0.0625 (2/32), reached only when all five pairs agree in direction. A value of 0.0625 in a 5-seed comparison therefore means the strongest possible rank agreement, not a borderline result.
- **Power:** with N = 5, a non-significant result cannot rule out small effects. The argument for equivalence in Sections 1–3 rests on consistency across metrics, configurations, and independent retrains, not on any single p-value.
- **Why multi-seed:** an early single-seed comparison suggested the adaptive variant outperformed the baseline. It did not replicate across five seeds.

## 1. Validation results, UNet3DPaper

### 1.1 Aggregate (N = 5 per configuration, best-Dice epoch)

| Config | N | Dice (mean ± SD) | HD95 mm (mean ± SD) | IoU (mean ± SD) |
|---|---|---|---|---|
| baseline | 5 | 0.7419 ± 0.0015 | 5.0825 ± 0.1455 | 0.6519 ± 0.0116 |
| boundary_fixed | 5 | 0.7413 ± 0.0020 | 5.0093 ± 0.1548 | 0.6501 ± 0.0142 |
| boundary_adaptive | 5 | 0.7417 ± 0.0037 | 5.1377 ± 0.1898 | 0.6445 ± 0.0079 |

### 1.2 Per-seed

**Dice**

| Seed | baseline | boundary_fixed | boundary_adaptive |
|---|---|---|---|
| 42 | 0.7444 | 0.7431 | 0.7438 |
| 123 | 0.7418 | 0.7383 | 0.7352 |
| 456 | 0.7425 | 0.7439 | 0.7436 |
| 789 | 0.7406 | 0.7408 | 0.7403 |
| 2024 | 0.7403 | 0.7404 | 0.7455 |

**HD95 (mm)**

| Seed | baseline | boundary_fixed | boundary_adaptive |
|---|---|---|---|
| 42 | 4.846 | 4.8695 | 4.938 |
| 123 | 5.2275 | 5.2896 | 5.3799 |
| 456 | 5.1302 | 4.8928 | 5.2684 |
| 789 | 4.991 | 4.9344 | 5.2084 |
| 2024 | 5.2176 | 5.0603 | 4.8939 |

### 1.3 Pairwise tests

**Dice**

| Comparison | Mean A | Mean B | Paired t | t p-value | Wilcoxon p |
|---|---|---|---|---|---|
| baseline vs. boundary_fixed | 0.7419 | 0.7413 | 0.729 | 0.5061 | 1.0000 |
| baseline vs. boundary_adaptive | 0.7419 | 0.7417 | 0.132 | 0.9011 | 1.0000 |
| boundary_fixed vs. boundary_adaptive | 0.7413 | 0.7417 | −0.276 | 0.7965 | 1.0000 |

**HD95**

| Comparison | Mean A | Mean B | Paired t | t p-value | Wilcoxon p |
|---|---|---|---|---|---|
| baseline vs. boundary_fixed | 5.0825 | 5.0093 | 1.314 | 0.2590 | 0.4375 |
| baseline vs. boundary_adaptive | 5.0825 | 5.1377 | −0.570 | 0.5989 | 0.6250 |
| boundary_fixed vs. boundary_adaptive | 5.0093 | 5.1377 | −1.375 | 0.2410 | 0.3125 |

**IoU** (computed after the original analysis): paired t-test p between 0.40 and 0.55 and Wilcoxon p = 0.625 for all three comparisons.

No comparison is significant on any of Dice, HD95, or IoU, under either test. Mean Dice differences (≤ 0.0006) are small against SDs of 0.0015–0.0037.

### 1.4 Size-stratified validation Dice

Nodules are split by ground-truth foreground voxel count into small, medium, and large terciles. Dice at each run's best-Dice epoch is flat across the three configurations: small ≈ 0.71–0.72, medium ≈ 0.75–0.76, large ≈ 0.76. The ordering small < medium < large holds in 13 of 15 runs individually. The two exceptions are `boundary_adaptive` seeds 456 and 789, where large falls just below medium. The small-vs-larger gap holds in all 15 runs.

### 1.5 Run integrity (15 UNet3DPaper runs)

All 15 runs share code-version fingerprint `6d28de4f780d`, completed 50/50 epochs, and used about 15.8 GPU-hours in total at 9.76–9.78 GB peak GPU memory per run. `hd95_skipped = 0` at every run's best-Dice epoch. Isolated skipped-HD95 events occurred as late as epoch 49 but never affected a reported statistic.

## 2. SDF approximation audits

Two geometric approximations in the cached SDF could in principle mask a real boundary-loss effect. Each was quantified, then tested by retraining both boundary configurations on corrected SDFs for all five seeds (10 runs each, 20 total) and comparing against the original runs with the same paired tests.

<!-- TODO(verify): the "Original" columns below (e.g. fixed Dice 0.7385) differ from the best-epoch means in Section 1.1 (0.7413). Confirm the metric definition used by compare_corrected_sdf.py (likely a last-N-epoch mean) and state it here in one line. -->

### 2.1 Patch-boundary truncation

The SDF is computed inside a 64³ patch with no knowledge of nodule extent beyond the patch edge. A reference SDF computed on a 128³ wide mask and center-cropped to 64³ was used as truth.

- 304 of 2,651 nodules (11.5%) have true extent touching or exceeding the patch boundary.
- Mean SDF error within 5 mm of the surface: 0.191 mm for truncated nodules vs. 0.023 mm for non-truncated (about 8×). Whole-patch mean error 0.015 mm. Maximum error 19.0 mm, near the ±20 clip ceiling.

Retrain with corrected SDFs (original vs. corrected):

| Config | Metric | Original | Corrected | Diff | t-test p | Wilcoxon p |
|---|---|---|---|---|---|---|
| fixed | Dice | 0.7385 ± 0.0027 | 0.7356 ± 0.0026 | −0.0029 | 0.1169 | 0.1875 |
| fixed | HD95 | 5.4018 ± 0.1660 | 5.4981 ± 0.2029 | +0.0964 | 0.4649 | 0.4375 |
| fixed | IoU | 0.6501 ± 0.0142 | 0.6397 ± 0.0127 | −0.0104 | 0.1221 | 0.1250 |
| adaptive | Dice | 0.7348 ± 0.0041 | 0.7348 ± 0.0023 | −0.0000 | 0.9933 | 1.0000 |
| adaptive | HD95 | 5.5812 ± 0.2654 | 5.5813 ± 0.1739 | +0.0001 | 0.9996 | 1.0000 |
| adaptive | IoU | 0.6445 ± 0.0079 | 0.6398 ± 0.0075 | −0.0048 | 0.5048 | 0.8125 |

### 2.2 Rotation under augmentation

Training applies `RandRotated` (±0.3 rad per axis) to the cached SDF by bilinear interpolation instead of recomputing it from the rotated mask, which violates the Eikonal condition. Using MONAI's own `Rotate` on 1,000 samples (500 nodules × 2 angle draws), the error against an SDF recomputed from the rotated wide mask was:

- Whole-patch mean 0.060 mm (max 0.388 mm).
- Near-boundary mean 0.240 mm (max 2.954 mm).

Retrain with the SDF recomputed on the fly from the rotated wide mask:

| Config | Metric | Original | Corrected | Diff | t-test p | Wilcoxon p |
|---|---|---|---|---|---|---|
| fixed | Dice | 0.7385 ± 0.0027 | 0.7347 ± 0.0032 | −0.0037 | 0.1122 | 0.1250 |
| fixed | HD95 | 5.4018 ± 0.1660 | 5.4981 ± 0.1039 | +0.0963 | 0.4140 | 0.6250 |
| fixed | IoU | 0.6501 ± 0.0142 | 0.6383 ± 0.0139 | −0.0118 | 0.3489 | 0.4375 |
| adaptive | Dice | 0.7348 ± 0.0041 | 0.7370 ± 0.0007 | +0.0022 | 0.3350 | 0.3125 |
| adaptive | HD95 | 5.5812 ± 0.2654 | 5.4300 ± 0.0661 | −0.1512 | 0.2515 | 0.4375 |
| adaptive | IoU | 0.6445 ± 0.0079 | 0.6411 ± 0.0145 | −0.0034 | 0.6849 | 0.6250 |

### 2.3 Verdict

All 12 comparisons are non-significant (minimum p = 0.1122). The adaptive configuration under truncation correction is nearly identical to the original (Dice 0.7348 vs. 0.7348, HD95 5.581 vs. 5.581). With N = 5 the tests cannot exclude small effects; the evidence is the consistency across two configurations, three metrics, and two independent corrections. The adaptive configuration's cross-seed variance is much lower in both corrected runs (for example Dice SD 0.0041 → 0.0007 under rotation correction). A plausible but untested explanation is that SDF noise made the timing of gate activation seed-dependent.

Neither audit was repeated for UNETR (see Section 5.5).

## 3. Held-out test set, UNet3DPaper

All 15 best-Dice checkpoints were evaluated once on the 434-patch, 132-patient test partition (one-look protocol). No result influenced any later model or loss decision.

| Config | Test Dice (mean ± SD) |
|---|---|
| baseline | 0.7303 ± 0.0011 |
| boundary_fixed | 0.7266 ± 0.0048 |
| boundary_adaptive | 0.7293 ± 0.0034 |

**The null result replicates on untouched data.** Pairwise comparisons across Dice, HD95, IoU, and ASSD are all non-significant (all p > 0.17). Surface-distance metrics at boundary tolerances (NSD at 1 mm and 2 mm, `compute_test_nsd.py`) also show no significant difference between configurations.

<!-- TODO(verify): add the NSD@1mm / NSD@2mm means and p-values from the NSD output; the handoff only records "no significant difference". -->

Size-stratified test Dice is similar across configurations: small ≈ 0.686–0.690, medium ≈ 0.732–0.741, large ≈ 0.768–0.769.

**Validation-to-test gap.** Dice and IoU generalize well, with test within about 0.01 of validation for all three configurations. Distance metrics degrade consistently: HD95 +1.06 to +1.37 mm, HD100 +1.25 to +1.39 mm, ASSD +0.23 to +0.42 mm. This asymmetry between overlap and distance metrics appears in all three configurations.

## 4. Held-out test set, UNETR

All 15 UNETR checkpoints were evaluated under the same protocol.

| Config | Test Dice mean |
|---|---|
| baseline | 0.6439 (SD 0.0032) |
| boundary_fixed | 0.6470 |
| boundary_adaptive | 0.6478 |

Across all 15 runs, per-run Dice lies between 0.6396 and 0.6504. The recipe reaches a stable ceiling and is not noisy.

The null result replicates. The closest comparison is baseline vs. boundary_fixed on Dice (t-test p = 0.0549, Wilcoxon p = 0.0625, the floor for N = 5), still above 0.05. Most other comparisons have p > 0.35. Full per-config tables are in `outputs/test_results/test_analysis_report_unetr.txt`.

**Caveat on the adaptive configuration.** Only 1 of 5 UNETR adaptive runs ever opened the gate (`gradient_analysis.md`, Section 4.2). For the other four seeds the adaptive run was identical in loss to the baseline. The UNETR adaptive-vs-baseline comparison is therefore largely uninformative. The fixed-schedule comparison is the substantive UNETR test of boundary loss.

**Larger generalization gap than the CNN.** Dice falls 0.023–0.027 from validation to test in every UNETR configuration (about 0.01 for the CNN). HD95 worsens by 1.7–1.9 mm and ASSD by 1.0–1.1 mm.

## 5. Architecture comparison: UNet3DPaper vs. UNETR

`compare_architectures.py` pairs the two architectures by configuration and seed on the frozen test results (N = 5 per configuration, N = 15 pooled). It reads existing JSONs and does no new inference.

### 5.1 Pooled comparison (N = 15)

| Metric | UNet3DPaper | UNETR | t-test p | Wilcoxon p | Cohen's d |
|---|---|---|---|---|---|
| Dice | 0.7287 ± 0.0038 | 0.6462 ± 0.0029 | < 0.001 | < 0.001 | 16.32 |
| HD95 (mm) | 6.6916 ± 0.3541 | 9.2702 ± 0.1482 | < 0.001 | < 0.001 | −6.85 |
| IoU | 0.6503 ± 0.0052 | 0.5399 ± 0.0087 | < 0.001 | < 0.001 | 12.47 |
| ASSD (mm) | 1.8135 ± 0.1582 | 3.4621 ± 0.1206 | < 0.001 | < 0.001 | −8.71 |

The pooled Dice gap is 0.0825 (8.25 points) and the pooled HD95 gap is 2.58 mm. Every per-configuration comparison on every metric also has t-test p < 0.001, with Wilcoxon p = 0.0625 (the N = 5 floor). The per-seed Dice difference (UNETR minus UNet3DPaper) ranges from −0.0718 to −0.0912 across the 15 matched pairs, always in the same direction. With |d| between 6 and 16, the two architectures' distributions do not overlap on any metric.

<!-- TODO(verify): the latest project handoff states "+8.48 pooled test Dice points". The pooled means above give 8.25, and the per-config means give 8.25 as well. Reconcile before this number appears in the README or paper. -->

### 5.2 Size-stratified test Dice (pooled)

| Size | UNet3DPaper | UNETR |
|---|---|---|
| Small | 0.6886 ± 0.0056 | 0.6482 ± 0.0062 |
| Medium | 0.7375 ± 0.0069 | 0.6632 ± 0.0033 |
| Large | 0.7684 ± 0.0054 | 0.6169 ± 0.0079 |

UNet3DPaper improves with nodule size. UNETR peaks at medium and is worst on large nodules, a qualitatively different pattern and not only a lower level.

### 5.3 Texture subgroup

On subtle non-solid ground-glass nodules, pooled test Dice is 0.5349 for UNETR vs. 0.6670 for UNet3DPaper, a much larger relative gap than overall. Subgroup labels come from `nodule_clinical_metadata.json` (radiologist consensus ratings).

<!-- TODO(verify): add subgroup N, the exact subgroup definition (texture rating cutoff), and whether the 0.5349 / 0.6670 figures are pooled over configs and seeds. -->

### 5.4 What the comparison rules in and out

- **Capacity is ruled out.** UNETR has 4.9× the parameters (92.69M vs. 19.07M) and performs worse.
- **Missing pretraining is ruled out.** The original UNETR paper reports that pretraining did not help it.
- **Training length is not the explanation.** The 50-epoch recipe is shorter than the original paper's 20,000 iterations at batch 6 (about 87 epochs on this dataset), but a 100-epoch probe did not close the gap, so the deficit looks structural.
- **Leading explanation (hypothesis, not tested here):** a 92.7M-parameter ViT trained from scratch on 1,833 fixed patches lacks the translation-equivariant priors that help the CNN, and MONAI's fixed 16³ tokenization gives only 64 tokens per 64³ patch, which limits boundary and large-structure precision. This is consistent with the large-nodule degradation, the ground-glass collapse, and the larger validation-to-test gap on distance metrics.

### 5.5 Limits of the comparison

- Pairing by seed controls shared data-order and initialization randomness but is not the same kind of repeated-measures design as the within-architecture loss comparisons, which use the same architecture and checkpoint lineage. Effect sizes are large enough that no conclusion depends on this, but the two kinds of pairing should not be read as equivalent rigor.
- UNETR was trained with the CNN's recipe (50 epochs, AdamW, LR 1e-4, no warmup, batch 8) for a clean same-protocol comparison. This is a deliberate choice, not a tuned configuration.
- The SDF truncation and rotation audits were not repeated for UNETR. UNETR's effective boundary gradient share (0.03%) is lower than the CNN's (0.05%), for which both approximations were shown not to matter, which makes SDF error an even less plausible explanation for its null result. This is an argument, not an empirical check.
- The benchmark uses one dataset, one fixed patient-level split, and nodule-centered patches (a segmentation task, not detection).

## 6. Summary

| Question | Answer | Evidence |
|---|---|---|
| Does boundary loss improve segmentation? | No, on either architecture | Sections 1, 3, 4 (all p > 0.05) |
| Is the null result an SDF artifact? | No, for UNet3DPaper | Section 2 (12 comparisons, all p > 0.11) |
| Does the null replicate on held-out data? | Yes | Sections 3 and 4 |
| Does it replicate across architectures? | Yes | Sections 3, 4; mechanism in `gradient_analysis.md` |
| Does architecture matter? | Strongly: the CNN is better on every metric | Section 5 |
