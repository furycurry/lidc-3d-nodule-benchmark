# Statistical Analysis

This document reports every statistical comparison in the benchmark: boundary-loss variants against the Dice+Focal baseline (validation and held-out test), the two SDF-approximation audits, and the UNet3DPaper vs. UNETR architecture comparison. Numbers come from the per-run JSONs in `outputs/results/` and `outputs/test_results/`, via `aggregate_results.py`, `statistical_test.py`, `analyze_test_results.py`, `compare_architectures.py`, and the audit comparison scripts.

## 0. Conventions

- **Configurations:** `baseline` (Dice+Focal), `boundary_fixed` (fixed-schedule boundary loss), `boundary_adaptive` (convergence-gated boundary loss).
- **Seeds:** 42, 123, 456, 789, 2024 (N = 5), identical across configurations and architectures. Seeds are the pairing unit.
- **Tests:** paired t-test and Wilcoxon signed-rank test, both reported. α = 0.05. No claim rests on a single seed.
- **Spread:** all ± values are **population SD** (`np.std`, ddof = 0). Cohen's d for paired comparisons uses the population SD of the differences.
- **Wilcoxon floor:** with 5 pairs, the smallest attainable two-sided p is 0.0625 (2/32), reached only when all five pairs agree in direction. A value of 0.0625 in a 5-seed comparison therefore means the strongest possible rank agreement, not a borderline result.
- **Power:** with N = 5, a non-significant result cannot rule out small effects. The argument for equivalence in Sections 1–3 rests on consistency across metrics, configurations, and independent retrains, not on any single p-value.
- **Multiple comparisons:** test-set subgroup results use Bonferroni correction over two families: the 12 volume-stratified comparisons (α ≈ 0.00417) and the 28 boundary-vs-baseline test comparisons that are reported (α ≈ 0.00179). The size of the second family depends on which comparisons are counted (Section 4.1).
- **Why multi-seed:** an early single-seed comparison suggested the adaptive variant outperformed the baseline. It did not replicate across five seeds.

## 1. Validation results, UNet3DPaper

### 1.1 Aggregate (N = 5 per configuration)

| Config | N | Dice, best epoch (mean ± SD) | HD95 mm, best epoch (mean ± SD) | IoU, final epoch (mean ± SD) |
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

**IoU** (final-epoch `final_val_iou`): paired t-test p between 0.40 and 0.55 and Wilcoxon p = 0.625 for all three comparisons. IoU at each run's best-Dice epoch is similar (0.6525 ± 0.0051, 0.6496 ± 0.0168, 0.6474 ± 0.0164).

No comparison is significant on any of Dice, HD95, or IoU, under either test. Mean Dice differences (≤ 0.0006) are small against SDs of 0.0015–0.0037.

### 1.4 Size-stratified validation Dice

Nodules are split by ground-truth foreground voxel count into small, medium, and large tiers (validation N = 145 / 148 / 91, the same voxel thresholds as the test tiers). Dice at each run's best-Dice epoch is flat across the three configurations: small ≈ 0.71–0.72, medium ≈ 0.75–0.76, large ≈ 0.76. The ordering small < medium < large holds in 13 of 15 runs individually. The two exceptions are `boundary_adaptive` seeds 456 and 789, where large falls just below medium. The small-vs-larger gap holds in all 15 runs.

### 1.5 Run integrity (15 UNet3DPaper runs)

All 15 UNet3DPaper runs share code-version fingerprint `6d28de4f780d`, completed 50/50 epochs, and used about 15.8 GPU-hours in total at 9.76–9.78 GB peak GPU memory per run (all on PyTorch 2.11.0+cu128). The 15 UNETR runs share a different fingerprint, `925996db30b0` (the code after UNETR support was added), completed 50/50 epochs, and used about 7.8 GPU-hours at 4.40–4.42 GB peak memory per run. `hd95_skipped = 0` at every run's best-Dice epoch in both sets. Isolated skipped-HD95 events occurred as late as epoch 49 but never affected a reported statistic.

### 1.6 Validation results, UNETR

| Config | Dice, best epoch | HD95 mm, best epoch | IoU, final epoch | Dice, last-10-epoch mean |
|---|---|---|---|---|
| baseline | 0.6763 ± 0.0025 | 7.2329 ± 0.0674 | 0.5012 ± 0.0129 | 0.6711 ± 0.0026 |
| boundary_fixed | 0.6795 ± 0.0019 | 7.1925 ± 0.0704 | 0.5135 ± 0.0109 | 0.6731 ± 0.0034 |
| boundary_adaptive | 0.6775 ± 0.0034 | 7.2542 ± 0.0318 | 0.4960 ± 0.0207 | 0.6704 ± 0.0016 |

One validation comparison is nominally significant: best-epoch Dice, baseline vs. fixed (+0.0032, t(4) = −3.398, t-test p = 0.0273; Wilcoxon p = 0.0625, the N = 5 floor). It is uncorrected for multiple comparisons, it disappears in the last-10-epoch Dice (p = 0.182), HD95 (p = 0.463) and IoU (p = 0.311), and it does not carry over to the test set (Section 4). Every other UNETR validation comparison has p > 0.18. For UNETR, the validation Dice by size tier peaks at medium nodules and is lowest on large ones in all 15 runs (small ≈ 0.674–0.680, medium ≈ 0.704–0.706, large ≈ 0.630–0.635), the same pattern as on the test set.

## 2. SDF approximation audits

Two geometric approximations in the cached SDF could in principle mask a real boundary-loss effect. Each was quantified, then tested by retraining both boundary configurations on corrected SDFs for all five seeds (10 runs each, 20 total) and comparing against the original runs with the same paired tests.

**Metric note.** The tables below come from `compare_corrected_sdf.py`, which uses the last-10-epoch mean validation Dice and HD95 (`val_dice_mean_last10`, `val_hd95_mean_last10`) and the final-epoch IoU. This is why the "Original" columns (for example fixed Dice 0.7385) differ from the best-epoch means in Section 1.1 (0.7413). The manuscript reports the truncation audit on peak validation Dice (`best_val_dice`). Both aggregates are given in Section 2.3, because they do not agree exactly for the rotation audit.

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

### 2.3 Best-epoch aggregate

Peak validation metrics (`best_val_dice`, `best_val_hd95`), original vs. corrected, 5 matched seeds:

| Audit | Config | Metric | Original | Corrected | Diff | t-test p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| truncation | fixed | Dice | 0.7413 ± 0.0020 | 0.7420 ± 0.0014 | +0.0007 | 0.5324 | 1.0000 |
| truncation | fixed | HD95 | 5.0093 ± 0.1548 | 5.0434 ± 0.0779 | +0.0340 | 0.6809 | 0.6250 |
| truncation | adaptive | Dice | 0.7417 ± 0.0037 | 0.7402 ± 0.0016 | −0.0015 | 0.4841 | 0.6250 |
| truncation | adaptive | HD95 | 5.1377 ± 0.1898 | 5.1045 ± 0.0473 | −0.0332 | 0.7262 | 0.8125 |
| rotation | fixed | Dice | 0.7413 ± 0.0020 | 0.7388 ± 0.0015 | −0.0025 | **0.0204** | 0.0625 |
| rotation | fixed | HD95 | 5.0093 ± 0.1548 | 5.1359 ± 0.1017 | +0.1266 | 0.0669 | 0.1250 |
| rotation | adaptive | Dice | 0.7417 ± 0.0037 | 0.7410 ± 0.0018 | −0.0007 | 0.8108 | 1.0000 |
| rotation | adaptive | HD95 | 5.1377 ± 0.1898 | 5.0978 ± 0.0941 | −0.0399 | 0.7437 | 1.0000 |

Under the last-10-epoch aggregate (and final-epoch IoU) all 12 comparisons in Sections 2.1 and 2.2 have p ≥ 0.1122. Under the best-epoch aggregate one comparison is nominally significant: the fixed schedule with rotation-corrected SDFs has slightly *lower* peak Dice (−0.0025, t-test p = 0.0204, Wilcoxon p = 0.0625). That p-value is uncorrected, it is one of 20 comparisons in this section, and its direction (corrected SDF is no better) does not support the idea that SDF error was hiding a benefit.

### 2.4 Verdict

Under the last-10-epoch aggregate, all 12 comparisons in Sections 2.1 and 2.2 are non-significant (minimum p = 0.1122). The adaptive configuration under truncation correction is nearly identical to the original (Dice 0.7348 vs. 0.7348, HD95 5.581 vs. 5.581). With N = 5 the tests cannot exclude small effects; the evidence is the consistency across two configurations, three metrics, and two independent corrections. The adaptive configuration's cross-seed variance is much lower in both corrected runs (for example Dice SD 0.0041 → 0.0007 under rotation correction). A plausible but untested explanation is that SDF noise made the timing of gate activation seed-dependent.

Neither audit was repeated for UNETR (see Section 5.5).

## 3. Held-out test set, UNet3DPaper

All 15 best-Dice checkpoints were evaluated once on the 434-patch, 132-patient test partition (one-look protocol). No result influenced any later model or loss decision.

| Config | Test Dice (mean ± SD) |
|---|---|
| baseline | 0.7303 ± 0.0011 |
| boundary_fixed | 0.7266 ± 0.0048 |
| boundary_adaptive | 0.7293 ± 0.0034 |

**The null result replicates on untouched data.** Pairwise comparisons across Dice, HD95, IoU, and ASSD are all non-significant (all p > 0.17). Surface-distance metrics at boundary tolerances (NSD at 1 mm and 2 mm, `compute_test_nsd.py`) also show no significant difference between configurations.

Full test metrics (mean ± SD, N = 5 seeds). `hd95_skipped` (cases where HD95 could not be computed and the maximum-diagonal penalty is applied) is 0–1 per run for UNet3DPaper and 0–2 per run for UNETR, which inflates UNETR's mean HD95 by roughly 0.2–0.5 mm per run, so the 2.58 mm pooled HD95 gap is slightly overstated. The Dice gap is unaffected. ΔDice is test Dice minus the validation last-10-epoch mean Dice (`val_dice_mean_last10`). Using the validation Dice of the selected checkpoint (`best_val_dice`) instead gives larger gaps (UNet3DPaper baseline −0.0117, UNETR baseline −0.0323), because the best epoch is selected on validation.

| Config | Dice | IoU | HD95 (mm) | ASSD (mm) | NSD@1mm | NSD@2mm | ΔDice (test − val) |
|---|---|---|---|---|---|---|---|
| baseline | 0.7303 ± 0.0011 | 0.6529 ± 0.0042 | 6.5435 ± 0.2346 | 1.7334 ± 0.0750 | 0.8693 ± 0.0014 | 0.9214 ± 0.0022 | −0.0076 |
| boundary_fixed | 0.7266 ± 0.0048 | 0.6493 ± 0.0041 | 6.7700 ± 0.3405 | 1.8890 ± 0.1502 | 0.8655 ± 0.0051 | 0.9183 ± 0.0050 | −0.0118 |
| boundary_adaptive | 0.7293 ± 0.0034 | 0.6487 ± 0.0061 | 6.7614 ± 0.4151 | 1.8181 ± 0.1864 | 0.8686 ± 0.0037 | 0.9218 ± 0.0041 | −0.0055 |

Paired t-test p-values: baseline vs. fixed, Dice 0.2579, HD95 0.3513, ASSD 0.1718, NSD@1mm 0.2722, NSD@2mm 0.3493; baseline vs. adaptive, Dice 0.6679, HD95 0.4818.

Test Dice by fixed volume tier (small < 100 voxels, N = 148; medium 100–500, N = 175; large > 500, N = 111), for baseline / fixed / adaptive: small 0.6900 / 0.6894 / 0.6863, medium 0.7401 / 0.7320 / 0.7405, large 0.7684 / 0.7679 / 0.7688. Boundary loss shows no benefit in any tier (all p > 0.11 for UNet3DPaper; the fixed schedule on medium nodules is lower than baseline in all five seeds, p = 0.120).

**Validation-to-test gap.** Dice and IoU generalize well, with test within about 0.01 of validation for all three configurations. Distance metrics degrade consistently: HD95 +1.06 to +1.37 mm, HD100 +1.25 to +1.39 mm, ASSD +0.23 to +0.42 mm. This asymmetry between overlap and distance metrics appears in all three configurations.

## 4. Held-out test set, UNETR

All 15 UNETR checkpoints were evaluated under the same protocol.

| Config | Dice | IoU | HD95 (mm) | ASSD (mm) | NSD@1mm | NSD@2mm | ΔDice (test − val) |
|---|---|---|---|---|---|---|---|
| baseline | 0.6439 ± 0.0032 | 0.5399 ± 0.0062 | 9.3519 ± 0.1500 | 3.4988 ± 0.1640 | 0.7750 ± 0.0038 | 0.8409 ± 0.0045 | −0.0271 |
| boundary_fixed | 0.6470 ± 0.0015 | 0.5393 ± 0.0121 | 9.2651 ± 0.1168 | 3.4395 ± 0.0376 | 0.7788 ± 0.0023 | 0.8437 ± 0.0024 | −0.0261 |
| boundary_adaptive | 0.6478 ± 0.0022 | 0.5405 ± 0.0066 | 9.1935 ± 0.1308 | 3.4480 ± 0.1153 | 0.7793 ± 0.0019 | 0.8446 ± 0.0020 | −0.0227 |

Across all 15 runs, per-run Dice lies between 0.6396 and 0.6504. The recipe reaches a stable ceiling and is not noisy.

**Full-cohort comparisons.** None reaches significance. Baseline vs. fixed: Dice p = 0.0549 (t-test; Wilcoxon 0.0625, the N = 5 floor), HD95 0.4678, ASSD 0.5693, NSD@1mm 0.0512, NSD@2mm 0.2226. Baseline vs. adaptive: Dice p = 0.1671. The closest to the 0.05 line is NSD@1mm at 0.0512, then Dice at 0.0549. All full-cohort comparisons across both architectures have p ≥ 0.051. Full per-config tables are in `outputs/test_results/test_analysis_report_unetr.txt`.

**Caveat on the adaptive configuration.** Only 1 of 5 UNETR adaptive runs ever opened the gate (`gradient_analysis.md`, Section 4.2). For the other four seeds the adaptive run used no boundary term. Across those four seeds (42, 123, 456, 789) the mean Dice difference from baseline was +0.0041 ± 0.0050 (per seed +0.0023, +0.0108, −0.0028, +0.0062), which gives an empirical measure of run-to-run noise at this sample size. The one seed whose gate did open (2024) differed by +0.0026, inside that noise. Seed 123, with the largest difference (+0.0108, baseline 0.6396 vs. 0.6504), never had a boundary term. The UNETR adaptive-vs-baseline comparison is therefore largely uninformative, and the fixed-schedule comparison is the substantive UNETR test of boundary loss.

**Larger generalization gap than the CNN.** Dice falls 0.023–0.027 from validation to test in every UNETR configuration (0.006–0.012 for the CNN). HD95 worsens by 1.7–1.9 mm and ASSD by 1.0–1.1 mm. On the baseline configuration the UNETR drop is about 3.6 times the CNN's (−0.0271 vs. −0.0076).

### 4.1 Volume-stratified comparisons

Test Dice by fixed volume tier, mean ± SD over 5 seeds (small < 100 voxels, N = 148; medium 100–500, N = 175; large > 500, N = 111):

| Architecture | Config | Small | Medium | Large |
|---|---|---|---|---|
| UNet3DPaper | baseline | 0.6900 ± 0.0049 | 0.7401 ± 0.0029 | 0.7684 ± 0.0031 |
| | fixed | 0.6894 ± 0.0060 | 0.7320 ± 0.0074 | 0.7679 ± 0.0056 |
| | adaptive | 0.6863 ± 0.0052 | 0.7405 ± 0.0056 | 0.7688 ± 0.0068 |
| UNETR | baseline | 0.6443 ± 0.0055 | 0.6598 ± 0.0017 | 0.6184 ± 0.0082 |
| | fixed | 0.6500 ± 0.0048 | 0.6633 ± 0.0024 | 0.6173 ± 0.0068 |
| | adaptive | 0.6503 ± 0.0063 | 0.6664 ± 0.0018 | 0.6150 ± 0.0082 |

Ten of the twelve volume-stratified boundary-vs-baseline Dice comparisons show no significant difference (all p > 0.11). This includes every UNet3DPaper comparison (small nodules: all p ≥ 0.374; the closest is the fixed schedule on medium nodules at p = 0.120, which is 0.0081 *lower* than baseline in all five seeds, Wilcoxon 0.0625) and UNETR on small nodules (fixed: t(4) = 1.520, p = 0.2032, Wilcoxon p = 0.3125; adaptive: t(4) = 1.171, p = 0.3067).

The two exceptions are UNETR on medium nodules:

- **Fixed schedule:** +0.0035 Dice, t(4) = 7.498, p = 0.00169.
- **Adaptive schedule:** +0.0066 Dice, t(4) = 6.551, p = 0.00281.

Both survive Bonferroni correction within the 12 volume-stratified tests (α ≈ 0.00417). Under the test-wide family of 28 reported comparisons (α ≈ 0.00179) only the fixed shift survives, narrowly (p = 0.00169), and the adaptive shift does not. Wilcoxon p is 0.0625 for both, the N = 5 floor. The adaptive medium shift arises largely in runs where the gate was never active (λ ≡ 0), so it reflects run-to-run variation (for example differences in random-number consumption between code paths) and not boundary regularization. The fixed shift is small and does not carry into the full-cohort overlap or surface metrics (p ≥ 0.051).

The 28-test family is defined by the comparisons reported (12 volume-stratified, 13 full-cohort, 3 margin-morphology). It leaves out IoU and some NSD comparisons that were also computed. If those are counted (39 tests: all six metrics for both losses on both architectures, plus the 12 tier tests and 3 margin tests), α falls to about 0.00128 and the fixed medium shift (p = 0.00169) would no longer survive. Treat it as an exploratory observation that needs replication.

### 4.2 Clinical-attribute strata

See Section 5.3 for the clinical-subgroup tables, including the test of whether boundary loss helps spiculated margins (it does not: Dice 0.7431 vs. 0.7429, p = 0.9679).

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

The manuscript reports the same +8.25 pooled Dice points. The "+8.48" in an earlier handoff was a transcription error. On the baseline configuration alone the gap is 8.63 points (unrounded means 0.73025 vs. 0.64395; the rounded values 0.7303 and 0.6439 would give 8.64). Pooled NSD@1mm is 0.868 vs. 0.778.

### 5.2 Size-stratified test Dice (pooled over the three loss configurations)

| Tier (voxels, N) | UNet3DPaper | UNETR |
|---|---|---|
| Small (< 100, N = 148) | 0.6886 ± 0.0056 | 0.6482 ± 0.0062 |
| Medium (100–500, N = 175) | 0.7375 ± 0.0069 | 0.6632 ± 0.0033 |
| Large (> 500, N = 111) | 0.7684 ± 0.0054 | 0.6169 ± 0.0079 |

UNet3DPaper improves with nodule size. UNETR peaks at medium and is worst on large nodules, a qualitatively different pattern and not only a lower level.

### 5.3 Clinical-attribute subgroups

Subgroup labels come from `nodule_clinical_metadata.json` (mean radiologist rating per nodule). Test Dice for the baseline loss configuration, mean ± SD over 5 seeds:

| Axis | Subgroup (N) | UNet3DPaper | UNETR | Cohen's d | t-test p |
|---|---|---|---|---|---|
| Texture | Non-solid / ground-glass ≤ 2.0 (51) | 0.6670 ± 0.0088 | 0.5349 ± 0.0121 | +7.49 | < 0.001 |
| | Part-solid 2.0–4.0 (38) | 0.6884 ± 0.0081 | 0.5683 ± 0.0074 | +21.58 | < 0.001 |
| | Solid ≥ 4.0 (345) | 0.7442 ± 0.0029 | 0.6684 ± 0.0029 | +13.80 | < 0.001 |
| Subtlety | Subtle ≤ 2.5 (64) | 0.6485 ± 0.0050 | 0.5515 ± 0.0043 | +12.46 | < 0.001 |
| | Moderate 2.5–3.5 (114) | 0.7119 ± 0.0028 | 0.6259 ± 0.0035 | +17.19 | < 0.001 |
| | Obvious > 3.5 (256) | 0.7589 ± 0.0015 | 0.6751 ± 0.0052 | +13.22 | < 0.001 |
| Spiculation | Smooth ≤ 2.0 (379) | 0.7284 ± 0.0014 | 0.6398 ± 0.0034 | +19.65 | < 0.001 |
| | Spiculated > 2.0 (55) | 0.7431 ± 0.0064 | 0.6727 ± 0.0033 | +11.19 | < 0.001 |
| Malignancy | Low ≤ 2.5 (160) | 0.7224 ± 0.0026 | 0.6630 ± 0.0036 | +10.65 | < 0.001 |
| | Indeterminate 2.5–3.5 (211) | 0.7212 ± 0.0015 | 0.6164 ± 0.0034 | +22.71 | < 0.001 |
| | High > 3.5 (63) | 0.7805 ± 0.0047 | 0.6879 ± 0.0052 | +17.12 | < 0.001 |

UNETR collapses on faint, low-attenuation lesions: 0.5349 on ground-glass nodules and 0.5515 on subtle nodules.

**Does boundary loss help irregular margins?** On UNet3DPaper (baseline vs. fixed):

| Subgroup | Metric | Baseline | Fixed | Difference | t-test p |
|---|---|---|---|---|---|
| Spiculated (> 2.0) | Dice | 0.7431 ± 0.0064 | 0.7429 ± 0.0060 | −0.0002 | 0.9679 |
| Spiculated (> 2.0) | NSD@1mm | 0.8711 ± 0.0042 | 0.8698 ± 0.0058 | −0.0013 | 0.7120 |
| Smooth (≤ 2.0) | Dice | 0.7284 ± 0.0014 | 0.7243 ± 0.0051 | −0.0041 | 0.2247 |

The Wilcoxon p for the spiculated Dice comparison is 1.0000. Boundary supervision gave no benefit even on the most irregular margins.

The same comparison for the other loss and architecture (Dice, baseline vs. config; recomputed by `tools/make_tables_clinical.py`):

| Arch | Config | Spiculated diff (p) | Smooth diff (p) |
|---|---|---|---|
| UNet3DPaper | fixed | −0.0002 (0.968) | −0.0041 (0.225) |
| UNet3DPaper | adaptive | −0.0016 (0.625) | −0.0009 (0.696) |
| UNETR | fixed | −0.0073 (0.157) | +0.0046 (0.028) |
| UNETR | adaptive | −0.0036 (0.425) | +0.0049 (0.086) |

Across all 11 clinical strata, both losses and both architectures (44 uncorrected comparisons), the smallest p-values are UNETR adaptive on moderate-subtlety nodules (+0.0095, p = 0.011) and UNETR fixed on smooth nodules (+0.0046, p = 0.028). None survives Bonferroni correction (α = 0.05/44 ≈ 0.0011). UNETR's differences are mostly small and positive under the fixed schedule, with the exception of spiculated nodules (−0.0073), so there is no consistent subgroup in which boundary loss helps. The NSD@1mm comparison for spiculated nodules (0.8711 vs. 0.8698, p = 0.7120) cannot be recomputed from the stored files, because per-sample NSD was not saved.

### 5.4 What the comparison rules in and out

- **Capacity is ruled out.** UNETR has 4.9× the parameters (92.69M vs. 19.07M) and performs worse.
- **Missing pretraining is ruled out.** The original UNETR paper reports that pretraining did not help it.
- **Training length is not the explanation.** The 50-epoch recipe is shorter than the original paper's 20,000 iterations at batch 6 (about 87 epochs on this dataset). A 100-epoch UNETR baseline probe (seed 42) reached a validation Dice peak of 0.6914 at epoch 97 and a last-10-epoch mean of 0.6857 ± 0.0035. The 50-epoch run with the same seed had a peak of 0.6741 (epoch 48) and a last-10-epoch mean of 0.6684, so doubling the training gained about 1.7 Dice points. The probe still left the model about 5.3 points below the CNN's seed-42 baseline (best 0.7444, last-10-epoch mean 0.7410 ± 0.0027). The peak came at epoch 97 of 100, so validation was still creeping up. The probe shows that doubling the training does not close the gap. It does not show that UNETR has plateaued, so "structural" is a reading of the 5-point gap, not something this single-seed probe proves.
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
| Is the null result an SDF artifact? | No, for UNet3DPaper | Section 2 (no corrected SDF improved on the original; one best-epoch Dice comparison is nominally *worse*, p = 0.020) |
| Does the null replicate on held-out data? | Yes on all full-cohort metrics. One small UNETR medium-nodule shift (+0.35 Dice points) is exploratory (Section 4.1) | Sections 3 and 4 |
| Does it replicate across architectures? | Yes | Sections 3, 4; mechanism in `gradient_analysis.md` |
| Does architecture matter? | Strongly: the CNN is better on every metric | Section 5 |
