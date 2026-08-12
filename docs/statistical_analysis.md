# Statistical Analysis

This document presents the full multi-seed experimental results and statistical comparison for the three loss configurations evaluated on the LIDC-IDRI 3D pulmonary nodule segmentation task.

---

## 1. Experimental Design

- **Architecture:** `UNet3DPaper` (faithful 3D U-Net reimplementation, 19,073,665 parameters)
- **Dataset:** 2,651 patches (64³ voxels), patient-level 3-way split (1,833 train / 384 val / 434 test)
- **Seeds:** 5 independent random seeds — 42, 123, 456, 789, 2024
- **Configurations:** 3 loss variants × 5 seeds = **15 total runs**
- **Epochs:** 50 per run
- **Checkpoint selection:** Dual-criterion — best validation Dice and best validation HD95 saved independently

---

## 2. Aggregate Results

### 2.1 Summary Table

| Config | N | Val Dice (mean ± SD) | Val HD95 mm (mean ± SD) |
|---|---|---|---|
| **Baseline** (Dice + Focal) | 5 | 0.7419 ± 0.0015 | 5.0825 ± 0.1455 |
| **Boundary loss** (fixed schedule) | 5 | 0.7413 ± 0.0020 | 5.0093 ± 0.1548 |
| **Boundary loss** (adaptive, hysteresis-gated) | 5 | 0.7417 ± 0.0037 | 5.1377 ± 0.1898 |

### 2.2 Per-Seed Breakdown

#### Baseline (Dice + Focal)

| Seed | Best Val Dice | Best Val HD95 (mm) |
|---|---|---|
| 42 | 0.7444 | 4.846 |
| 123 | 0.7418 | 5.2275 |
| 456 | 0.7425 | 5.1302 |
| 789 | 0.7406 | 4.991 |
| 2024 | 0.7403 | 5.2176 |

#### Boundary Loss — Fixed Schedule

| Seed | Best Val Dice | Best Val HD95 (mm) |
|---|---|---|
| 42 | 0.7431 | 4.8695 |
| 123 | 0.7383 | 5.2896 |
| 456 | 0.7439 | 4.8928 |
| 789 | 0.7408 | 4.9344 |
| 2024 | 0.7404 | 5.0603 |

#### Boundary Loss — Adaptive Hysteresis-Gated

| Seed | Best Val Dice | Best Val HD95 (mm) |
|---|---|---|
| 42 | 0.7438 | 4.938 |
| 123 | 0.7352 | 5.3799 |
| 456 | 0.7436 | 5.2684 |
| 789 | 0.7403 | 5.2084 |
| 2024 | 0.7455 | 4.8939 |

---

## 3. Pairwise Statistical Tests

All comparisons use **matched-seed pairing** (same seed across configurations). Two tests were applied per comparison:
- **Paired t-test** — parametric, assumes normal distribution of differences
- **Wilcoxon signed-rank test** — non-parametric, robust to outliers

Significance threshold: $\alpha = 0.05$.

### 3.1 Dice Coefficient

| Comparison | Mean Diff | Paired t-test | Wilcoxon |
|---|---|---|---|
| Baseline vs. Fixed boundary | +0.0006 | $t = 0.729$, $p = 0.5061$ | $p = 1.0000$ |
| Baseline vs. Adaptive boundary | +0.0002 | $t = 0.132$, $p = 0.9011$ | $p = 1.0000$ |
| Fixed vs. Adaptive boundary | −0.0004 | $t = -0.276$, $p = 0.7965$ | $p = 1.0000$ |

### 3.2 HD95 (mm)

| Comparison | Mean Diff (mm) | Paired t-test | Wilcoxon |
|---|---|---|---|
| Baseline vs. Fixed boundary | +0.0732 | $t = 1.314$, $p = 0.2590$ | $p = 0.4375$ |
| Baseline vs. Adaptive boundary | −0.0552 | $t = -0.570$, $p = 0.5989$ | $p = 0.6250$ |
| Fixed vs. Adaptive boundary | −0.1284 | $t = -1.375$, $p = 0.2410$ | $p = 0.3125$ |

---

## 4. Interpretation

### 4.1 No Statistically Significant Differences

**No pairwise comparison reaches statistical significance** at $\alpha = 0.05$. The lowest observed p-value is **0.2410** (fixed vs. adaptive boundary on HD95, paired t-test) — nearly 5× above the significance threshold. Most p-values fall in the 0.3–1.0 range, indicating substantial overlap in the distributions.

### 4.2 Effect Size

The mean differences between configurations are numerically small:
- **Dice:** maximum mean difference = 0.0006 (baseline vs. fixed boundary) — less than 0.1% relative.
- **HD95:** maximum mean difference = 0.1284 mm (fixed vs. adaptive boundary) — approximately 2.5% relative.

Given the null hypothesis cannot be rejected, these differences are indistinguishable from random seed-to-seed variation.

### 4.3 Variance Patterns

- **Baseline** shows the tightest Dice variance ($\sigma = 0.0015$) but moderate HD95 variance.
- **Fixed-schedule boundary** shows comparable Dice variance ($\sigma = 0.0020$) and the lowest mean HD95 (5.0093 mm), though not significantly so.
- **Adaptive boundary** shows the highest variance in both metrics ($\sigma_{Dice} = 0.0037$, $\sigma_{HD95} = 0.1898$), suggesting the hysteresis-gating mechanism may introduce additional seed-dependent instability despite its design intent.

### 4.4 Consistency Across Tests

Paired t-test and Wilcoxon signed-rank test agree directionally on all comparisons. The Wilcoxon p-values are generally higher (or equal), consistent with its lower power for small sample sizes ($N = 5$). Neither test approaches significance.

---

## 5. Conclusion

At $N = 5$ seeds, there is **no evidence** that either fixed-schedule or adaptive hysteresis-gated boundary loss provides a statistically significant improvement over the Dice+Focal baseline, or over each other, on either Dice overlap or HD95 boundary precision.

This null result is further explained by the gradient telemetry diagnosis (see `gradient_analysis.md`): the boundary loss term contributes approximately **0.05% of total gradient magnitude** at its trained weight ($\lambda = 0.05$) — too small to meaningfully influence optimization, regardless of scheduling strategy.

---

## 6. Data Provenance

Results were generated by running `aggregate_results.py` and `statistical_test.py`
against all 15 output files in `outputs/results/`.