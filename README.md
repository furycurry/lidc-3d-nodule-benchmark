# 3D Pulmonary Nodule Segmentation on LIDC-IDRI: A Rigorous Multi-Seed Benchmark

A faithful reimplementation of the original 3D U-Net architecture ([Çiçek et al., 2016](https://arxiv.org/abs/1606.06650)), evaluated on patient-disjoint LIDC-IDRI CT data, with a statistically rigorous (N=5 seeds, paired significance testing) empirical study of fixed-schedule vs. a novel convergence-gated adaptive variant, verified via gradient telemetry, two independent SDF-approximation retrain-and-compare threads, and a one-look held-out test-set evaluation.

## Summary

This repository documents an end-to-end, leakage-free pipeline for 3D lung nodule segmentation, built with careful attention to common methodological pitfalls in medical imaging ML: patient-level data leakage, single-seed result instability, and metric-selection bias in checkpointing. The headline empirical finding is a **null result, diagnosed at the gradient level and independently confirmed four separate ways**: boundary-aware loss terms (Kervadec et al., 2019 formulation, both fixed-schedule and a novel convergence-gated adaptive variant) show no statistically significant improvement over standard Dice+Focal loss. Gradient telemetry shows why — the boundary term contributes only ~0.05% of total gradient magnitude at its operating weight — and the same null result replicates under two corrected SDF formulations (patch-boundary-truncation-corrected and rotation-corrected, 10 retrain runs each) and on a genuinely held-out test set (15 evaluations, never touched during model/loss development).

## Key Results

**Validation set (N=5 seeds):**

| Configuration | Val Dice (mean ± SD) | Val HD95 mm (mean ± SD) |
|---|---|---|
| Baseline (Dice + Focal) | 0.7419 ± 0.0015 | 5.0825 ± 0.1455 |
| + Boundary Loss (fixed schedule) | 0.7413 ± 0.0020 | 5.0093 ± 0.1548 |
| + Boundary Loss (convergence-gated, adaptive) | 0.7417 ± 0.0037 | 5.1377 ± 0.1898 |

**Held-out test set (N=5 seeds, one-look evaluation, 434 patient-disjoint patches):**

| Configuration | Test Dice (mean ± SD) | Test HD95 mm (mean ± SD) | Test ASSD mm (mean ± SD) |
|---|---|---|---|
| Baseline (Dice + Focal) | 0.7303 ± 0.0011 | 6.5435 ± 0.2346 | 1.7334 ± 0.0750 |
| + Boundary Loss (fixed schedule) | 0.7266 ± 0.0048 | 6.7700 ± 0.3405 | 1.8890 ± 0.1502 |
| + Boundary Loss (convergence-gated, adaptive) | 0.7293 ± 0.0034 | 6.7614 ± 0.4151 | 1.8181 ± 0.1864 |

No pairwise comparison reaches statistical significance on either set (paired t-test and Wilcoxon signed-rank test; validation all p > 0.05, test set all p > 0.17). Overlap metrics (Dice, IoU) generalize well from validation to test; distance metrics (HD95, HD100, ASSD) show a consistent ~1.1–1.4 mm / ~0.2–0.4 mm gap across *all three* configurations equally, so it does not affect the comparison. See [`docs/statistical_analysis.md`](docs/statistical_analysis.md) for the full test matrix, including the SDF-approximation retrain comparisons.

**Gradient telemetry** (see [`docs/gradient_analysis.md`](docs/gradient_analysis.md)) shows the boundary loss term contributes ~0.94% of raw combined gradient magnitude, and ~0.05% at its trained effective weight, explaining the null result at the optimization level, not just the metric level.

**Architecture fidelity**: the reimplemented 3D U-Net has 19,073,665 parameters, closely matching the original paper's reported 19,069,955 (difference fully explained by output-channel count: 1 vs. the paper's 3).

## Repository Structure

```
├── src/
│   ├── train.py               # Main training loop (CLI: --seed, --boundary, --adaptive-boundary, --architecture, --interactive-save)
│   ├── config.py               # Hyperparameters and paths
│   ├── unet3d_paper.py         # Faithful 3D U-Net reimplementation (Çiçek et al., 2016)
│   ├── boundary_loss.py        # Kervadec et al. boundary loss + adaptive gating wrapper
│   ├── lidc_dataset.py         # PyTorch Dataset (patient-split-aware, SDF-cache-aware)
│   ├── build_hdf5.py           # LIDC-IDRI patch extraction pipeline (pylidc-based)
│   ├── evaluate_test_set.py    # One-look held-out test-set evaluation (all 15 checkpoints, full metric suite)
│   ├── analyze_test_results.py # Test-set pairwise stats, val-vs-test table, size-stratified table
│   ├── archiving.py            # promote_to_archive(): archives a run's checkpoints, per-sample metrics, qualitative examples
│   └── archive_run.py          # CLI wrapper around archiving.py
├── scripts/
│   ├── patient_split.py        # Patient-level, balance-verified train/val/test split
│   ├── precompute_sdf.py       # Signed distance field precomputation
│   ├── aggregate_results.py    # Multi-seed result aggregation
│   ├── statistical_test.py     # Paired significance testing (t-test + Wilcoxon)
│   └── grad_telemetry_check.py # Gradient-magnitude diagnostic
├── docs/
│   ├── methodology.md          # Full dataset construction, SDF-approximation verification & test-set methodology
│   ├── statistical_analysis.md # Full statistical tables: validation, SDF-corrected retrains, held-out test set
│   └── gradient_analysis.md
├── outputs/
│   ├── results/                 # Per-seed, per-config training result JSONs (35 files: 15 original +
│   │                            # 10 patch-boundary-truncation-corrected + 10 rotation-corrected retrains) —
│   │                            # raw evidence behind Key Results / statistical_analysis.md.
│   └── test_results/            # Held-out test-set evaluation outputs: 15 per-run JSONs, test_summary_all.json,
│                                # test_analysis_summary.json, test_analysis_report.txt
│                                # outputs/checkpoints/ and outputs/archive/ are gitignored (large).
├── train_val_test_split.json   # Patient-level train/val/test split (versioned for auditability)
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Methodology Highlights

- **Patient-level data splitting**: an initial patch-index-based split was found to permit train/val leakage (patients contributing multiple patches could appear in both partitions). Fixed via `GroupShuffleSplit` on patient ID, with a 30-seed search to select a split balanced on nodule-size distribution across partitions (verified via pairwise Kolmogorov–Smirnov tests, minimum p=0.553).
- **Dual-criterion checkpointing**: early experiments selected checkpoints by best Dice only, which was found to silently misrepresent the model's true best boundary-precision (HD95) epoch. Fixed by independently tracking and saving best-Dice and best-HD95 checkpoints.
- **Multi-seed statistical validation**: single-seed comparisons were found to be unreliable predictors of multi-seed outcomes (a promising single-seed adaptive-loss result did not replicate across a 5-seed batch). All reported comparisons use N=5 seeds with paired significance testing.
- **Gradient-level diagnosis**: rather than stopping at "no significant difference," gradient norm telemetry was used to directly measure why the boundary loss term's raw gradient magnitude is ~100x smaller than the regional loss terms', and further attenuated by its trained weighting coefficient.
- **SDF-approximation verification**: two known geometric approximations in the boundary-loss SDF pipeline (64³-patch-boundary truncation, affecting 11.5% of nodules; bilinear-interpolation-under-rotation error) were each quantified directly and then stress-tested by re-training both boundary-loss configurations on a corrected SDF across all 5 seeds. Neither correction produced a significant change from the original results (all p > 0.11), ruling out both as contributing factors to the null result.
- **Held-out test-set confirmation**: after both SDF threads and all architecture/loss decisions were closed, all 15 checkpoints were evaluated exactly once on a genuinely untouched 434-patch test partition under strict one-look discipline. The null result replicated cleanly (all p > 0.17), and no post-hoc changes were made based on the test-set numbers.

Full methodology, including the dataset construction, integrity audit, and SDF-verification/test-set protocols, is documented in [`docs/methodology.md`](docs/methodology.md).

## Reproducing Results

**Data availability**: this repository does not redistribute LIDC-IDRI data. Raw DICOM scans must be downloaded separately from [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/collection/lidc-idri/) and made queryable via [`pylidc`](https://pylidc.github.io/) (requires a local `pylidc` config pointing at your downloaded DICOM directory) before `build_hdf5.py` can run.

```bash
pip install -r requirements.txt

# Build the dataset (requires local pylidc + LIDC-IDRI DICOM access — see Data availability above)
python src/build_hdf5.py
python scripts/patient_split.py
python scripts/precompute_sdf.py

# Train (single run)
python src/train.py --seed 42                              # baseline
python src/train.py --seed 42 --boundary                    # fixed-schedule boundary loss
python src/train.py --seed 42 --boundary --adaptive-boundary  # adaptive gated boundary loss

# Aggregate and test across seeds (validation set)
python scripts/aggregate_results.py
python scripts/statistical_test.py

# Evaluate on the held-out test set (run only after all model/loss decisions are final —
# see the one-look discipline note in docs/methodology.md, Section 4.6)
python src/evaluate_test_set.py
python src/analyze_test_results.py
```

### Archiving a run (optional)

To keep a full record of a specific run (checkpoints, per-sample validation metrics, and qualitative best/median/worst examples) beyond the raw JSON summary in `outputs/results/`:

```bash
# Interactively, at the end of a single foreground run:
python src/train.py --seed 42 --interactive-save

# Or after the fact, for any already-completed run:
python src/archive_run.py --run <experiment_name> --name <archive_name> --notes "why this run is worth keeping"
```

`--interactive-save` is intended for single foreground runs only, it is not safe to use with batched or background-launched training.

## Limitations & Future Work

- The null result on boundary-loss weighting is specific to the tested weight ceiling (λ_max = 0.05) and architecture. Gradient-magnitude-aware dynamic balancing (e.g., GradNorm) was not evaluated but is a natural next step given the diagnosed gradient imbalance.
- Distance-based metrics (HD95, HD100, ASSD) show a consistent validation-to-test generalization gap (~1.1–1.4 mm for HD95/HD100) that overlap metrics (Dice, IoU) do not. This gap is uniform across all three configurations and does not change the paper's central comparison, but is a property of this model/dataset combination worth noting for anyone relying on validation-set distance metrics alone.
- Evaluation is limited to a single architecture family and a single dataset (LIDC-IDRI); extending the comparison to an attention-augmented or transformer-hybrid U-Net variant, or to other pulmonary-nodule datasets, is a potential direction.

## Citation

If you use this codebase, please cite the original 3D U-Net paper this reimplementation is based on:

```bibtex
@inproceedings{cicek20163d,
  title={3D U-Net: learning dense volumetric segmentation from sparse annotation},
  author={{\c{C}}i{\c{c}}ek, {\"O}zg{\"u}n and Abdulkadir, Ahmed and Lienkamp, Soeren S and Brox, Thomas and Ronneberger, Olaf},
  booktitle={International conference on medical image computing and computer-assisted intervention},
  pages={424--432},
  year={2016},
  organization={Springer}
}
```

## License
Released under the MIT License
