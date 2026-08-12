# 3D Pulmonary Nodule Segmentation on LIDC-IDRI: A Rigorous Multi-Seed Benchmark

A faithful reimplementation of the original 3D U-Net architecture ([Çiçek et al., 2016](https://arxiv.org/abs/1606.06650)), evaluated on patient-disjoint LIDC-IDRI CT data, with a statistically rigorous (N=5 seeds, paired significance testing) empirical study of fixed-schedule vs. a novel convergence-gated adaptive variant.

## Summary

This repository documents an end-to-end, leakage-free pipeline for 3D lung nodule segmentation, built with careful attention to common methodological pitfalls in medical imaging ML: patient-level data leakage, single-seed result instability, and metric-selection bias in checkpointing. The headline empirical finding is a **null result, diagnosed at the gradient level**: boundary-aware loss terms (Kervadec et al., 2019 formulation, both fixed-schedule and a novel convergence-gated adaptive variant) show no statistically significant improvement over standard Dice+Focal loss — and gradient telemetry shows why, with the boundary term contributing only ~0.05% of total gradient magnitude at its operating weight.

## Key Results

| Configuration | Val Dice (mean ± SD, N=5 seeds) | Val HD95 mm (mean ± SD, N=5 seeds) |
|---|---|---|
| Baseline (Dice + Focal) | 0.7419 ± 0.0015 | 5.0825 ± 0.1455 |
| + Boundary Loss (fixed schedule) | 0.7413 ± 0.0020 | 5.0093 ± 0.1548 |
| + Boundary Loss (convergence-gated, adaptive) | 0.7417 ± 0.0037 | 5.1377 ± 0.1898 |

No pairwise comparison reaches statistical significance (paired t-test and Wilcoxon signed-rank test, all p > 0.05). See [`docs/statistical_analysis.md`](docs/statistical_analysis.md) for the full test matrix.

**Gradient telemetry** (see [`docs/gradient_analysis.md`](docs/gradient_analysis.md)) shows the boundary loss term contributes ~0.94% of raw combined gradient magnitude, and ~0.05% at its trained effective weight — explaining the null result at the optimization level, not just the metric level.

**Architecture fidelity**: the reimplemented 3D U-Net has 19,073,665 parameters, closely matching the original paper's reported 19,069,955 (difference fully explained by output-channel count: 1 vs. the paper's 3).

## Repository Structure

```
├── src/
│   ├── train.py              # Main training loop (CLI: --seed, --boundary, --adaptive-boundary, --architecture, --interactive-save)
│   ├── config.py             # Hyperparameters and paths
│   ├── unet3d_paper.py       # Faithful 3D U-Net reimplementation (Çiçek et al., 2016)
│   ├── boundary_loss.py      # Kervadec et al. boundary loss + adaptive gating wrapper
│   ├── lidc_dataset.py       # PyTorch Dataset (patient-split-aware, SDF-cache-aware)
│   ├── build_hdf5.py         # LIDC-IDRI patch extraction pipeline (pylidc-based)
│   ├── archiving.py          # promote_to_archive(): archives a run's checkpoints, per-sample metrics, qualitative examples
│   └── archive_run.py        # CLI wrapper around archiving.py
├── scripts/
│   ├── patient_split.py      # Patient-level, balance-verified train/val/test split
│   ├── precompute_sdf.py     # Signed distance field precomputation
│   ├── aggregate_results.py  # Multi-seed result aggregation
│   ├── statistical_test.py   # Paired significance testing (t-test + Wilcoxon)
│   └── grad_telemetry_check.py  # Gradient-magnitude diagnostic
├── docs/
│   ├── methodology.md        # Full dataset construction & verification methodology
│   ├── statistical_analysis.md
│   └── gradient_analysis.md
├── outputs/
│   └── results/               # Per-seed, per-config training result JSONs (15 files) —
│                               # raw evidence behind Key Results / statistical_analysis.md.
│                               # outputs/checkpoints/ and outputs/archive/ are gitignored (large).
├── train_val_test_split.json  # Patient-level train/val/test split (versioned for auditability)
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Methodology Highlights

- **Patient-level data splitting**: an initial patch-index-based split was found to permit train/val leakage (patients contributing multiple patches could appear in both partitions). Fixed via `GroupShuffleSplit` on patient ID, with a 30-seed search to select a split balanced on nodule-size distribution across partitions (verified via pairwise Kolmogorov-Smirnov tests, minimum p=0.553).
- **Dual-criterion checkpointing**: early experiments selected checkpoints by best Dice only, which was found to silently misrepresent the model's true best boundary-precision (HD95) epoch. Fixed by independently tracking and saving best-Dice and best-HD95 checkpoints.
- **Multi-seed statistical validation**: single-seed comparisons were found to be unreliable predictors of multi-seed outcomes (a promising single-seed adaptive-loss result did not replicate across a 5-seed batch). All reported comparisons use N=5 seeds with paired significance testing.
- **Gradient-level diagnosis**: rather than stopping at "no significant difference," gradient norm telemetry was used to directly measure why — the boundary loss term's raw gradient magnitude is ~100x smaller than the regional loss terms', and further attenuated by its trained weighting coefficient.

Full methodology, including the dataset construction and integrity audit, is documented in [`docs/methodology.md`](docs/methodology.md).

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

# Aggregate and test across seeds
python scripts/aggregate_results.py
python scripts/statistical_test.py
```

### Archiving a run (optional)

To keep a full record of a specific run (checkpoints, per-sample validation metrics, and qualitative best/median/worst examples) beyond the raw JSON summary in `outputs/results/`:

```bash
# Interactively, at the end of a single foreground run:
python src/train.py --seed 42 --interactive-save

# Or after the fact, for any already-completed run:
python src/archive_run.py --run <experiment_name> --name <archive_name> --notes "why this run is worth keeping"
```

`--interactive-save` is intended for single foreground runs only — it is not safe to use with batched or background-launched training.

## Limitations & Future Work

- Signed distance fields are computed on cropped 64³ patches; nodules near patch boundaries may have truncated distance information, a potential source of boundary-loss noise not fully characterized here.
- The observed null result is specific to this loss weighting range (λ_max = 0.05) and architecture; gradient-magnitude-aware dynamic balancing (e.g., GradNorm) was not evaluated but is a natural next step given the diagnosed gradient imbalance.
- Evaluation is limited to a single architecture family; extending the comparison to an attention-augmented or transformer-hybrid U-Net variant is a potential direction.

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
