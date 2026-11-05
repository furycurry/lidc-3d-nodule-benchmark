# Boundary Loss for 3D Pulmonary Nodule Segmentation on LIDC-IDRI: A Multi-Seed Benchmark and Audit

A controlled, multi-seed study of whether auxiliary signed-distance-field (SDF) boundary supervision improves 3D pulmonary nodule segmentation over a regional Dice + Focal loss. Two architectures are compared on patient-disjoint LIDC-IDRI data: a faithful reimplementation of the original 3D U-Net ([Çiçek et al., 2016](https://arxiv.org/abs/1606.06650)) and UNETR ([Hatamizadeh et al., 2022](https://arxiv.org/abs/2103.10504)), a Vision Transformer hybrid. Every comparison uses N = 5 seeds with paired significance testing and a one-look held-out test evaluation.

<!-- TODO: add arXiv link and DOI badge for the manuscript (docs/report.tex) once submitted. -->

## Summary

The headline result is a **null result that holds on both architectures**: boundary-aware loss terms (the Kervadec et al., 2019 formulation, in a fixed-schedule variant and in a convergence-gated adaptive variant) give no statistically significant improvement over Dice + Focal on overlap metrics (Dice, IoU) or surface-distance metrics (HD95, ASSD, NSD at 1 mm and 2 mm), on validation or on the full held-out test set. The only detectable effect is a small shift (+0.35 Dice points) in one UNETR volume subgroup (medium nodules), which survives multiple-comparison correction only under a narrow definition of the test family.

Gradient telemetry explains why. The boundary term's raw gradient norm is about two orders of magnitude below the regional term's, and at the operating weight (λ = 0.05) it contributes roughly 0.05% of the effective update on the CNN and 0.03% on UNETR. A tenfold weight increase (λ = 0.50) did not produce surface refinement. The null result also survives two independent SDF-approximation audits (patch-boundary truncation and rotation interpolation, 10 retrains each).

Secondary findings:

- The pure ConvNet decisively outperforms UNETR on every metric (pooled test Dice 0.7287 vs. 0.6462, HD95 6.69 vs. 9.27 mm, p < 0.001, Cohen's d = 16.3 on Dice).
- UNETR is weakest on subtle ground-glass nodules (Dice 0.5349 vs. 0.6670 for the CNN), and a 100-epoch probe did not close the gap.
- Study design controls for patient-level leakage, single-seed instability, preprocessing confounds, and metric-selection bias in checkpointing.

## Key Results

**Validation set, UNet3DPaper (N = 5 seeds):**

| Configuration | Val Dice (mean ± SD) | Val HD95 mm (mean ± SD) |
|---|---|---|
| Baseline (Dice + Focal) | 0.7419 ± 0.0015 | 5.0825 ± 0.1455 |
| + Boundary loss (fixed schedule) | 0.7413 ± 0.0020 | 5.0093 ± 0.1548 |
| + Boundary loss (convergence-gated, adaptive) | 0.7417 ± 0.0037 | 5.1377 ± 0.1898 |

**Held-out test set, UNet3DPaper (N = 5 seeds, one-look, 434 patient-disjoint patches):**

| Configuration | Test Dice | Test HD95 mm | Test ASSD mm |
|---|---|---|---|
| Baseline (Dice + Focal) | 0.7303 ± 0.0011 | 6.5435 ± 0.2346 | 1.7334 ± 0.0750 |
| + Boundary loss (fixed schedule) | 0.7266 ± 0.0048 | 6.7700 ± 0.3405 | 1.8890 ± 0.1502 |
| + Boundary loss (adaptive) | 0.7293 ± 0.0034 | 6.7614 ± 0.4151 | 1.8181 ± 0.1864 |

**Held-out test set, UNETR (N = 5 seeds, same protocol):**

| Configuration | Test Dice | Test HD95 mm | Test ASSD mm |
|---|---|---|---|
| Baseline (Dice + Focal) | 0.6439 ± 0.0032 | 9.3519 ± 0.1500 | 3.4988 ± 0.1640 |
| + Boundary loss (fixed schedule) | 0.6470 ± 0.0015 | 9.2651 ± 0.1168 | 3.4395 ± 0.0376 |
| + Boundary loss (adaptive) | 0.6478 ± 0.0022 | 9.1935 ± 0.1308 | 3.4480 ± 0.1153 |

For the fixed schedule, the full-cohort p-values against baseline are 0.0549 (Dice), 0.4678 (HD95), 0.5693 (ASSD), 0.0512 (NSD@1mm), and 0.2226 (NSD@2mm), all above 0.05. Only one of five UNETR adaptive runs opened the gate, so the UNETR adaptive configuration is largely a second baseline. Full tables are in `outputs/test_results/`.

**Architecture comparison (pooled test, N = 15 per architecture):**

| Metric | UNet3DPaper (19.07M params) | UNETR (92.69M params) |
|---|---|---|
| Dice | 0.7287 ± 0.0038 | 0.6462 ± 0.0029 |
| HD95 (mm) | 6.6916 ± 0.3541 | 9.2702 ± 0.1482 |
| IoU | 0.6503 ± 0.0052 | 0.5399 ± 0.0087 |
| ASSD (mm) | 1.8135 ± 0.1582 | 3.4621 ± 0.1206 |

No full-cohort boundary-loss comparison reaches significance (UNet3DPaper validation: all p > 0.05; UNETR validation: one nominal t-test p = 0.027 for best-epoch Dice, baseline vs. fixed, with Wilcoxon p = 0.0625 and no effect on the test set; UNet3DPaper test: all p > 0.17; UNETR test: all p ≥ 0.051, the closest being NSD@1mm at 0.0512). Overlap metrics generalize well from validation to test on the CNN, while distance metrics degrade consistently (about 1.1–1.4 mm for HD95) and equally across all three configurations, so the gap does not affect the comparison. UNETR shows a larger gap on both. See [`docs/statistical_analysis.md`](docs/statistical_analysis.md).

**Gradient telemetry** ([`docs/gradient_analysis.md`](docs/gradient_analysis.md)):

| Architecture | Boundary share of combined gradient norm (weight 1.0) | Effective share (λ = 0.05) |
|---|---|---|
| UNet3DPaper | ≈ 0.94% | ≈ 0.05% |
| UNETR | 0.64% | 0.03% |

**Architecture fidelity:** the reimplemented 3D U-Net has 19,073,665 parameters, closely matching the original paper's 19,069,955 (the difference is the output channel count, 1 vs. 3). The UNETR wrapper has 92,685,521 parameters, against 92.58M reported for the original.

## Repository Structure

```
├── train.py                  # Training CLI (--architecture, --boundary, --adaptive-boundary, --seed, ...)
├── config.py                 # Hyperparameters and paths
├── lidc_dataset.py           # PyTorch Dataset: patient-split-aware, SDF-cache-aware, 3D augmentation
├── boundary_loss.py          # Kervadec boundary loss + adaptive (PCG-BW) gating wrapper
├── unet3d_paper.py           # Faithful 3D U-Net reimplementation (Çiçek et al., 2016)
├── unetr_paper.py            # UNETR (MONAI-based wrapper)
├── evaluate_test_set.py      # One-look held-out test evaluation (30 primary checkpoints)
├── compute_test_nsd.py       # Normalized surface distance at 1 mm / 2 mm tolerances
├── analyze_test_result.py    # Test-set pairwise statistics, val-vs-test and size-stratified tables
├── analyze_clinical_stratification.py  # Clinical-attribute strata (texture, subtlety, spiculation, malignancy)
├── analyze_adaptive_gate.py  # Per-seed gate-activation analysis from the per-epoch history
├── aggregate_results.py      # Validation aggregates for one architecture (--architecture)
├── statistical_test.py       # Paired t-test / Wilcoxon across matched seeds (--architecture)
├── results_io.py             # Shared loader that selects the main-matrix result files
├── grad_telemetry_check.py   # Per-term gradient-norm diagnostic
├── compare_corrected_sdf.py  # Original vs. corrected-SDF retrains (SDF audits)
├── quantify_rotation_error.py# Rotation-interpolation error of the cached SDF
├── audit_ground_truth.py, per_seed_values.py, inspect_dataset.py, grad_inspector.py, make_figures.py
├── archiving.py / archive_run.py   # Archive a run's checkpoints, per-sample metrics, examples
├── run_multiseed.sh          # Full 15-run matrix for one architecture, sequential
├── run_clean_matrix.sh       # Same matrix, parallel across GPUs
├── data_prep/
│   ├── build_hdf5.py         # LIDC-IDRI patch extraction (needs pylidc + DICOM access)
│   ├── patient_split.py      # Patient-disjoint split with the 30-seed KS balance search
│   ├── precompute_sdf.py     # Signed distance field cache
│   └── generate_corrected_sdf.py  # Wide-mask SDFs for the truncation audit
├── tools/                    # Regenerate every reported number from the raw JSONs
│   ├── make_tables_validation.py, make_tables_test.py, make_tables_clinical.py
│   └── inspect_schema.py, inspect_test_schema.py
├── train_val_test_split.json       # Patient-disjoint split (seed 5, KS p-values)
├── patch_patient_mapping.json      # Patch key -> patient ID provenance
├── nodule_clinical_metadata.json   # Radiologist ratings for subgroup analysis (not model inputs)
├── docs/
│   ├── methodology.md, statistical_analysis.md, gradient_analysis.md
│   └── report.tex            # Manuscript source
├── outputs/
│   ├── results/              # 52 training result JSONs (both architectures, SDF-audit retrains, probes)
│   ├── test_results/         # Held-out test JSONs, summaries and analysis reports
│   ├── generated/            # Tables regenerated by tools/ from the JSONs above
│   └── checkpoints/          # checkpoint_checksums.sha256 tracked; weights are not (see Artifacts)
├── requirements.txt
└── LICENSE
```

<!-- TODO(verify): confirm data_prep/ contains the four scripts listed after the flatten step. -->

## Methodology Highlights

- **Patient-level splitting.** An initial patch-index split permitted train/validation leakage. It was replaced by `GroupShuffleSplit` on patient ID (611 / 132 / 132 patients; 1,833 / 384 / 434 patches), with a 30-seed search for a split balanced on nodule size (pairwise Kolmogorov–Smirnov, minimum p = 0.553).
- **Dual-criterion checkpointing.** Best-Dice and best-HD95 checkpoints are saved independently, because selecting on Dice alone misreported boundary precision in early experiments.
- **Multi-seed paired testing.** A promising single-seed adaptive result did not replicate across five seeds. Every reported comparison uses seeds 42, 123, 456, 789, and 2024 with paired tests. With N = 5 the smallest possible Wilcoxon p is 0.0625.
- **Gradient-level diagnosis.** Gradient norms of the regional and boundary terms are measured separately to explain the null at the optimization level.
- **SDF-approximation audits.** Two approximations in the cached SDF were quantified and then stress-tested by retraining both boundary configurations on corrected SDFs for all five seeds. Truncation at the 64³ patch boundary affects 11.5% of nodules. Bilinear interpolation of the SDF under rotation causes a near-boundary error of 0.24 mm. Neither correction improved on the original runs (last-10-epoch aggregate: all p ≥ 0.11; on peak validation Dice the rotation-corrected fixed schedule is nominally slightly *worse*, p = 0.020 uncorrected).
- **One-look test protocol.** After all architecture, loss, and audit decisions were closed, all 30 best-Dice checkpoints (15 per architecture) were evaluated once on an untouched test partition. No post-hoc changes were made from the test numbers.
- **Matched-recipe architecture comparison.** UNETR was trained with exactly the CNN's recipe for a same-protocol comparison. A 100-epoch probe indicated the deficit is structural and not a training-length artifact.

Full details are in [`docs/methodology.md`](docs/methodology.md).

## Reproducing Results

**Data availability.** This repository does not redistribute LIDC-IDRI. Download the scans from [TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/) and make them queryable with [`pylidc`](https://pylidc.github.io/) (requires a local `pylidc` config pointing at the DICOM directory) before the dataset can be built.

**Artifacts.** The extracted patch file (`lidc_patches_int16.h5`), the SDF caches (`lidc_sdf_cache.h5`, `lidc_corrected_sdf_64.h5`), the wide masks used in the audits (`lidc_wide_masks_and_provenance.h5`), and the 32 trained checkpoints are too large for GitHub. They are verified by SHA-256 and hosted at:

<!-- TODO: replace with the Zenodo / Hugging Face URL once uploaded. -->
`[artifact URL to be added]`

```bash
pip install -r requirements.txt   # includes einops (required by UNETR)

# 1. Build the dataset (needs pylidc + LIDC-IDRI DICOM access)
python data_prep/build_hdf5.py --output lidc_patches_int16.h5
python data_prep/precompute_sdf.py
# Verify the published split regenerates exactly (seed 5, 1833 / 384 / 434 patches):
python data_prep/patient_split.py --h5 lidc_patches_int16.h5 \
    --mapping patch_patient_mapping.json --out split_regenerated.json \
    --verify train_val_test_split.json

# 2. Train one run (architecture: unet3d_paper or unetr)
python train.py --seed 42 --architecture unet3d_paper                                  # baseline
python train.py --seed 42 --architecture unet3d_paper --boundary                       # fixed schedule
python train.py --seed 42 --architecture unet3d_paper --boundary --adaptive-boundary   # adaptive gated

# ... or the whole 15-run matrix for one architecture
bash run_multiseed.sh unetr              # sequential
bash run_clean_matrix.sh unetr 8         # parallel across 8 GPUs

# 3. Validation aggregates and paired tests (one architecture at a time)
python aggregate_results.py --architecture unet3d_paper
python statistical_test.py  --architecture unet3d_paper

# 4. Gradient telemetry and gate analysis
python grad_telemetry_check.py --architecture unet3d_paper --seed 42
python analyze_adaptive_gate.py

# 5. Held-out test evaluation: run only after all model/loss decisions are final
#    (one-look discipline, docs/methodology.md Section 4.5)
python evaluate_test_set.py
python compute_test_nsd.py
python analyze_test_result.py
python analyze_clinical_stratification.py

# 6. SDF audits
python compare_corrected_sdf.py --results-dir outputs/results
python quantify_rotation_error.py

# 7. Regenerate every reported number from the raw JSONs
python tools/make_tables_validation.py   # -> outputs/generated/validation_tables.md
python tools/make_tables_test.py         # -> outputs/generated/test_tables.md
python tools/make_tables_clinical.py     # -> outputs/generated/clinical_tables.md
```

**SDF audits.** The audit retrains reuse `train.py` with a corrected SDF source and a result-file tag, so they never overwrite the original runs:

```bash
# Corrected (non-truncated) SDFs from the 128^3 wide masks
python data_prep/generate_corrected_sdf.py --wide-h5 lidc_wide_masks_and_provenance.h5 --out-h5 lidc_corrected_sdf_64.h5

# Truncation audit: corrected cached SDF (rotation approximation unchanged)
python train.py --seed 42 --architecture unet3d_paper --boundary \
    --sdf-path lidc_corrected_sdf_64.h5 --experiment-tag correctedSDF
# add --adaptive-boundary for the adaptive variant

# Exact-SDF audit: wide masks rotated with the image, SDF recomputed per sample
# (removes truncation and interpolation error together)
python train.py --seed 42 --architecture unet3d_paper --boundary \
    --sdf-path lidc_corrected_sdf_64.h5 --wide-mask-path lidc_wide_masks_and_provenance.h5 \
    --experiment-tag rotationCorrected
```

<!-- TODO(verify): --wide-mask-path is added to train.py in a release patch; confirm the rotationCorrected result JSONs were produced with the same loader code path. -->

Rerunning training reproduces results only approximately, because GPU operations are not bit-deterministic. The result JSONs in `outputs/results/` are the reference values behind every table in the paper.

### Archiving a run (optional)

To keep a full record of a run (checkpoints, per-sample validation metrics, qualitative best/median/worst examples) beyond the JSON summary:

```bash
# At the end of a single foreground run:
python train.py --seed 42 --interactive-save

# Or afterwards, for any completed run:
python archive_run.py --run <experiment_name> --name <archive_name> --notes "why this run is worth keeping"
```

`--interactive-save` is for single foreground runs only and is not safe with batched or background-launched training.

## Limitations & Future Work

- **Scope of the null.** It applies to λ_max = 0.05, the SDF formulation of Kervadec et al., and the two schedules tested, plus one higher-weight probe (λ = 0.50). Gradient-magnitude-aware balancing (e.g. GradNorm) and boundary loss inside self-configuring frameworks such as nnU-Net were not evaluated and are the natural next steps.
- **Adaptive gate on UNETR.** The gate thresholds were developed on the CNN, and only one of five UNETR adaptive seeds opened the gate. The UNETR fixed-schedule comparison is the informative one.
- **Audits.** The SDF retrain audits were run for UNet3DPaper only. Gradient telemetry is one snapshot per architecture, not a trajectory.
- **Validation-to-test gap.** Distance metrics degrade from validation to test (about 1.1–1.4 mm HD95 on the CNN, more on UNETR) where overlap metrics do not. The gap is uniform across configurations and does not change the central comparison, but anyone relying on validation distance metrics alone should account for it.
- **Task definition.** Every patch is centered on an annotated nodule and there are no background-only patches, so this is segmentation of a known nodule, not detection. The 50% radiologist-consensus threshold was not varied.
- **UNETR.** One configuration (64 tokens per 64³ patch, trained from scratch with the CNN recipe) was tested. Explanations for its deficit are hypotheses and were not isolated by ablation.
- **Subgroup analyses are exploratory.** The one UNETR medium-nodule shift (+0.35 Dice points, p = 0.0017) survives Bonferroni correction only if the test family is defined as the 28 comparisons reported. The clinical-attribute strata (texture, subtlety, spiculation, malignancy) were not corrected beyond that family.
- **Data.** One dataset (LIDC-IDRI), one fixed patient-level split, five seeds. No cross-validation or external validation.

## Citation

If you use this codebase, please cite the original 3D U-Net and UNETR papers:

```bibtex
@inproceedings{cicek20163d,
  title={3D U-Net: learning dense volumetric segmentation from sparse annotation},
  author={{\c{C}}i{\c{c}}ek, {\"O}zg{\"u}n and Abdulkadir, Ahmed and Lienkamp, Soeren S and Brox, Thomas and Ronneberger, Olaf},
  booktitle={International conference on medical image computing and computer-assisted intervention},
  pages={424--432},
  year={2016},
  organization={Springer}
}

@inproceedings{hatamizadeh2022unetr,
  title={UNETR: Transformers for 3D medical image segmentation},
  author={Hatamizadeh, Ali and Tang, Yucheng and Nath, Vishwesh and Yang, Dong and Myronenko, Andriy and Landman, Bennett and Roth, Holger R and Xu, Daguang},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={574--584},
  year={2022}
}
```

## License
Released under the MIT License.
