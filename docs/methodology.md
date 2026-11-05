# Methodology

This document describes the full pipeline for the 3D pulmonary nodule segmentation benchmark on LIDC-IDRI: dataset construction, the two architectures, the loss functions, the experimental and test protocols, the robustness audits, and the evaluation metrics. Results are in `statistical_analysis.md` and `gradient_analysis.md`.

---

## 1. Dataset Construction

### 1.1 Raw data
LIDC-IDRI comprises 1,018 thoracic CT scans, each with up to four radiologist annotations. Raw DICOM data (about 124 GB) was accessed through the `pylidc` library. Only the extracted patch dataset is needed for training and evaluation, and the raw DICOMs are not part of the released artifacts.

### 1.2 Patch extraction (`build_hdf5.py`)
For each scan, `pylidc`'s `cluster_annotations()` (default parameters) groups radiologist contours into nodule clusters. Scans with no clusters produce no patches. 875 patients contribute at least one patch. For each cluster:

- A **50% consensus binary mask** is built with `pylidc.utils.consensus(cluster, clevel=0.5)`: a voxel is foreground if at least half of the annotating radiologists included it. The threshold was fixed once and not ablated.
- The centroid is the center of the consensus mask's bounding box in the native voxel grid.
- A native-resolution window is cropped around the centroid. Its radius is spacing-aware per scan: `ceil((64/2 + 2) / orig_spacing)` per axis, with `orig_spacing = [slice_thickness, pixel_spacing, pixel_spacing]`, giving a 2 mm interpolation margin.
- The window is resampled to **isotropic 1.0 mm³** with `scipy.ndimage.zoom` (`order=1` for the image, `order=0` for the mask). LIDC-IDRI slice thickness varies from about 0.6 to 5 mm, so without resampling a fixed 64³ patch would cover different physical volumes in different scans.
- The result is center-cropped, or padded where the patch crosses the scan boundary, to **64×64×64** voxels. Image padding is −1000 HU (air) and mask padding is 0.
- Images are stored as **raw `int16` HU**, not pre-windowed. Windowing is lossy and irreversible, so it is deferred to load time. Masks are stored as `uint8`.

No minimum nodule-size filter exists. Every clustered annotation yields one patch. Every patch is centered on a confirmed nodule, and the dataset contains no background-only patches.

**Output:** `lidc_patches_int16.h5`, 2,651 patches, 924 MB (gzip level 4), keys `nodule_0000` to `nodule_2650`. Extraction finished with zero failed scans.

At load time, intensities are clamped to [−1000, 400] HU and scaled to [0, 1].

### 1.3 Integrity audit
An audit script checked 12 categories: key alignment, shape (all exactly 64³), dtype, HU range, NaN/Inf, mask binarity, empty masks, exact duplicates (content hash), patient-mapping completeness, and split leakage. Ten passed cleanly. Two flags were investigated and found benign:

- **HU range outliers:** 132 of 2,651 patches have a minimum of exactly −2048 or −3024 HU. These are out-of-field-of-view sentinel values in the source DICOMs. Load-time clamping to [−1000, 400] maps them to the same value as air. One patch (`nodule_2211`) reaches 4137 HU, consistent with dense calcification.
- **One exact duplicate pair:** `nodule_2400` and `nodule_2401` are byte-identical and come from the same patient (`LIDC-IDRI-0733`), so patient-level splitting keeps them in one partition.

A separate spatial check overlaid masks on images for 12 randomly sampled patches (14 to 7,808 foreground voxels) and found all 12 correctly aligned.

### 1.4 Patient identity
The HDF5 file stores no patient ID. `patch_patient_mapping.json` was reconstructed by replaying the scan and cluster iteration order of the extraction script. This relies on identical, deterministic ordering in both scripts. It was checked three ways: both scripts apply no filtering between cluster discovery and counter increment, the extraction had zero failed scans, and the mapping has exactly 2,651 entries. It was later confirmed by an independent re-derivation from a separate script run (`patch_patient_mapping_v2.json`), which matched on all 2,651 keys with no differences. Recording `patient_id` at extraction time would remove the ordering assumption and is the recommended approach for any rebuild.

### 1.5 Patient-level partitioning
An early implementation split patch indices at random (`torch.randperm`, 80/20). A patient contributes 1 to 23 patches (mean 3.03), so patches from one patient could fall in both training and validation. This leakage was found and fixed before any reported result.

The split now uses `sklearn.model_selection.GroupShuffleSplit` grouped by patient, in two stages: 15% of patients form the test set, then 17.65% of the remainder form the validation set (about 70/15/15). Thirty candidate seeds (0 to 29) were scored by pairwise two-sample Kolmogorov-Smirnov tests on nodule size (foreground voxel count) across the three partitions, and the seed with the highest minimum p-value was kept. **Seed 5** won, with p = 0.822 (train vs. val), 0.835 (train vs. test), and 0.553 (val vs. test).

| Partition | Patches | Patients |
|---|---|---|
| Train | 1,833 | 611 |
| Validation | 384 | 132 |
| Test | 434 | 132 |

Pairwise patient-set intersections are asserted empty at split creation and again on every `train.py` run. The split, seed, and KS p-values are stored in `train_val_test_split.json`. Stratification is on nodule size only.

### 1.6 Clinical metadata
Radiologist consensus ratings (spiculation, subtlety, texture) are kept outside the patch file in `nodule_clinical_metadata.json` and used only for post-hoc subgroup analysis. They are not model inputs. Test nodules are stratified by the mean rating over annotators: texture (non-solid ≤ 2.0, N = 51; part-solid 2.0–4.0, N = 38; solid ≥ 4.0, N = 345), subtlety (≤ 2.5, N = 64; 2.5–3.5, N = 114; > 3.5, N = 256), spiculation (smooth ≤ 2.0, N = 379; spiculated > 2.0, N = 55), and malignancy suspicion (≤ 2.5, N = 160; 2.5–3.5, N = 211; > 3.5, N = 63). Per-radiologist contours are not retained, so inter-observer agreement and other consensus thresholds cannot be examined without re-extraction.

### 1.7 Precomputed SDF cache
`lidc_sdf_cache.h5` holds one signed distance field per mask (`distance_transform_edt` on foreground and background, combined as negative inside, positive outside, float32). It is computed once by `precompute_sdf.py` and checked byte-identical to fresh computation on five patches. It lives in its own file so the checksum-verified patch dataset stays immutable.

Two approximations in this cache are audited in Section 6: truncation at the patch boundary and interpolation under rotation augmentation.

### 1.8 Integrity of artifacts
The patch dataset was SHA-256 checksummed before and after transfer. Model checkpoints are covered by `outputs/checkpoints/checkpoint_checksums.sha256`. Large binaries (dataset, SDF caches, checkpoints) are not stored in the repository.

---

## 2. Model Architectures

### 2.1 UNet3DPaper
A reimplementation of the 3D U-Net of Çiçek et al. (MICCAI 2016, arXiv:1606.06650), following the paper's Figure 2:

- Four resolution levels in the analysis and synthesis paths.
- Two 3×3×3 convolutions per level, each followed by `BatchNorm3d` and ReLU.
- 2×2×2 max pooling between encoder levels, 2×2×2 `ConvTranspose3d` up-convolutions in the decoder.
- Channel doubling before pooling: encoder in→32→64 | 64→64→128 | 128→128→256 | 256→256→512 (bottleneck); decoder 512+256→256→256 | 256+128→128→128 | 128+64→64→64.
- Skip connections by concatenation, final 1×1×1 convolution.

**19,073,665 parameters** against the paper's 19,069,955. The difference is the output channel count (1 here, 3 in the paper).

Three deliberate deviations:

1. **'Same' padding** instead of valid convolutions, since the pipeline needs a fixed 64³ → 64³ mapping.
2. **Single-channel sigmoid output** (binary task) instead of 3-class softmax.
3. **Standard BatchNorm running statistics at evaluation.** The paper's current-batch-statistics workaround was for batch size 1. This project trains at batch size 8.

An earlier MONAI generic U-Net (4.8M parameters) had very high seed variance (Dice SD about 0.078 over three seeds) and is retained only as historical context. Its single test evaluation (Dice 0.6209) is superseded and not reported as a result.

### 2.2 UNETR
UNETR (Hatamizadeh et al., WACV 2022, arXiv:2103.10504) is a pure Vision Transformer encoder on non-overlapping 3D patches with a convolutional decoder. `unetr_paper.py` wraps `monai.networks.nets.UNETR` rather than reimplementing it, because a library call is easier to verify than a hand-written ViT and decoder.

- Configuration: `feature_size=16`, `hidden_size=768`, `mlp_dim=3072`, `num_heads=12`, 12 layers, instance norm, residual blocks, convolutional patch projection (MONAI defaults, the original ViT-Base configuration).
- Skip connections are tapped at transformer layers 3, 6, 9, and 12.
- Input is 64³. MONAI fixes the internal patch size at 16³, so the encoder sees **64 tokens**, far fewer than the 96³ or 128³ crops UNETR was designed for.
- Output is raw logits with one channel, the same interface as UNet3DPaper.
- **92,685,521 parameters** against the paper's 92.58M, about 4.9× UNet3DPaper.
- The wrapper inspects MONAI's constructor signature to handle the `pos_embed` to `proj_type` rename across versions. `einops` is a required dependency.
- No pretrained weights are used, as in the original paper.

### 2.3 Training recipe parity
UNETR uses exactly the UNet3DPaper recipe (Section 4.4). The original paper trained for 20,000 iterations at batch size 6 (about 120,000 samples seen). Fifty epochs here is about 91,650 samples, roughly 87 epochs to match the iteration count. The identical recipe was chosen so the architectures are compared under one protocol, and the recipe was not changed in response to UNETR's validation numbers. A seed-42 smoke run plateaued with the learning rate fully decayed and train and validation loss converged, and a separate 100-epoch probe did not close the gap to the CNN.

---

## 3. Loss Functions

### 3.1 Baseline
```python
monai.losses.DiceFocalLoss(sigmoid=True, lambda_dice=1.0, lambda_focal=1.0)
```
Used in every configuration.

### 3.2 Boundary loss
Following Kervadec et al. (2019), with predicted probability map p(x) and ground-truth signed distance field φ(x):

$$\mathcal{L}_B = \frac{1}{N}\sum_x p(x)\,\phi(x)$$

An unclipped version collapsed training to empty predictions (validation Dice near 0), because large distant-background SDF values dominated the loss. SDF values are therefore **clipped to [−20, +20]** before averaging.

### 3.3 Weighting schedules

**Fixed schedule.** Weight 0 through epoch 16 (0-indexed), linear ramp over epochs 17 to 26, held at λ = 0.05 from epoch 26 to 50. These epochs come from the training logs. An earlier version of this document said 15 to 24, which was off by one.

**Adaptive schedule (PCG-BW).** A Schmitt-trigger gate on validation-loss convergence velocity:

- Validation loss is EMA-smoothed (α = 0.3, window k = 3). Velocity is the relative EMA decrease over the window.
- The gate opens when velocity stays below `GATE_TAU_ENTER = 0.005` for `GATE_PATIENCE = 3` consecutive epochs, and closes when velocity stays above `GATE_TAU_EXIT = 0.02` for 3 epochs. Between the thresholds it holds its state.
- Once open, velocity maps to a target weight through a sigmoid (γ = 0.001, λ_max = 0.05), and the weight is smoothed by an EMA (`WEIGHT_EMA_ALPHA = 0.3`).
- The weight is 0 for t < k and whenever velocity ≤ 0.
- The exponent is clipped to [−500, 500] to prevent overflow.

Without hysteresis the gate toggled on single-epoch noise. Hysteresis removed that, but behavior still varies by seed and architecture:

- **UNet3DPaper:** all five seeds opened the gate, first at epochs 33–42. Three (42, 789, 2024) did so once and stayed open. Two (123, 456) closed and reopened once. The realized weight never exceeded **0.0231**, under half of λ_max, and stayed below 0.01 for seeds 42 and 123.
- **UNETR:** only one seed (2024) opened the gate. The other four never did, so those runs used no boundary term (details in `gradient_analysis.md`).

### 3.4 Output-bias calibration
The final `Conv3d` bias is initialized to `-log((1 - 0.01) / 0.01)`, the focal-loss foreground prior of 1%. The same initialization is used in every configuration and architecture, so it does not differ between the compared conditions. It does differ from the earliest MONAI-era archives.

---

## 4. Experimental Protocol

### 4.1 Reproducibility
`torch.manual_seed`, `torch.cuda.manual_seed_all`, `np.random.seed`, and `random.seed` are set from a required `--seed` argument. GPU and cuDNN operations are not bit-deterministic, so a rerun with the same seed reproduces results only approximately. A second full execution of the 15-run matrix reproduced the reported means to within small floating-point differences.

### 4.2 Multi-seed standard
No conclusion rests on a single seed. All reported comparisons use N = 5 seeds (42, 123, 456, 789, 2024) with paired tests. Each architecture has 3 configurations × 5 seeds = 15 runs.

### 4.3 Dual-criterion checkpointing
Two checkpoints are saved independently per run: `best_dice_unet3d.pt` and `best_hd95_unet3d.pt`, each updated only when its own metric improves. Tracking a single Dice-best checkpoint made the reported HD95 the value at the Dice-best epoch, which on the early MONAI baseline differed greatly from the true best (28.23 mm vs. 8.13 mm in one run). Reported validation results use the best-Dice checkpoint, and test evaluation uses that checkpoint for every run, selected by validation only.

### 4.4 Training hyperparameters
- Batch size 8, 50 epochs, no early stopping.
- **AdamW**, weight decay 1e-5, initial learning rate 1e-4, `ReduceLROnPlateau` (factor 0.5, patience 5; final learning rate between 1.25e-5 and 1e-4 depending on the run).
- Augmentation (training only): `RandRotated` (±0.3 rad, about ±17.2° per axis, 50% probability), three independent `RandFlipd` (50% each), `RandGaussianNoised` (20% probability, std 0.1). Image and label use matched transforms, and the SDF is rotated with the same sampled angle. Validation and test use no augmentation.
- DataLoaders use `persistent_workers=True`.
- A SHA-256 fingerprint of the core source files (`get_code_version()`, first 12 hex characters) is stored with each run, so code provenance does not depend on git. All 15 original UNet3DPaper runs share `6d28de4f780d`, and all 15 UNETR runs share `925996db30b0` (the code after UNETR support was added). The SDF-audit retrains record `8ab2f50d9136` (truncation audit, which adds `--sdf-path`) and `aa564d7dc34d` (exact-SDF audit, which adds the wide-mask path). In both audit retrains validation uses the corrected cached SDF (`--sdf-path`), and in the exact-SDF retrain the training SDF is recomputed from the rotated wide mask. The two probe runs (the λ = 0.50 run and the 100-epoch UNETR baseline) record `05da343e13a1`.

### 4.5 Held-out test protocol (one look)
The 434-patch test partition was not used for any model, loss, or hyperparameter decision. After the architecture, losses, and SDF audits were final, all 15 best-Dice checkpoints of each architecture (30 total) were evaluated once with `evaluate_test_set.py` and the results are final. The evaluation covers all configurations and seeds rather than a selected best model, so the test set also checks whether the validation-time null result replicates. Further analyses on the frozen test outputs (`analyze_test_result.py`, `analyze_clinical_stratification.py`, `compute_test_nsd.py`, and `tools/make_tables_test.py`) compute new comparisons only and do not re-open model selection. The earlier test evaluation of the MONAI baseline is superseded.

### 4.6 Compute
Training used one NVIDIA RTX 4090 (24 GB) per run, inside the container `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel` (PyTorch 2.11, CUDA 12.8). Peak GPU memory was 9.76–9.78 GB (UNet3DPaper) and 4.40–4.42 GB (UNETR). The 15 UNet3DPaper runs took about 15.8 GPU-hours and the 15 UNETR runs about 7.8.

---

## 5. Evaluation Metrics

All metrics are computed per patch on the full 64³ prediction (threshold 0.5 on the sigmoid output) and then averaged.

- **Dice** (primary overlap metric) and **IoU**.
- **HD95** and **HD100** (millimeters). **Empty or skipped predictions are penalized** with the maximum patch diagonal (about 110.85 mm) rather than excluded. This removes an earlier optimistic bias but means HD95 values are not comparable with results from before that change.
- **ASSD**, the average symmetric surface distance.
- **NSD**, normalized surface distance at 1 mm and 2 mm tolerance, computed on the test set by `compute_test_nsd.py`.
- Precision, recall, specificity, and F1 via MONAI's `ConfusionMatrixMetric`, and raw TP/FP/FN/TN counts per epoch.
- **Size-stratified Dice** by ground-truth foreground voxel count. On the test set the tiers are fixed thresholds: small < 100 voxels (N = 148), medium 100–500 (N = 175), large > 500 (N = 111). Post-hoc subgroup Dice by clinical ratings (Section 1.6) is also reported.

### 5.1 Statistics
Paired t-test and Wilcoxon signed-rank test across matched seeds, α = 0.05, population SD (ddof = 0) for every ± value, paired Cohen's d for the architecture comparison. With five seeds the smallest possible Wilcoxon p is 0.0625. Test-set subgroup analyses use Bonferroni correction over two families: the 12 volume-stratified comparisons, and the 28 boundary-vs-baseline test comparisons that are reported (12 volume-stratified, 13 full-cohort metric comparisons, 3 margin-morphology comparisons). Counting all six metrics for both losses on both architectures gives a larger family of 39. See `statistical_analysis.md` for the full conventions.

### 5.2 Gradient telemetry
`grad_telemetry_check.py` loads a trained checkpoint, passes real validation batches through it, and uses `torch.autograd.grad` to measure the gradient norm of the regional (Dice + Focal) term and of the boundary term at raw weight 1.0. It reports the boundary term's raw and effective (at the trained weight) share. It does not modify `train.py`. It was run on one fixed-schedule checkpoint per architecture (seed 42).

---

## 6. Robustness Audits

Both approximations in the SDF cache (Section 1.7) were tested by quantifying the error and then retraining both boundary configurations across all five seeds with corrected SDFs (10 runs each, 20 in total), comparing against the original runs with the standard paired tests.

**Patch-boundary truncation.** 304 of 2,651 nodules (11.5%) touch or exceed the 64³ patch boundary. Near-boundary SDF error is 0.191 mm for these nodules against 0.023 mm for the rest. Corrected reference SDFs were computed from 128³ wide masks, center-cropped to 64³ and clipped at ±20 mm (`generate_corrected_sdf.py`).

**Rotation under augmentation.** Bilinear interpolation of the cached SDF under rotation violates the Eikonal condition. The measured near-boundary error is 0.240 mm (whole-patch 0.060 mm). The corrected runs rotate the 128³ wide mask together with the image and label and recompute the SDF from the rotated wide mask on the fly. This is an exact-SDF condition: it removes the truncation error as well as the interpolation error, so it does not isolate rotation alone. (The truncation audit changes only the cached SDF and leaves the rotation approximation in place.)

Neither correction improved on the original runs. Under the last-10-epoch aggregate all 12 comparisons have p ≥ 0.11. Under peak validation Dice, one comparison (fixed schedule, rotation-corrected SDF) is nominally *worse* than the original (−0.0025, p = 0.020 uncorrected, Wilcoxon 0.0625). Details are in `statistical_analysis.md`, Section 2. The audits were done for UNet3DPaper only.

The 128³ wide-mask extraction code has a known flaw: with anisotropic spacing its 128³ crop does not align exactly with the 64³ grid of the original patches (synthetic-sphere test: overlap 0.86 to 0.97 under anisotropy and 1.0 under isotropy). The audits were therefore designed to be self-referential, comparing SDFs derived from the same wide mask, and do not depend on bit-exact reproduction of the original grid.

---

## 7. Limitations and Scope

- **Task definition.** Every patch is centered on an annotated nodule and the dataset has no negatives. The task is segmenting a known nodule, which is easier than detection plus segmentation. Clinical claims should not be extended beyond that.
- **Consensus threshold.** The 50% threshold was not varied, and only the merged mask is stored.
- **Cohort coverage.** Scans with no annotated nodule contribute nothing, so the patch set is not characterized against the full 1,018-scan cohort or a clinical population.
- **One dataset, one split.** Robustness comes from five training seeds on a single patient-level split, not from k-fold cross-validation or an external dataset.
- **Resampling.** Linear and nearest-neighbor interpolation only, with no record of per-patch interpolation amount.
- **Boundary-loss scope.** The null result applies to λ_max = 0.05, the SDF formulation used, and the two schedules tested. One higher-weight probe (λ = 0.50) showed no change (`gradient_analysis.md`). Dynamic gradient balancing (for example GradNorm) is untested.
- **Gradient telemetry** is a single snapshot per architecture, not a trajectory.
- **Adaptive gate.** Its thresholds were developed on the CNN. On UNETR only one of five seeds opened the gate, so the UNETR adaptive configuration largely duplicates the baseline.
- **Audits.** The SDF truncation and rotation retrains were not repeated for UNETR. The argument that they are unnecessary rests on UNETR's even smaller boundary gradient share and is not an empirical check.
- **UNETR scope.** The comparison uses one UNETR configuration with 64 tokens per patch and the CNN's training recipe. Explanations for its deficit (data scarcity, missing convolutional priors, coarse tokenization) are hypotheses consistent with the data and were not isolated by ablation.
- **Not evaluated:** nnU-Net-style self-configuring pipelines and other architectures. Testing boundary loss inside such a framework remains open.
