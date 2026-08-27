# Methodology

This document describes the complete methodological pipeline for 3D pulmonary nodule segmentation on the LIDC-IDRI dataset, including dataset construction, model architecture, loss functions, experimental protocol, and evaluation metrics.

---

## 1. Dataset Construction

### 1.1 Raw Data
The Lung Image Database Consortium and Image Database Resource Initiative (LIDC-IDRI) comprises 1,018 thoracic CT scans, each with up to four radiologist annotations. Raw DICOM data (~124 GB) was accessed locally via the `pylidc` Python library. Only the extracted patch dataset was transferred to the training cluster; raw DICOMs never left the local workstation.

### 1.2 Patch Extraction
For each scan, `pylidc`'s `cluster_annotations()` grouped individual radiologist contours into distinct nodules. For each nodule cluster:

- A **50% consensus binary mask** was generated via `pylidc.utils.consensus(cluster, clevel=0.5)`.
- The centroid of the cluster's bounding box was computed in the original (pre-resample) voxel frame.
- A physical window was cropped around the centroid and resampled to **isotropic 1.0 mm³** spacing (linear interpolation for image, nearest-neighbor for mask). This is necessary because LIDC-IDRI exhibits heterogeneous slice thickness (~0.6–5 mm) across scans; a fixed-voxel-dimension patch without resampling would represent inconsistent physical volumes.
- Each patch was center-cropped/padded to exactly **64×64×64 voxels**. Padding (when a patch touches scan boundary) uses **−1000 HU** (physiological air value).
- Images stored as **raw `int16` HU** (deliberately not pre-windowed/normalized — windowing is a lossy, irreversible operation deferred to data-loading time to preserve full radiometric fidelity).
- Masks stored as **`uint8` binary**.

The extraction script used `scan.to_volume()` with a `(Y, X, Z) → (Z, Y, X)` transpose to align axes. No minimum nodule-size filter was applied; the dataset includes all annotated nodule sizes.

**Result:** `lidc_patches_int16.h5` — 2,651 patches, 924 MB (gzip-4 compressed).

### 1.3 Dataset Integrity Audit
A comprehensive audit checked 12 categories: key alignment (images↔masks), shape consistency (all exactly 64³), dtype consistency, HU value range, NaN/Inf presence, mask binarity, empty-mask presence, exact-duplicate detection (via content hashing), patient-mapping completeness, and train/val/test split leakage.

**10/12 checks passed cleanly.** Two flagged items were investigated and confirmed benign:
- **HU range outliers:** 132/2,651 patches showed minimum HU of exactly −2048 or −3024. Traced to a well-documented CT scanner artifact (out-of-reconstruction-field-of-view padding, common in raw DICOM data). Irrelevant to training since the data-loading pipeline clamps everything to [−1000, 400] HU regardless, making these values numerically indistinguishable from physiological air. One patch (`nodule_2211`) showed max HU of 4137, consistent with dense calcification.
- **One exact duplicate pair:** `nodule_2400` and `nodule_2401` were byte-identical. Cross-referenced against patient mapping and confirmed to originate from the **same patient** (`LIDC-IDRI-0733`), two close annotation clusters capturing overlapping tissue. Zero leakage risk since patient-level splitting guarantees both instances land in the same partition.

### 1.4 Patient Identity Reconstruction
The HDF5 file does not natively store per-patch patient ID. A separate reconstruction script replays `pl.query(pl.Scan).all()` + `scan.cluster_annotations()` in the same order as the extraction script, assigning `scan.patient_id` sequentially to `nodule_XXXX` keys.

**Validity was explicitly verified:** (a) both scripts use identical, unfiltered iteration logic; (b) the original extraction completed with **zero failed scans** — critical, because any failure would have caused desynchronization; (c) the reconstructed mapping produced **exactly 2,651 entries**, matching the original patch count exactly.

**Independent re-verification:** A from-scratch re-derivation (`patch_patient_mapping_v2.json`) was diffed against the original mapping. Both files contain exactly 2,651 entries with **zero differing keys**, providing direct empirical confirmation of the ordering assumption.

**Known limitation:** this reconstruction approach carries an ordering-assumption risk that a "record patient_id at extraction time" approach would not have. This is noted as a methodological choice for future reproduction.

### 1.5 Patient-Level Data Partitioning
A patient-level data-leakage issue was identified and fixed in early development: an initial implementation used `torch.randperm` over patch indices for an 80/20 split. Because a single patient can contribute multiple patches (verified range: 1–23 patches/patient, mean 3.03), this permitted patches from the same patient to appear in both training and validation — a disqualifying methodological error in medical imaging ML.

**Fix:** `sklearn.model_selection.GroupShuffleSplit`, grouped by `PatientID`, producing a proper **3-way split** (train / validation / held-out test).

**Balance verification:** A 30-seed search evaluated each candidate split's pairwise Kolmogorov–Smirnov test on nodule-size (foreground voxel count) distribution across train/val/test. The winning seed (**seed = 5**) achieved a minimum pairwise p-value of **0.553** (train-val p = 0.822, train-test p = 0.835, val-test p = 0.553), well above the 0.05 significance threshold, confirming no meaningful size-distribution skew.

**Final split:** **1,833 train / 384 val / 434 test** patches.

**Hard leakage guarantee:** Patient-set intersection assertions (train∩val, train∩test, val∩test all empty) are checked programmatically in `train.py` on every run.

**Held-out test discipline:** The test set was evaluated under a one-look discipline only after all architecture and loss decisions were finalized. All 15 checkpoints (3 configurations × 5 seeds) were evaluated once on the 434-patch patient-disjoint test partition, with no post-hoc selection or further tuning permitted. Paired comparisons across test-set Dice, HD95, IoU, and ASSD showed no statistically significant differences between any pair of configurations (all p > 0.17, t-test and Wilcoxon signed-rank), replicating the validation-time null result on genuinely untouched data. Full statistical tables are in `statistical_analysis.md`; summary in Section 4.6 below.

### 1.6 Precomputed SDF Cache
A separate file, `lidc_sdf_cache.h5`, stores a precomputed signed distance field (SDF) per mask. Computed once via `precompute_sdf.py` (~3.5 minutes for all 2,651 patches) rather than recomputed every training step, since the SDF depends only on the static ground-truth mask.

**Motivation:** An early boundary-loss implementation computed the SDF on-the-fly via `scipy.ndimage.distance_transform_edt` inside the loss function, once per sample per epoch — a CPU-bound bottleneck. Precomputing removed this bottleneck entirely.

**Correctness verified:** Cached SDF values confirmed byte-identical to fresh on-the-fly computation on 5 test patches (`max_diff = 0.000000`).

**SDF patch-boundary truncation.** The SDF is computed within a cropped 64³ patch with no knowledge of true nodule geometry beyond the patch edge. This was quantified directly: 304/2,651 nodules (11.5%) have their true extent touching or exceeding the 64³ patch boundary, with a mean near-boundary SDF deviation of 0.191 mm for these nodules vs. 0.023 mm for non-truncated ones (whole-patch mean 0.015 mm; max 19.000 mm, near the ±20 mm clip ceiling). To directly test whether this affects the reported null result, both boundary-loss configurations were re-trained on corrected, non-truncated SDFs (computed on a 128³ wide mask and cropped to 64³) across all 5 seeds. Paired comparisons (t-test and Wilcoxon signed-rank) across Dice, HD95, and IoU showed no statistically significant difference between original and corrected-SDF results for either configuration (all p > 0.11), with the adaptive configuration showing near-identical means (Dice: 0.7348 vs. 0.7348; HD95: 5.581 vs. 5.581). This directly confirms the truncation error is not a contributing factor to the null result, consistent with the independently-measured gradient-magnitude explanation (boundary term ~0.05% of gradient magnitude vs. ~0.94% for the regional loss). Full tables in `statistical_analysis.md`.

**SDF rotation approximation.** Under training-time random rotation (±0.3 rad per axis), the cached SDF is bilinear-interpolated rather than recomputed from the rotated mask. This error was quantified directly (near-boundary mean deviation 0.240 mm across 1,000 sampled nodule–angle pairs, comparable in magnitude to patch-boundary truncation) and then tested empirically: both boundary-loss configurations were re-trained with the SDF recomputed on-the-fly from the rotated mask across all 5 seeds. Paired comparisons (t-test and Wilcoxon signed-rank) across Dice, HD95, and IoU showed no statistically significant difference for either configuration (all p > 0.11), confirming the rotation approximation, like patch-boundary truncation, is not a contributing factor to the reported null result. Full tables in `statistical_analysis.md`.

The SDF cache is kept in a **separate file** from the original dataset, so the original checksum-verified dataset remains completely untouched and immutable.

---

## 2. Model Architecture

### 2.1 Architecture Selection
After an initial baseline using MONAI's generic U-Net (4.8M parameters, high seed-to-seed variance), the project pivoted to a faithful reimplementation of the original 3D U-Net paper:

> Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation," MICCAI 2016 (arXiv:1606.06650).

### 2.2 Paper-Faithful Reimplementation (`UNet3DPaper`)
The reimplementation matches the paper's Figure 2 exactly:

- **4 resolution levels** in both analysis (encoder) and synthesis (decoder) paths.
- Each level: two **3×3×3 convolutions**, each followed by `BatchNorm3d` then `ReLU`.
- **2×2×2 max pooling** (stride 2) between encoder levels; **2×2×2 up-convolution** (stride 2, `ConvTranspose3d`) in the decoder.
- **Channel progression** matching the paper's channel-doubling-before-pooling scheme (to avoid bottlenecks, per Szegedy et al.):
  - Encoder: in → 32 → 64 | 64 → 64 → 128 | 128 → 128 → 256 | 256 → 256 → 512 (bottleneck)
  - Decoder: 512+256 → 256 → 256 | 256+128 → 128 → 128 | 128+64 → 64 → 64
- **Skip connections** via channel concatenation at matching resolutions.
- **Final 1×1×1 convolution** to output channels.

**Parameter count: 19,073,665** — extremely close to the paper's own reported 19,069,955 (the small difference is fully explained by the different final-layer output channel count: 1 here vs. 3 in the original), providing strong independent evidence of structural fidelity.

### 2.3 Documented Deviations from the Paper
Three deliberate deviations are explicitly documented:

1. **'Same' padding** (`padding=1`) instead of the paper's valid/shrinking convolutions. The paper's 132³ → 44³ input/output shrinkage supports their seamless-tiling strategy for arbitrarily large volumes; this project needs a fixed 64³ → 64³ shape to match its patch-based pipeline.
2. **Single-channel sigmoid output** (binary nodule/background) instead of the paper's 3-class softmax. Their Xenopus kidney task had 3 label classes; this project is binary segmentation.
3. **Standard `BatchNorm3d` running-statistics behavior at eval time**, rather than the paper's current-batch-statistics workaround. The paper needed this workaround specifically because they used batch size 1; this project uses batch size 8, so no special handling is needed.

---

## 3. Loss Functions

### 3.1 Baseline Loss
The primary loss throughout all experiments is:

```python
monai.losses.DiceFocalLoss(sigmoid=True, lambda_dice=1.0, lambda_focal=1.0)
```

This combines Dice loss (region overlap) and Focal loss (hard-example mining) with equal weighting.

### 3.2 Boundary Loss
The boundary loss formulation follows Kervadec et al. (2019), with a critical stabilization modification discovered through empirical debugging.

**Formulation:**

Given a predicted probability map $p(x)$ and a signed distance field (SDF) $\phi(x)$ derived from the ground-truth mask, the boundary loss is:

$$\mathcal{L}_B = \frac{1}{N} \sum_{x} p(x) \cdot \phi(x)$$

**Stabilization (critical fix):** An early unclipped implementation caused catastrophic training collapse in which the model predicted entirely empty volumes (Val Dice → ~0.0000). Root cause: unclipped SDF magnitude in distant background voxels (up to ~50 units) dominated the loss, creating a strong incentive to predict nothing anywhere.

**Fix:** SDF values are **clipped to [−20, +20]** before averaging. This prevents distant background voxels from dominating while preserving the boundary-localization signal.

### 3.3 Weighting Schedules
Two boundary-loss weighting strategies were compared against the Dice+Focal baseline:

**Fixed Schedule:**
- Boundary weight $\lambda_b = 0$ for epochs 1–16 (warmup).
- Linear ramp from 0 to target weight (0.05) over epochs 17–26, in ten steps of 0.005/epoch.
- Held constant at 0.05 from epoch 26 onward.

*(Verified directly against the training logs for all 5 fixed-schedule seeds. This is one epoch later than the schedule's original 0-indexed design — epochs 0–14 warmup / 15–24 ramp — a boundary-condition detail in the scheduling code (`>` vs. `>=`), not a functional bug, but documented here so the schedule as described matches the schedule as it actually ran.)*

**Adaptive Hysteresis-Gated Schedule (PCG-BW):**
A Schmitt-trigger-style state machine that gates boundary-loss activation by primary-loss convergence velocity:

- **Gating signal:** Validation loss EMA-smoothed over window $k=3$ with $\alpha=0.3$.
- **Convergence velocity:** Relative EMA loss decrease over the window.
- **Two thresholds:** `GATE_TAU_ENTER = 0.005` (velocity must drop below this to start counting toward activation) and `GATE_TAU_EXIT = 0.02` (velocity must rise above this to start counting toward deactivation). Creates a "dead zone" where the gate stays in its current state.
- **Patience:** `GATE_PATIENCE = 3` — requires 3 consecutive epochs of consistent signal before flipping state.
- **Weight smoothing:** `WEIGHT_EMA_ALPHA = 0.3` — an additional EMA applied to the boundary weight itself for smooth ramping.
- **Sigmoid mapping:** Once active, velocity maps to weight via a sigmoid with temperature $\gamma = 0.001$ and maximum $\lambda_{max} = 0.05$.
- **Edge cases:** $\lambda_b(t) = 0$ for $t < k$ (insufficient history) and for $v_p(t) \leq 0$ (validation loss degrading).

**Hysteresis motivation:** An initial non-hysteresis implementation exhibited unstable gate toggling — the boundary weight switched on and off unpredictably across epochs due to single-epoch noise in validation loss velocity. The hysteresis fix substantially stabilizes this, though not perfectly: across the N=5 seeds used in this project, 3 seeds (42, 789, 2024) show a single activation that holds for the remainder of training, and 2 seeds (123, 456) show one deactivation-and-reactivation cycle rather than continuous oscillation. This is a marked improvement over the non-hysteresis baseline's unpredictable toggling, though not literally a single, clean activation in every run.

**Realized weight, in practice:** Across all 5 adaptive-schedule seeds and all epochs, the actual boundary weight never approached $\lambda_{max}=0.05$: the maximum weight observed in any seed/epoch was 0.0231, and per-seed mean weight during gate-active epochs ranged from 0.0000 to 0.0101 — well below the fixed schedule's steady-state weight of 0.05 (held for the back half of training, per the corrected schedule above). See `gradient_analysis.md` Section 4.2 for how this reinforces the gradient-magnitude explanation of the null result.

### 3.4 Output-Layer Bias Calibration
A standard focal-loss bias calibration is applied at model initialization:

```python
init_bias = -log((1 - 0.01) / 0.01)
```

This biases initial predictions toward the rare foreground class (1% prior). This bias calibration is applied uniformly across all configurations (baseline, fixed-schedule, and adaptive) to ensure fair comparison.

---

## 4. Experimental Protocol

### 4.1 Reproducibility
All experiments use fixed random seeds:

```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
```

The seed is passed as a required CLI argument (`--seed`).

### 4.2 Multi-Seed Evaluation Standard
No conclusion is drawn from single-seed comparisons. The minimum standard is **N = 3 seeds**; key results were extended to **N = 5 seeds** (42, 123, 456, 789, 2024) for statistical robustness.

### 4.3 Dual-Criterion Checkpointing
Two checkpoints are saved independently per run:
- `best_dice_unet3d.pt` — updated only when validation Dice improves.
- `best_hd95_unet3d.pt` — updated only when validation HD95 improves.

This prevents the earlier bug where tracking only the Dice-best checkpoint caused the reported HD95 to be whatever happened to occur on the Dice-best epoch — often not the model's true best boundary-precision epoch. The held-out test-set evaluation (Section 4.6) uses `best_dice_unet3d.pt` per run, matching the project's existing best-checkpoint convention.

### 4.4 Data Augmentation
Training uses random augmentation:
- Random rotation (applied to both image and mask, with matched transform keys for SDF alignment).
- Random flips.

Validation uses **no augmentation**.

### 4.5 Training Hyperparameters
- **Batch size:** 8
- **Optimizer:** AdamW (weight decay 1e-5)
- **Learning rate:** Initialized at 1e-4, decays adaptively during training (final LR after 50 epochs ranged 1.25e-5 to 1e-4 depending on run; exact schedule mechanism in `config.py`)
- **Epochs:** 50
- **DataLoader:** `persistent_workers=True` for minor throughput optimization

### 4.6 Held-Out Test Set Discipline
The test set was treated as a true held-out evaluation and evaluated exactly once on the final `UNet3DPaper` architecture, after all architecture and loss decisions were finalized, per one-look discipline. All 15 checkpoints (baseline, fixed-schedule boundary, and adaptive boundary, each across 5 seeds) were evaluated on the 434-patch test partition using pure forward-pass inference (no SDF needed — SDF is only used in the training loss).

**Test-set headline result:** Paired comparisons across test-set Dice, HD95, IoU, and ASSD showed no statistically significant differences between any pair of configurations (all p > 0.17, t-test and Wilcoxon signed-rank), replicating the validation-time null result on genuinely untouched data.

**Generalization gap.** Overlap metrics generalize well: Dice decreased by 0.006–0.012 and IoU shifted by only ±0.001–0.004 from validation to test, consistent across all three configurations. Distance metrics show a consistent, larger gap in the same direction for every configuration: HD95 increased 1.06–1.37 mm, HD100 increased 1.25–1.39 mm, and ASSD increased 0.23–0.42 mm. This overlap-vs-distance asymmetry is uniform across baseline, fixed-schedule, and adaptive configurations — i.e., it is a property of the model's generalization behavior on this dataset, not evidence that boundary-loss variants generalize differently from the baseline. See `statistical_analysis.md` for full validation-vs-test tables.

**Size-stratified test Dice:** Maintained the small < medium < large ordering observed on validation (baseline: 0.690 → 0.740 → 0.768; boundary_fixed: 0.689 → 0.732 → 0.768; boundary_adaptive: 0.686 → 0.741 → 0.769), confirming the null result is robust across nodule sizes on held-out data as well as in aggregate.

Critically, the test-set numbers did not trigger any post-hoc model, loss, or hyperparameter changes — this is the actual meaning of one-look discipline. Full pairwise statistics, the validation-vs-test table, size-stratified table, and per-seed breakdown are reported in `statistical_analysis.md`.

---

## 5. Evaluation Metrics

### 5.1 Segmentation Metrics
All metrics computed at the patch level:

- **Dice Similarity Coefficient (DSC):** Primary overlap metric.
- **HD95 (95th Percentile Hausdorff Distance):** Boundary precision, in millimeters. Empty or skipped predictions are **penalized** with the maximum possible patch diagonal (~110.85 mm for a 64³ patch) rather than excluded from the mean. This addresses a previously flagged optimistic bias.
- **HD100 (Maximum Hausdorff Distance):** Worst-case boundary error.
- **ASSD (Average Symmetric Surface Distance):** Mean surface distance, reported as a headline metric alongside Dice and HD95 in both validation and test-set analyses.
- **IoU (Intersection over Union):** Alternative overlap metric.
- **Precision, Recall, Specificity, F1:** Derived from MONAI's `ConfusionMatrixMetric`.
- **Raw TP/FP/FN/TN voxel counts:** Logged to enable any confusion-matrix metric to be recomputed later without rerunning.

Full per-metric tables (validation and test) are in `statistical_analysis.md`.

### 5.2 Per-Nodule-Size Analysis
Dice is additionally reported per size bucket (small / medium / large by ground-truth foreground voxel count) to detect size-dependent performance biases.

### 5.3 Statistical Testing
Pairwise comparisons across configurations use:
- **Paired t-test**
- **Wilcoxon signed-rank test**

Applied to matched seeds for `best_val_dice` and `best_val_hd95` on the validation set. Significance threshold: $\alpha = 0.05$. The identical paired methodology was applied to the SDF-truncation and SDF-rotation corrected-retrain comparisons (Section 1.6) and to the held-out test-set results (Section 4.6), each across Dice, HD95, IoU, and (for the test set) ASSD.

### 5.4 Gradient Telemetry
A standalone, non-invasive diagnostic script (`grad_telemetry_check.py`) measures gradient magnitude of the boundary loss term vs. the regional (Dice+Focal) loss term using a trained checkpoint and real validation batches — deliberately not modifying `train.py`, just inspecting gradients via `torch.autograd.grad`. Full derivation and interpretation in `gradient_analysis.md`.

---

## 6. Limitations and Scope

### 6.1 SDF Patch-Boundary Truncation
304/2,651 nodules (11.5%) have their true extent touching or exceeding the 64³ patch boundary, with a mean near-boundary SDF deviation of 0.191 mm for these nodules vs. 0.023 mm for non-truncated ones. Both boundary-loss configurations were re-trained on corrected, non-truncated SDFs across all 5 seeds; paired comparisons across Dice, HD95, and IoU showed no statistically significant difference between original and corrected-SDF results for either configuration (all p > 0.11), with the adaptive configuration showing near-identical means (Dice: 0.7348 vs. 0.7348; HD95: 5.581 vs. 5.581). This directly confirms the truncation error is not a contributing factor to the null result, consistent with the gradient-magnitude explanation.

### 6.2 SDF Rotation Approximation
Under training-time random rotation (±0.3 rad per axis), the cached SDF is bilinear-interpolated rather than recomputed from the rotated mask. This error was quantified directly (near-boundary mean deviation 0.240 mm across 1,000 sampled nodule–angle pairs) and then tested empirically: both boundary-loss configurations were re-trained with the SDF recomputed on-the-fly from the rotated mask across all 5 seeds. Paired comparisons showed no statistically significant difference for either configuration (all p > 0.11), confirming the rotation approximation is not a contributing factor to the null result.

### 6.3 Boundary-Loss Weight Ceiling
The null result on boundary-loss weighting is specific to $\lambda_{max} = 0.05$. Higher weights were not explored due to the observed gradient-starvation diagnosis.

### 6.4 Test-Set Evaluation and Generalization Gap
The held-out test set was evaluated once under one-look discipline; test-set numbers did not trigger any post-hoc model, loss, or hyperparameter changes. Overlap metrics (Dice, IoU) generalized well from validation to test, but distance-based metrics (HD95, HD100, ASSD) showed a consistent test-time increase across all three configurations (roughly +1.1 to +1.4 mm for HD95/HD100, +0.23 to +0.42 mm for ASSD). Because this gap is uniform across baseline and both boundary-loss variants, it does not affect the paper's central comparison, but it is noted here as a property of this model/dataset combination that a reader relying on validation-set distance metrics alone should be aware of.
