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
- **HU range outliers:** 132/2,651 patches showed minimum HU of exactly −2048 or −3024. Traced to a well-documented CT scanner artifact out-of-reconstruction-field-of-view padding, common in raw DICOM data. Irrelevant to training since the data-loading pipeline clamps everything to [−1000, 400] HU regardless, making these values numerically indistinguishable from physiological air. One patch (`nodule_2211`) showed max HU of 4137, consistent with dense calcification.
- **One exact duplicate pair:** `nodule_2400` and `nodule_2401` were byte-identical. Cross-referenced against patient mapping and confirmed to originate from the **same patient** (`LIDC-IDRI-0733`), two close annotation clusters capturing overlapping tissue. Zero leakage risk since patient-level splitting guarantees both instances land in the same partition.

### 1.4 Patient Identity Reconstruction
The HDF5 file does not natively store per-patch patient ID. A separate reconstruction script replays `pl.query(pl.Scan).all()` + `scan.cluster_annotations()` in the same order as the extraction script, assigning `scan.patient_id` sequentially to `nodule_XXXX` keys.

**Validity was explicitly verified:** (a) both scripts use identical, unfiltered iteration logic; (b) the original extraction completed with **zero failed scans** — critical, because any failure would have caused desynchronization; (c) the reconstructed mapping produced **exactly 2,651 entries**, matching the original patch count exactly.

**Known limitation:** this reconstruction approach carries an ordering-assumption risk that a "record patient_id at extraction time" approach would not have. This is noted as a methodological choice for future reproduction.

### 1.5 Patient-Level Data Partitioning
A patient-level data-leakage issue was identified and fixed in early development: an initial implementation used `torch.randperm` over patch indices for an 80/20 split. Because a single patient can contribute multiple patches (verified range: 1–23 patches/patient, mean 3.03), this permitted patches from the same patient to appear in both training and validation — a disqualifying methodological error in medical imaging ML.

**Fix:** `sklearn.model_selection.GroupShuffleSplit`, grouped by `PatientID`, producing a proper **3-way split** (train / validation / held-out test).

**Balance verification:** A 30-seed search evaluated each candidate split's pairwise Kolmogorov-Smirnov test on nodule-size (foreground voxel count) distribution across train/val/test. The winning seed (**seed = 5**) achieved a minimum pairwise p-value of **0.553** (train-val p = 0.822, train-test p = 0.835, val-test p = 0.553) which is well above the 0.05 significance threshold, confirming no meaningful size-distribution skew.

**Final split:** **1,833 train / 384 val / 434 test** patches.

**Hard leakage guarantee:** Patient-set intersection assertions (train∩val, train∩test, val∩test all empty) are checked programmatically in `train.py` on every run.

**Held-out test discipline:** The test set has been evaluated exactly once (on the earlier, now-superseded MONAI baseline architecture). It has **not yet been evaluated** on the final `UNet3DPaper` architecture; this evaluation is deliberately deferred until all architecture/loss decisions are finalized (one-look discipline).

### 1.6 Precomputed SDF Cache
A separate file, `lidc_sdf_cache.h5`, stores a precomputed signed distance field (SDF) per mask. Computed once via `precompute_sdf.py` (~3.5 minutes for all 2,651 patches) rather than recomputed every training step, since the SDF depends only on the static ground-truth mask.

**Motivation:** An early boundary-loss implementation computed the SDF on-the-fly via `scipy.ndimage.distance_transform_edt` inside the loss function, once per sample per epoch — a CPU-bound bottleneck. Precomputing removed this bottleneck entirely.

**Correctness verified:** Cached SDF values confirmed byte-identical to fresh on-the-fly computation on 5 test patches (`max_diff = 0.000000`).

**Documented caveat:** Under data augmentation (random rotation), the cached SDF is transformed via bilinear interpolation rather than exactly recomputed from the rotated mask — a deliberate speed/exactness tradeoff with low but nonzero geometric error. The validation set uses no augmentation, so its cached SDF is used exactly as computed, with zero approximation.

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

**Stabilization (critical fix):** An early unclipped implementation caused catastrophic training collapse which the model predicted entirely empty volumes (Val Dice → ~0.0000). Root cause: unclipped SDF magnitude in distant background voxels (up to ~50 units) dominated the loss, creating a strong incentive to predict nothing anywhere.

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
No conclusion is drawn from single-seed comparisons. The minimum standard is **N = 3 seeds**; key results were extended to **N = 5 seeds** (42, 123, 2024, 456, 789) for statistical robustness.

### 4.3 Dual-Criterion Checkpointing
Two checkpoints are saved independently per run:
- `best_dice_unet3d.pt` — updated only when validation Dice improves.
- `best_hd95_unet3d.pt` — updated only when validation HD95 improves.

This prevents the earlier bug where tracking only the Dice-best checkpoint caused the reported HD95 to be whatever happened to occur on the Dice-best epoch — often not the model's true best boundary-precision epoch.

### 4.4 Data Augmentation
Training uses random augmentation:
- Random rotation (applied to both image and mask, with matched transform keys for SDF alignment).
- Random flips.

Validation uses **no augmentation**.

### 4.5 Training Hyperparameters
- **Batch size:** 8
- **Optimizer:** Adam
- **Learning rate:** Standard schedule (details in `config.py`)
- **Epochs:** 50
- **DataLoader:** `persistent_workers=True` for minor throughput optimization

### 4.6 Held-Out Test Set Discipline
The test set is treated as a true held-out evaluation. It has **not been evaluated** on the final `UNet3DPaper` architecture. Evaluation will occur exactly once after all architecture and loss decisions are finalized, per one-look discipline.

---

## 5. Evaluation Metrics

### 5.1 Segmentation Metrics
All metrics computed at the patch level:

- **Dice Similarity Coefficient (DSC):** Primary overlap metric.
- **HD95 (95th Percentile Hausdorff Distance):** Boundary precision, in millimeters. Empty or skipped predictions are **penalized** with the maximum possible patch diagonal (~110.85 mm for a 64³ patch) rather than excluded from the mean. This addresses a previously flagged optimistic bias.
- **HD100 (Maximum Hausdorff Distance):** Worst-case boundary error.
- **ASSD (Average Symmetric Surface Distance):** Mean surface distance.
- **IoU (Intersection over Union):** Alternative overlap metric.
- **Precision, Recall, Specificity, F1:** Derived from MONAI's `ConfusionMatrixMetric`.
- **Raw TP/FP/FN/TN voxel counts:** Logged to enable any confusion-matrix metric to be recomputed later without rerunning.

### 5.2 Per-Nodule-Size Analysis
Dice is additionally reported per size bucket (small / medium / large by ground-truth foreground voxel count) to detect size-dependent performance biases.

### 5.3 Statistical Testing
Pairwise comparisons across configurations use:
- **Paired t-test**
- **Wilcoxon signed-rank test**

Both applied to matched seeds for `best_val_dice` and `best_val_hd95`. Significance threshold: $\alpha = 0.05$.

### 5.4 Gradient Telemetry
A standalone, non-invasive diagnostic script (`grad_telemetry_check.py`) measures gradient magnitude of the boundary loss term vs. the regional (Dice+Focal) loss term using a trained checkpoint and real validation batches — deliberately not modifying `train.py`, just inspecting gradients via `torch.autograd.grad`.

---

## 6. Limitations and Scope

- **SDF patch-truncation:** The SDF is computed within a cropped 64³ patch with no knowledge of true nodule geometry beyond the patch edge. This is a known geometric approximation.
- **Single-architecture-family scope:** Only the 3D U-Net family was evaluated; no attention mechanisms, transformer blocks, or other architectures were tested.
- **Boundary-loss weight ceiling:** The null result on boundary-loss weighting is specific to $\lambda_{max} = 0.05$. Higher weights were not explored due to the observed gradient-starvation diagnosis.
- **Single dataset:** Results are specific to LIDC-IDRI; generalization to other pulmonary nodule datasets or other 3D segmentation tasks is not claimed.
