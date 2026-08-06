import torch
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
H5_PATH = BASE_DIR / "lidc_patches_int16.h5"
SDF_H5_PATH = BASE_DIR / "lidc_sdf_cache.h5"
OUTPUT_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Preprocessing Specs
PATCH_SIZE = (64, 64, 64)
HU_MIN = -1000.0
HU_MAX = 400.0

# Training Hyperparameters
BATCH_SIZE = 8
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
EPOCHS = 50
TRAIN_VAL_SPLIT = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Boundary loss settings (seed and USE_BOUNDARY_LOSS now passed via CLI, not hardcoded here)
BOUNDARY_LOSS_WEIGHT = 0.05
BOUNDARY_WARMUP_EPOCHS = 15