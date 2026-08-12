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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Boundary loss settings (fixed schedule fallback)
BOUNDARY_LOSS_WEIGHT = 0.05
BOUNDARY_WARMUP_EPOCHS = 15

# Adaptive Boundary-Loss Gating (PCG-BW) Hyperparameters
ALPHA_EMA = 0.3          # EMA smoothing coefficient for validation primary loss
K_WINDOW = 3             # Evaluation window (epochs) for velocity computation
TAU_VELOCITY = 0.005     # Velocity threshold (0.5% relative improvement per k epochs)
GAMMA_SIGMOID = 0.001    # Transition sharpness parameter
LAMBDA_MAX = 0.05        # Upper bound weight matching fixed-schedule maximum

# Hysteresis settings for adaptive gating (replaces direct instantaneous gating)
GATE_TAU_ENTER = 0.005    # velocity must drop below this to start counting toward activation
GATE_TAU_EXIT = 0.02      # velocity must rise above this to start counting toward deactivation
                          # (deliberately higher than TAU_ENTER — creates a dead zone, prevents
                          #  flicker when velocity hovers near a single threshold)
GATE_PATIENCE = 3         # consecutive epochs required before actually flipping state
WEIGHT_EMA_ALPHA = 0.3    # smooths the weight itself once active, avoids sudden jumps