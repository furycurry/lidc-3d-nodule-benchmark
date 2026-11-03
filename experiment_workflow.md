# Experiment Tracking & Archiving Workflow

Reference guide for the multi-seed training, result-capture, and archiving system built for the LIDC-IDRI nodule segmentation project. Come back to this whenever you forget a command or the overall flow.

---

## The core idea

Every run automatically captures **everything** (Tier A: all metrics, cheap, no decisions needed). You only manually **promote** the runs worth keeping long-term into a permanent, clearly-named archive (Tier B: per-sample data, qualitative examples, model complexity — captured once, on demand).

```
[ Run a batch of experiments ]
            │
            ▼
[ outputs/results/*.json ]   <- every run lands here automatically, disposable/working data
            │
            ▼
[ aggregate_results.py ]     <- quick console summary, decide what's worth keeping
            │
            ▼
[ archive_run.py ]           <- promote ONLY the chosen run(s) into a permanent, named folder
            │
            ▼
[ outputs/archive/<name>/ ]  <- checkpoints + full metrics + per-sample data + qualitative examples
            │
            ▼
[ export_archive_csv.py ]    <- rebuilds two clean CSVs from everything in the archive
            │
            ▼
[ archive_summary.csv, archive_epoch_history.csv ]  <- pull these into pandas/Excel for charts
```

---

## File locations (inside the container, all under `/workspace`)

| Path | Purpose |
|---|---|
| `train.py` | Training script. Captures full Tier A metrics every run. |
| `boundary_loss.py` | Adaptive/fixed boundary loss implementation. |
| `config.py` | Shared hyperparameters and paths. |
| `run_multiseed.sh` | Launches a batch of seeds/configs in parallel across GPUs. |
| `aggregate_results.py` | Quick mean±std summary across a batch, printed to console. |
| `archive_run.py` | Promotes one chosen run into the permanent archive. |
| `export_archive_csv.py` | Rebuilds the two plotting-ready CSVs from the archive. |
| `outputs/results/*.json` | Raw, disposable results — one per run, every run. |
| `outputs/checkpoints/<experiment_name>/` | `best_dice_unet3d.pt` and `best_hd95_unet3d.pt` per run. |
| `outputs/archive/<archived_name>/` | Permanent, curated runs — checkpoints, metrics, qualitative examples. |
| `outputs/archive_summary.csv` | One row per archived run — for bar charts, comparison tables. |
| `outputs/archive_epoch_history.csv` | One row per epoch per archived run — for learning curves. |
| `outputs/logs/*.log` | Live training output per run, for monitoring. |

---

## Step-by-step: running a batch and archiving the result

### 1. Launch a batch of runs (detached, survives disconnection)

```bash
docker exec -d medtest setsid bash -c "cd /workspace && bash run_multiseed.sh > outputs/logs/launcher.log 2>&1 < /dev/null"
```

**Run this exactly once per batch.** Do not re-run it to "check" — that stacks duplicate processes. Just wait and check logs.

### 2. Check progress periodically (safe to Ctrl+C, only stops the viewing)

```bash
docker exec -it medtest bash -c "tail -n 3 /workspace/outputs/logs/*.log"
```

### 3. Once all runs finish, get the quick comparison

```bash
docker exec -it medtest python /workspace/aggregate_results.py
```

### 4. Decide which run(s) are worth keeping, then promote

```bash
docker exec -it medtest python /workspace/archive_run.py \
    --run boundary_seed2024 \
    --name "unet3d_boundary_final_seed2024" \
    --notes "best HD95 across 3-seed batch, most stable convergence"
```

Naming convention: `{architecture}_{loss_variant}_{descriptive_tag}_seed{N}`
e.g. `unet3d_adaptive-boundary_final_seed42`, `attn-unet3d_baseline_seed123`

### 5. Rebuild the plotting-ready CSVs

```bash
docker exec -it medtest python /workspace/export_archive_csv.py
```

### 6. Pull the CSVs to your laptop whenever you want to actually plot something

```bash
scp -P 80 arnob@10.143.0.81:/data/project/aj/outputs/archive_summary.csv ~/Downloads/
scp -P 80 arnob@10.143.0.81:/data/project/aj/outputs/archive_epoch_history.csv ~/Downloads/
```

---

## What each run captures automatically (Tier A — every run)

- Standard: Dice, HD95, HD100, ASSD, IoU
- Confusion-matrix based: Precision, Recall/Sensitivity, Specificity, F1
- Raw counts: TP, FP, FN, TN (summed across validation set — lets you recompute any confusion-matrix metric later without re-running)
- Per-size-bucket Dice: small / medium / large nodules (based on ground-truth voxel count)
- Per-loss-component values: Dice component, Focal component, Boundary component (tracked separately, not just the summed total)
- Training dynamics: epoch wall-clock time, peak GPU memory, learning rate, boundary weight (if applicable)
- Dual best checkpoints: `best_dice_unet3d.pt` and `best_hd95_unet3d.pt`, tracked independently
- Full run metadata: seed, architecture, loss variant, git commit hash, timestamp, GPU name, PyTorch version, hostname, parameter count, config snapshot (batch size, LR, epochs, split sizes)
- Complete per-epoch history for every metric above

## What gets added only when archived (Tier B — promoted runs only)

- Per-sample (per-patch) validation metrics — one row per patch, needed for statistical significance testing (paired t-test/Wilcoxon) and violin plots
- Three qualitative examples saved as `.npz` (image + ground truth + prediction arrays): best case, worst case, median case — ready-made for figure panels, no need to reload the model later
- Model parameter count

---

## Reminders / lessons learned (don't repeat these mistakes)

- **Never launch the multi-seed script with `docker exec -it ... &`** — background jobs tied to an interactive terminal die if that terminal session ends or receives Ctrl+C. Always use `docker exec -d medtest setsid bash -c "..."` for true detachment.
- **Never re-run the launcher "just to check"** — each run stacks new processes on top of old ones without killing the originals, causing GPU/CPU contention and corrupted checkpoints (multiple processes writing to the same file).
- **`ps aux` truncates long command lines** — use `ps auxww` (with the double-w) if you need to grep for `train.py` specifically, or just grep for `python` alone.
- **A single seed is not sufficient evidence for any claim.** Always compare across at least 3-5 seeds before concluding a method helps or hurts.
- **Dice-based checkpoint selection can hide the true best HD95** — this is why dual checkpointing (best-by-Dice and best-by-HD95, tracked independently) matters. A single checkpoint criterion can make a genuinely good method look bad just because its best boundary-precision epoch didn't coincide with its best overlap epoch.
- **The test set (`train_val_test_split.json`'s `test_keys`) is evaluated exactly once**, only for the final, fully-decided model — never during architecture/loss iteration.

---

## Quick troubleshooting

| Symptom | Fix |
|---|---|
| `ps aux \| grep train.py` shows nothing but training seems to have stopped early | Check `outputs/logs/<name>.log` directly for the actual last line — could be a real crash, or `ps aux`'s column truncation hiding it (use `ps auxww` instead) |
| Empty `outputs/results/` after a batch | Runs haven't finished yet (results only write at the very end), or something crashed — check individual logs |
| Multiple processes for the same seed at different timestamps | A previous launch was never properly killed before relaunching — `docker exec -it medtest pkill -9 -f "train.py"`, verify with `ps auxww`, clean up `outputs/checkpoints/`, then relaunch once |
