#!/usr/bin/env python3
"""Recompute every training/validation-side number from the raw result JSONs.

Run from the repo root:
    python tools/make_tables_validation.py
Writes outputs/generated/validation_tables.md and prints it.
Needs numpy and scipy. Reads only outputs/results/*.json.

All +/- values are population SD (ddof=0), matching aggregate_results.py.
"""
import glob, json, os, re, sys
from collections import Counter
import numpy as np
from scipy import stats

RES, OUT = "outputs/results", "outputs/generated"
CONFIGS = ["baseline", "boundary", "boundary_adaptive"]
LABEL = {"baseline": "baseline", "boundary": "boundary_fixed", "boundary_adaptive": "boundary_adaptive"}
SEEDS = [42, 123, 456, 789, 2024]
ARCHS = ["unet3d_paper", "unetr"]
PAT = re.compile(r"^(baseline|boundary_adaptive|boundary)_(unet3d_paper|unetr)_seed(\d+)(?:_(.+))?\.json$")
GATE_ENTER = 0.005

lines = []
def P(s=""):
    lines.append(s); print(s)

def load_all():
    runs = {}
    for f in sorted(glob.glob(f"{RES}/*.json")):
        m = PAT.match(os.path.basename(f))
        if not m:
            P(f"<!-- skipped unrecognised file: {os.path.basename(f)} -->"); continue
        cfg, arch, seed, tag = m.group(1), m.group(2), int(m.group(3)), m.group(4) or ""
        runs[(arch, cfg, seed, tag)] = json.load(open(f))
    return runs

def vals(runs, arch, cfg, key, tag=""):
    return np.array([runs[(arch, cfg, s, tag)][key] for s in SEEDS], dtype=float)

def msd(x, nd=4):
    return f"{np.mean(x):.{nd}f} ± {np.std(x):.{nd}f}"

def ptests(a, b):
    t, pt = stats.ttest_rel(a, b)
    try:
        pw = stats.wilcoxon(a, b).pvalue
    except ValueError:
        pw = float("nan")
    return t, pt, pw

def best_row(run, field):
    return run["history"][run["best_val_dice_epoch"] - 1][field]

def missing(runs):
    miss = [(a, c, s, t) for a in ARCHS for c in CONFIGS for s in SEEDS for t in [""] if (a, c, s, t) not in runs]
    miss += [("unet3d_paper", c, s, t) for c in ["boundary", "boundary_adaptive"] for s in SEEDS
             for t in ["correctedSDF", "rotationCorrected"] if ("unet3d_paper", c, s, t) not in runs]
    return miss

# ---------------------------------------------------------------- sections
def sec_inventory(runs):
    P("## 0. Inventory\n")
    miss = missing(runs)
    P(f"Runs loaded: {len(runs)}. Missing expected runs: {len(miss)}" + (f" -> {miss}" if miss else "") + "\n")

def sec_aggregate(runs):
    P("## 1. Validation aggregates (N = 5 per config, population SD)\n")
    for arch in ARCHS:
        P(f"### {arch}\n")
        P("| Config | best_val_dice | best_val_hd95 | final_val_iou | IoU at best-Dice epoch | val_dice_mean_last10 | val_hd95_mean_last10 |")
        P("|---|---|---|---|---|---|---|")
        for c in CONFIGS:
            iou_best = np.array([best_row(runs[(arch, c, s, "")], "val_iou") for s in SEEDS])
            P(f"| {LABEL[c]} | {msd(vals(runs,arch,c,'best_val_dice'))} | {msd(vals(runs,arch,c,'best_val_hd95'))} | "
              f"{msd(vals(runs,arch,c,'final_val_iou'))} | {msd(iou_best)} | "
              f"{msd(vals(runs,arch,c,'val_dice_mean_last10'))} | {msd(vals(runs,arch,c,'val_hd95_mean_last10'))} |")
        P()

def sec_pairwise(runs):
    P("## 2. Pairwise validation tests (paired t, Wilcoxon), matched by seed\n")
    pairs = [("baseline", "boundary"), ("baseline", "boundary_adaptive"), ("boundary", "boundary_adaptive")]
    for arch in ARCHS:
        P(f"### {arch}\n")
        P("| Metric | A vs B | mean A | mean B | t | t p | Wilcoxon p |")
        P("|---|---|---|---|---|---|---|")
        for key in ["best_val_dice", "best_val_hd95", "final_val_iou", "val_dice_mean_last10"]:
            for a, b in pairs:
                x, y = vals(runs, arch, a, key), vals(runs, arch, b, key)
                t, pt, pw = ptests(x, y)
                P(f"| {key} | {LABEL[a]} vs {LABEL[b]} | {x.mean():.4f} | {y.mean():.4f} | {t:.3f} | {pt:.4f} | {pw:.4f} |")
        P()

def sec_size(runs):
    P("## 3. Size-stratified validation Dice at each run's best-Dice epoch\n")
    for arch in ARCHS:
        P(f"### {arch}\n")
        P("| Config | small | medium | large |")
        P("|---|---|---|---|")
        ordered, total, exceptions = 0, 0, []
        for c in CONFIGS:
            cols = []
            for sz in ["small", "medium", "large"]:
                cols.append(np.array([best_row(runs[(arch, c, s, "")], f"dice_{sz}_mean") for s in SEEDS]))
            P(f"| {LABEL[c]} | " + " | ".join(msd(x) for x in cols) + " |")
            for i, s in enumerate(SEEDS):
                total += 1
                if cols[0][i] < cols[1][i] < cols[2][i]:
                    ordered += 1
                else:
                    exceptions.append((LABEL[c], s))
        P(f"\nsmall < medium < large holds in {ordered}/{total} runs. Exceptions: {exceptions}\n")

def gate_stats(run):
    H = run["history"]
    act = [bool(h.get("gate_active")) for h in H]
    w = [(h.get("boundary_weight") or 0.0) for h in H]
    vel = [h.get("velocity") for h in H]
    on_events = sum(1 for i, a in enumerate(act) if a and (i == 0 or not act[i - 1]))
    off_events = sum(1 for i in range(1, len(act)) if (not act[i]) and act[i - 1])
    first_on = next((i + 1 for i, a in enumerate(act) if a), None)
    qual = [(v is not None and v < GATE_ENTER) for v in vel]
    streak = best = 0
    for q in qual:
        streak = streak + 1 if q else 0
        best = max(best, streak)
    vv = [v for v in vel if v is not None]
    active_w = [x for x, a in zip(w, act) if a]
    return dict(first_on=first_on, on_events=on_events, off_events=off_events, n_active=sum(act),
                final_active=act[-1] if act else None, max_w=max(w) if w else 0.0,
                mean_w_active=(float(np.mean(active_w)) if active_w else 0.0), mean_w_all=float(np.mean(w)),
                min_vel=(min(vv) if vv else None), n_qual=sum(qual), n_vel=len(vv), longest_streak=best)

def sec_gate(runs):
    P("## 4. Adaptive gate behaviour, from per-epoch history\n")
    for arch in ARCHS:
        P(f"### {arch}: boundary_adaptive\n")
        P("| Seed | activated? | first-on epoch | on-events | off-events | active epochs | final state | max weight | mean wt (active) | mean wt (all) | min velocity | qualifying epochs | longest streak |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        n_act = 0
        for s in SEEDS:
            g = gate_stats(runs[(arch, "boundary_adaptive", s, "")])
            n_act += g["on_events"] > 0
            mv = f"{g['min_vel']:.5f}" if g["min_vel"] is not None else "n/a"
            P(f"| {s} | {'yes' if g['on_events'] else 'NO'} | {g['first_on']} | {g['on_events']} | {g['off_events']} | {g['n_active']} | "
              f"{g['final_active']} | {g['max_w']:.4f} | {g['mean_w_active']:.4f} | {g['mean_w_all']:.4f} | {mv} | {g['n_qual']}/{g['n_vel']} | {g['longest_streak']} |")
        P(f"\nSeeds that opened the gate at least once: {n_act}/5\n")
    P("### Fixed-schedule weight ramp (boundary_fixed)\n")
    P("| Arch | Seed | first epoch weight > 0 | first epoch weight >= 0.05 | max weight |")
    P("|---|---|---|---|---|")
    for arch in ARCHS:
        for s in SEEDS:
            H = runs[(arch, "boundary", s, "")]["history"]
            w = [(h.get("boundary_weight") or 0.0) for h in H]
            f0 = next((i + 1 for i, x in enumerate(w) if x > 0), None)
            f1 = next((i + 1 for i, x in enumerate(w) if x >= 0.05 - 1e-9), None)
            P(f"| {arch} | {s} | {f0} | {f1} | {max(w):.4f} |")
    P("\n(epochs are 1-indexed as stored in the JSON `epoch` field)\n")

def sec_audits(runs):
    P("## 5. SDF audits: original vs corrected retrains (UNet3DPaper)\n")
    for tag in ["correctedSDF", "rotationCorrected"]:
        P(f"### {tag}\n")
        P("| Config | Metric | Original | Corrected | Diff | t p | Wilcoxon p |")
        P("|---|---|---|---|---|---|---|")
        for c in ["boundary", "boundary_adaptive"]:
            for key in ["best_val_dice", "val_dice_mean_last10", "best_val_hd95", "val_hd95_mean_last10", "final_val_iou"]:
                o = vals(runs, "unet3d_paper", c, key, "")
                n = vals(runs, "unet3d_paper", c, key, tag)
                t, pt, pw = ptests(o, n)
                P(f"| {LABEL[c]} | {key} | {msd(o)} | {msd(n)} | {n.mean()-o.mean():+.4f} | {pt:.4f} | {pw:.4f} |")
        P()

def sec_probes(runs):
    P("## 6. Probes and report-claim checks\n")
    k = ("unet3d_paper", "boundary", 42, "lambda0.5")
    ref = runs.get(("unet3d_paper", "boundary", 42, ""))
    if k in runs and ref:
        r = runs[k]
        P("### λ = 0.50 probe (UNet3DPaper, seed 42)\n")
        P(f"- target weight field: {r.get('boundary_loss_weight_target')}, warmup epochs: {r.get('boundary_warmup_epochs')}, epochs completed: {r['epochs_completed']}")
        P(f"- last-10 val Dice: {r['val_dice_mean_last10']:.4f} ± {r['val_dice_std_last10']:.4f}; last-10 HD95: {r['val_hd95_mean_last10']:.4f} ± {r['val_hd95_std_last10']:.4f}")
        P(f"- same seed at λ = 0.05: last-10 Dice {ref['val_dice_mean_last10']:.4f}, last-10 HD95 {ref['val_hd95_mean_last10']:.4f}")
        P(f"- best_val_dice: {r['best_val_dice']:.4f} (λ=0.50) vs {ref['best_val_dice']:.4f} (λ=0.05)\n")
    else:
        P("λ probe file not found.\n")
    kk = ("unetr", "baseline", 42, "epochs100")
    r50 = runs.get(("unetr", "baseline", 42, ""))
    if kk in runs and r50:
        r = runs[kk]
        P("### 100-epoch UNETR probe (baseline, seed 42)\n")
        P(f"- epochs completed: {r['epochs_completed']}; best_val_dice {r['best_val_dice']:.4f} at epoch {r['best_val_dice_epoch']}; last-10 {r['val_dice_mean_last10']:.4f} ± {r['val_dice_std_last10']:.4f}")
        P(f"- 50-epoch run, same seed: best_val_dice {r50['best_val_dice']:.4f} at epoch {r50['best_val_dice_epoch']}; last-10 {r50['val_dice_mean_last10']:.4f}\n")
    cnn = runs.get(("unet3d_paper", "baseline", 42, ""))
    if cnn:
        P("### UNet3DPaper baseline, seed 42 (for the report's plateau comparison)\n")
        P(f"- best_val_dice {cnn['best_val_dice']:.4f}; last-10 mean {cnn['val_dice_mean_last10']:.4f} ± {cnn['val_dice_std_last10']:.4f}")
        P(f"- 5-seed baseline best_val_dice mean ± SD: {msd(vals(runs,'unet3d_paper','baseline','best_val_dice'))} (this is the number the report labels as seed-42 last-10)\n")

def sec_integrity(runs):
    P("## 7. Run integrity\n")
    for arch in ARCHS:
        main = [runs[(arch, c, s, "")] for c in CONFIGS for s in SEEDS if (arch, c, s, "") in runs]
        gpu_h = sum(sum(h.get("epoch_time_sec", 0) for h in r["history"]) for r in main) / 3600
        peak = [max(h.get("peak_gpu_mem_mb", 0) for h in r["history"]) for r in main]
        skipped = [best_row(r, "hd95_skipped") for r in main]
        P(f"- **{arch}**: {len(main)} runs; code_version {dict(Counter(r['code_version'] for r in main))}; "
          f"pytorch {dict(Counter(r.get('pytorch_version') for r in main))}; epochs_completed {dict(Counter(r['epochs_completed'] for r in main))}; "
          f"params {set(r['n_parameters'] for r in main)}; GPU-hours {gpu_h:.2f}; peak memory {min(peak)/1000:.2f}–{max(peak)/1000:.2f} GB; "
          f"hd95_skipped at best-Dice epoch max {max(skipped)}")
    P()

def main():
    if not os.path.isdir(RES):
        sys.exit(f"{RES} not found: run from the repo root")
    runs = load_all()
    P("# Validation-side tables (generated by tools/make_tables_validation.py)\n")
    for f in (sec_inventory, sec_aggregate, sec_pairwise, sec_size, sec_gate, sec_audits, sec_probes, sec_integrity):
        f(runs)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/validation_tables.md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[written to {OUT}/validation_tables.md]")

if __name__ == "__main__":
    main()
