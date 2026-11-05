"""Shared loader: select exactly the main training-run JSONs for one architecture.

Filenames follow  {baseline|boundary|boundary_adaptive}_{architecture}_seed{N}[_{tag}].json
  tag ''                  -> the main 15-run matrix for that architecture
  tag 'correctedSDF'      -> truncation-corrected SDF retrains (UNet3DPaper)
  tag 'rotationCorrected' -> rotation-corrected SDF retrains (UNet3DPaper)
Probe runs (..._lambda0.5, ..._epochs100) are never matched unless requested as a tag.
"""
import json
import re
from pathlib import Path

CONFIG_NAMES = {"baseline": "baseline", "boundary": "boundary_fixed", "boundary_adaptive": "boundary_adaptive"}


def load_runs(results_dir="outputs/results", architecture="unet3d_paper", tag=""):
    suffix = rf"_{re.escape(tag)}" if tag else ""
    pat = re.compile(rf"^(baseline|boundary_adaptive|boundary)_{re.escape(architecture)}_seed(\d+){suffix}\.json$")
    runs = {name: {} for name in CONFIG_NAMES.values()}
    for f in sorted(Path(results_dir).glob("*.json")):
        m = pat.match(f.name)
        if m:
            with open(f) as fp:
                runs[CONFIG_NAMES[m.group(1)]][int(m.group(2))] = json.load(fp)
    return runs
