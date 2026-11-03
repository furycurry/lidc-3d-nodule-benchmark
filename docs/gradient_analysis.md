# Gradient Analysis: Mechanistically Explaining the Null Result

`statistical_analysis.md` shows that boundary-aware loss terms produce no statistically significant improvement over the Dice+Focal baseline, across N=5 seeds, on both Dice and HD95. This document goes one level deeper: rather than stopping at "no significant difference was found," it uses gradient-magnitude telemetry (`scripts/grad_telemetry_check.py`) to measure *why* — directly, at the optimization level rather than the metric level.

## Method

`grad_telemetry_check.py` is a standalone, non-invasive diagnostic that measures the gradient norm contributed by each loss component separately:

- The **regional loss** (Dice + Focal) gradient norm.
- The **boundary loss** gradient norm, computed at its raw, unweighted magnitude (weight = 1.0), isolating how large the boundary term's gradient signal is on its own, before any scaling is applied.

Both are averaged over batches to get a stable estimate, rather than relying on a single batch's noise.

## Result

```
Regional (Dice+Focal) gradient norm, mean over batches: 0.000855
Boundary (raw, weight=1.0) gradient norm, mean over batches: 0.000008
Boundary share of combined raw gradient magnitude: 0.94%
At the actual trained weight (0.05), boundary's effective contribution is scaled down further: ~0.05%
```

The boundary loss term's raw gradient magnitude (0.000008) is roughly **two orders of magnitude smaller** than the regional loss term's (0.000855) — even before any weighting is applied. At raw magnitude, it accounts for only **0.94%** of the combined gradient signal.

The boundary loss is then further attenuated by its actual trained weighting coefficient (0.05 — the `LAMBDA_MAX` / fixed-schedule target used in this project's configuration). Applying that weight brings its **effective contribution to the combined gradient down to roughly 0.05%** of the total.

## Interpretation

This is the mechanistic explanation for the null result in `statistical_analysis.md`: it is not that boundary-aware supervision is conceptually ineffective for this task, but that — at the weighting range explored here (up to λ_max = 0.05) — the boundary term's gradient signal is small enough, relative to the regional loss, that it has negligible influence on the direction of optimization. A loss term contributing ~0.05% of the effective gradient magnitude has very little room to move model parameters differently than the regional loss alone would, regardless of how the weighting schedule (fixed vs. adaptive/gated) shapes *when* that small signal is applied.

This reframes the finding from "boundary loss didn't help" to a more specific and actionable one: **the boundary term's gradient magnitude, at the weight range tested, is too small relative to the regional loss to meaningfully influence training** — which points directly at gradient-magnitude-aware dynamic balancing (e.g., GradNorm-style rescaling of the boundary term to be comparable in magnitude to the regional loss, rather than a small fixed or gated multiplier) as the natural next step, rather than further tuning of the gating schedule itself. This is noted as future work in the main README.

## Scope note

This measurement was taken at a single point (one weight configuration, one gradient snapshot) rather than tracked across the full training trajectory. The magnitude gap reported here (~100x) is large enough that it is unlikely to fully close over training, but a per-epoch gradient trajectory was not measured, so this document reports a diagnostic snapshot rather than an exhaustive characterization of gradient dynamics throughout training.