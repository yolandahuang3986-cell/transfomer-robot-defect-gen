# Quality and cost ledger — 18 October 2026

Supersedes [the 2026-10-17 ledger](quality_cost_ledger_20261017.md): adds the
accepted screw Procedural pool (replacing the prior on-hold v1 pool) and the
accepted Diffusion pool for both categories. GAN is unchanged — no new training,
still `not_run_quality_gate`.

## Candidate-pool generation cost

| Method | Category | Attempts | Shared-QC passes | Pass rate | Formal status |
|---|---|---:|---:|---:|---|
| Procedural | metal_nut | 83 | 75 | 90.36% | accepted_and_evaluated |
| Procedural | screw | 134 | 125 | 93.28% | accepted_and_evaluated (v2, replaces on-hold v1) |
| Diffusion | metal_nut (3 types) | 467 | 75/86 | — | accepted_and_evaluated |
| Diffusion | screw (5 types) | 1963 | 125/125 | — | accepted_and_evaluated |
| GAN | both | — | — | — | not_run_quality_gate |

Procedural and Diffusion use incomparable attempt-cost units: Procedural's
"attempts" are whole CPU patch-composite-and-QC cycles; Diffusion's are internal
stage-1 mask-generation attempts, a separate, earlier filter before the pair even
reaches shared QC (see `member_c_delivery.md` §3). Diffusion's shared-QC step
itself passes 200/211 pairs (94.8%) once a pair clears stage 1.

Procedural screw's new v2 pool (`screw_dark_core_colored_polygon_v4`, PR #10)
replaces the held v1 pool: 125/125 accepted from 134 attempts, 9 `mask_too_small`
rejects, 93.28% acceptance. A discarded development attempt (120/125 at 1250
attempts, 1130 too-small rejects) is recorded separately; none of its images were
reused.

## Diffusion: cost per accepted pair, by defect type

| Type | Worker calls | Stage-1 attempts/budget | Shared QC | Generation s | Model load s | s/pair (gen only) | s/pair (incl. load) |
|---|---:|---:|---:|---:|---:|---:|---:|
| metal_nut/bent | 1 | 56/3250 | 25/25 | 212.3 | 155.9 | 8.5 | 14.7 |
| metal_nut/color | 1 | 225/3250 | 25/25 | 635.9 | 62.7 | 25.4 | 27.9 |
| metal_nut/scratch | 6 | 186/4680 | 25/36 | 620.8 | 382.0 | 24.8 | 40.1 |
| screw/manipulated_front | 1 | 147/3250 | 25/25 | 443.0 | 65.3 | 17.7 | 20.3 |
| screw/scratch_head | 1 | 645/3250 | 25/25 | 1651.2 | 63.3 | 66.1 | 68.6 |
| screw/scratch_neck | 1 | 102/3250 | 25/25 | 328.0 | 65.1 | 13.1 | 15.7 |
| screw/thread_side | 1 | 1025/3250 | 25/25 | 2582.4 | 72.6 | 103.3 | 106.2 |
| screw/thread_top | 1 | 44/3250 | 25/25 | 193.3 | 65.4 | 7.7 | 10.3 |
| **Total** | **13** | **2430/27430** | **200/211** | **6666.9** | **932.5** | **33.3** | **38.0** |

A independently recomputed these totals from each bundle's `diffusion_internal.json`
and `qc_report.json` on 2026-10-18; they match C's reported figures exactly on
attempts, accepted counts and generation seconds (932.5s recomputed vs 932.3s
reported for model loads — rounding only).

Cost per accepted pair varies 40x across defect types (7.7s to 103.3s), driven by
stage-1 mask yield (2.4%–56.8%, see `member_c_delivery.md` §10.1), not by inpainting:
the sampling-steps study (§9 of the same doc) shows that halving inpaint steps
200→50 saves only ~2s/pair, a small fraction of the per-type cost spread.
`metal_nut/scratch` needed 5 refill worker calls (11/36 rejected `mask_too_large`
by shared QC's object-relative area rule, though they passed the internal
image-relative rule), each reloading the model — its 382.0s of load time against
63–73s for single-call types.

Diffusion's reported costs do not include the one-time, shared-across-all-formal-
generation model training (anomaly model job 40227, mask models job 39986); those
are fixed costs not amortized per defect type here.

## Downstream training cost

| Method | Category | Steps/run | Timing | Scope |
|---|---|---:|---:|---|
| real_only (E0) | metal_nut | 1500 | 80.79s mean | Trainer-internal timer |
| real_only (E0) | screw | 2200 | 120.05s mean | Trainer-internal timer |
| Procedural (E1) | metal_nut | 1500 | 68.59s mean | Trainer-internal timer |
| Procedural (E1) | screw | 2200 | 106.98s mean (subprocess) | Slurm/subprocess wall time* |
| Diffusion (E3) | metal_nut | 1500 | 74.74s mean (subprocess) | Slurm/subprocess wall time* |
| Diffusion (E3) | screw | 2200 | 109.76s mean (subprocess) | Slurm/subprocess wall time* |

\*The three 2026-10-18 rows use a different timing method than the 2026-10-17 rows
above them (no internal trainer-clock field was captured for these runs; the value
is each seed's subprocess wall time from `run_plan.json`, including bundle loading
and the formal-loader shared-QC recheck). Diffusion's subprocess time includes
loading 3 (metal_nut) or 5 (screw) bundle directories per seed, vs Procedural's 1.
Do not read small cross-row differences here as a controlled speed comparison;
full per-seed values and scope notes are in the [JSON ledger](quality_cost_ledger_20261018.json).

## What this does and does not establish

- No controlled cross-method speed ranking: Procedural (CPU patch compositing),
  Diffusion (GPU sampling) and GAN (no formal pool) were measured with different
  timers, on different hardware paths, at different pipeline stages.
- Diffusion's per-type cost spread (7.7–103.3 s/pair) is itself informative for
  planning future generation budgets, independent of any cross-method comparison.
- GAN remains `not_run_quality_gate`; no cost or downstream-effectiveness
  comparison is possible or claimed for it.
- Null/missing values denote unmeasured or inapplicable quantities, never a
  zero cost or a zero quality score.

[Full-precision ledger and source figures](quality_cost_ledger_20261018.json).
