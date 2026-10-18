# Member C delivery: AnomalyDiffusion generator (formal pool generated)

Status on 2026-10-18: formal pool generated, 8 types x 25 = 200 pairs (§8). Inpaint
sampling-steps study (§9) and failure taxonomy (§10) complete. **The formal pool is waiting
for A's leakage review and human audit, and must not enter training until the audit has
passed and is recorded in `acceptance.json`.** No downstream (E3) result exists. The 16
screw/thread_side validation samples (§4) are evidence only and must not enter any formal pool.

## 1. What is implemented

| Part | Where |
|---|---|
| Upstream pin, environment, local patches, checkpoint policy | `external/anomalydiffusion/README.md`, `external/anomalydiffusion/patches/` |
| Worker (Python 3.8 env, subprocess boundary) | `scripts/diffusion/ad_worker.py` |
| Adapter `DiffusionDefectGenerator` (project env) | `src/generation/diffusion_adapter.py`, `src/generation/diffusion_foreground.py` |
| Support-mask statistics in the upstream env | `scripts/diffusion/support_stats.py` |
| Bundle export through `generate_with_qc` | `scripts/generate_diffusion.py`, `configs/diffusion/generation.json` |
| Frozen internal rules and their sources | `configs/diffusion/` (see its README) |
| Analysis / evidence scripts | `scripts/diffusion/analysis/` (see its README) |
| Fake-worker tests (no GPU) | `tests/test_diffusion_adapter.py` |

The adapter rejects out-of-scope strata (metal_nut/flip), any source that is not a frozen
train/good image, DRAFT rule files, and raw outputs inside Git. It never backfills: a
stage-1 shortfall raises `DiffusionUnderfilledError` and the export writes an
`underfilled_*` status. Every worker call keeps its request, result, log, placement
foregrounds and (with `save_rejects`) every rejected stage-1 mask.

## 2. Training provenance (no held-out data, no flip)

- Base weights: upstream generic LDM `text2img-large` only. Official AnomalyDiffusion
  defect checkpoints were used for feasibility only and are not used for generation.
- Training data: the frozen 5-shot support (`fewshot_n5_seed42`) via the
  `SUPPORT_LIST` patch; each job log prints `[patch] SUPPORT_LIST ...`.
- Anomaly model: one joint model on the 8 in-scope groups (flip excluded), 10000 steps,
  SLURM job 40227; checkpoint step 9999 (final step, not selected on any metric).
  `check_anomaly_embeddings.py`: 73 name slots, exactly the 8 in-scope names trained,
  `metal_nut+flip` untrained.
- Mask models: one per defect type on its 5 support masks, 5000 steps; step 2999 used.
  The step was chosen from a mask probe on generated masks only (yield and area vs
  support), not from held-out data.
- An earlier anomaly model trained with flip (`p1_anomaly_s42`) and the flip mask model
  exist on TC2, are marked analysis-only, and are not referenced by any config.

## 3. Diffusion-internal rules (before shared QC)

Frozen versions in use: `mask_filter_v2` (`07de291f8010989e`) and `placement_v2`
(`51159d583f2821c7`); full description and disclosure in `configs/diffusion/README.md`.

Latent-cell rule (exact): after placement, `torchvision.transforms.Resize((32, 32))`
on the 256 x 256 binary mask, keep cells `>= 0.5`; require
`cells >= max(2, min over the 5 support masks of the same defect type)`. Computed in the
upstream environment (torchvision 0.13.1), as in `ddpm.log_images`.
Source disclosure: support minima come from training data; the floor 2 and the other
filter thresholds were set during development after looking at generated samples.
They were not set in advance. No held-out image, mask or metric was used.

Shared QC (`synthetic-qc-v1`) is unchanged. Exports load sources with the formal
loader's convention (RGB bilinear to 256) so the export QC equals A's recheck.

## 4. Validation runs: screw/thread_side, 16 samples, base seed 42

| | Run 1: job 42012 | Run 2: job 42144 |
|---|---|---|
| Code | pre-PR scripts (worker `63768a0d`) | PR commit `2465105`, clean |
| Sources | first 16 sorted train/good | 16 drawn with seed 42 from 320 |
| Stage-1 budget | 50 x (an earlier attempt at 30 x, job 40879, underfilled 14/16 and was rerun) | 50 x, fixed in config before the run |
| Stage-1 attempts / yield | 562 / 2.8 % | 705 / 2.3 % |
| Rejected: placement_failed | 213 | 284 |
| Rejected: too_small_or_empty | 124 | 146 |
| Rejected: dirty_bg | 86 | 106 |
| Rejected: upstream check_mask | 68 | 81 |
| Rejected: inverted | 32 | 41 |
| Rejected: latent_too_small | 15 | 21 |
| Rejected: too_large | 8 | 10 |
| Shared QC (export) | 16/16 | 16/16, 16 attempts |
| Shared QC, formal-loader convention | 16/16 | 16/16 |
| Generation s / accepted sample (incl. its share of all rejected attempts) | 88.6 | 110.6 |
| Model loads | 126 s | 150 s |
| Wall time | 26 min | 33 min |
| Rejected masks kept as images | no (counts only) | yes, all 689 |

Shape-passing masks that then failed placement: 87 % (run 1) and 88 % (run 2).
Placement onto the thin screw core is the largest single cost driver.

> **Note added after commit `741d235` (text above unchanged).** Every run above, and every run
> before `741d235`, used a stage-1 budget of 50 x per requested sample. From `741d235` on the
> budget is 130 x for all categories (`sampling.mask_attempt_factor` in
> `configs/diffusion/generation.json`; 4 samples -> 520 attempts). Reason: validation job 43273
> (seed 1042) underfilled screw/thread_side 3/4 at 200/200 attempts (yield 1.5 %, 95 % CI
> 0.3–4.3 %). Only the budget changed; `placement_v2`, `mask_filter_v2` and shared QC are
> unchanged. Evidence: `~/outputs/diffusion/raw_runs/validation/screw_thread_side/seed1042/`
> (`result.json`, `rejects.json`, `rejects/`). First run at 130 x: job 43341 (seed 2042),
> screw/thread_side and screw/scratch_head both 4/4, raw runs in `.../seed2042/`.

### Human visibility (separate from QC)

Scale: V visible without comparing / B only when compared to the source / N not visible.
Labels in `docs/diffusion_evidence/`.

| | Run 1 | Run 2 |
|---|---|---|
| Annotators | Yuchen (no ratios shown); Claude saw ratios first (**not blind**) | both blind; Claude's file (with nonce) hash-committed `38c2a691...` before Yuchen's labels were seen |
| Exact agreement (3 levels) | 10/16 | 11/16 (kappa 0.49) |
| Both say V | 8/16 | 6/16 |
| Neither says V | 3/16 | 6/16 |

All disagreements were one level apart. Human labels are a judgement, not ground truth.

### Automatic visibility metrics: not validated, not adopted

| Metric | AUC run 1 (metric designed on this run) | AUC run 2 (independent) |
|---|---|---|
| Old ratio: change in mask / 6 px ring | 0.73 | 0.67 |
| Latent-footprint ratio: change in edited 8x8 cells / far background | 0.96 | 0.64 |

The footprint ratio did not hold on independent data. Pixel change measures how much
was regenerated, not whether the result looks defective (for example run 2 #15: large
change, both annotators N). Neither metric filters or ranks samples. Visibility
acceptance relies on A's human audit.

## 5. Known issues and limits

1. **Label vs edited region.** Upstream inpaints whole 32x32-latent cells (8 x 8 px). The
   edited region is about 1–3 x the saved pixel mask, and in visible samples the change
   often extends outside the mask (run 2: #2, #10, #12, #13). Saved masks therefore
   under-label part of the visible defect. Not changed in this round. Candidate for a
   next round, pending A: snap the mask to its latent cells before inpainting so the
   label equals the editable region (cost: blocky 8 px mask edges).
2. **Many labelled samples show no visible defect** (run 2: 6/16 neither annotator V).
   Recorded, not filtered, per the agreed rule for this round.
3. **Cost.** About 90–110 s of A40 time per accepted sample, stage-1 yield about 2.5 %.
4. **Resolution.** Generation is 256 px (latent 32 x 32). Several screw defect types have
   support masks of only 1–3 latent cells; thread_top has 2/5 support masks of 1 cell, below
   the floor, so the generator does not produce the smallest real thread_top defects.
5. **Shared-QC foreground on screw** includes a shadow halo (24–36 % of the image vs
   about 11–18 % for the screw core). QC is unchanged; reported for A.
6. Placement is object-aware, not part-aware (thread vs head), and translation only.
7. Metal_nut types have not been run yet; no cost or quality claim is made for them.

### Fixed after the PR #5 review (accounting only; sampling and rules unchanged)

| Review finding | Cause | Fix | Effect on the two validation runs |
|---|---|---|---|
| Budget 50, 56 masks judged | The budget was checked once per 8-mask batch | `stage1.run_stage1` checks it before every judged mask; unjudged outputs of the last batch are `mask_surplus_discarded` | None: 562 and 705 attempts, both below the 16 x 50 = 800 budget |
| Zero-output run lost reasons and load cost | The zero-output path wrote a smaller `result.json` | Both paths write `stage1.stage1_result`; the adapter reads the keys strictly | None: both runs produced 16 pairs |
| Cost summary missed QC-rejected pairs | `run_summary.py` summed accepted metadata only | `summarize_runs` totals every worker call; the summary uses it and keeps the old sum only as a cross-check | None: shared QC accepted 16/16 in both runs |

The worker hash changes with this fix; formal runs use the new worker and record
`worker_sha256` and `stage1_sha256`.

## 6. Evidence on TC2 (outside Git)

Home `/home/mcaai/yuchen027`:

- Run 2 bundle: `outputs/diffusion/validation/screw_thread_side_pr_s42/`
  (`batch.npz`, `metadata.json`, `provenance.json`, `qc_report.json`, `diffusion_internal.json`)
- Run 2 raw: `outputs/diffusion/raw_runs/screw_thread_side/seed42/`
  (request, result, worker log, foregrounds, `rejects/`, `rejects.json`, `rejects_sheet.png`,
  `zoom_review.png`, `visibility.json`, `shared_qc_report_formal_convention.json`)
- Run 1: `outputs/diffusion/worker_mf2_s42/screw-thread_side/`
- Models: `outputs/diffusion/p1_anomaly_core_s42/`, `outputs/diffusion/p1_masks_s42/`; job logs `jobs/logs/`

## 7. Reproduce

```bash
# project env, GPU node, clean checkout of this branch
python scripts/generate_diffusion.py --category screw --defect-type thread_side \
  --num-samples 16 --seed 42 --purpose validation --out <new dir outside Git>
python scripts/diffusion/analysis/run_summary.py <that dir>
```

A formal bundle uses `--num-samples 25 --purpose formal`, one bundle per in-scope defect type.
Raw worker runs go to `<work_dir>/<purpose>/<category>_<defect>/seed<seed>`, so validation and
formal runs with the same seed never share a directory. (The two runs in section 4 predate this
and sit directly under `<work_dir>/`.)

## 8. Formal run (seed 42, 8 types x 25)

SLURM jobs 43375 (screw/thread_side, screw/scratch_head) and 43376 (other 6 types), partition
MGPU-TC2, QoS normal, run one after the other (1 GPU per user). Code `6e2d490`, recorded in
every bundle with `code_dirty: false` and `purpose: formal`. Stage-1 budget 130 x, `placement_v2`,
`mask_filter_v2`, shared QC `synthetic-qc-v1`; `--num-samples 25 --seed 42 --purpose formal`.
Every bundle is kept; nothing was rerun or selected.

| Type | Status | Worker calls | Stage-1 attempts / budget | Shared QC accepted / attempted | Generation s | Model load s |
|---|---|---|---|---|---|---|
| metal_nut/bent | ok | 1 | 56 / 3250 | 25 / 25 | 212.3 | 155.9 |
| metal_nut/color | ok | 1 | 225 / 3250 | 25 / 25 | 635.9 | 62.7 |
| metal_nut/scratch | ok | 6 | 186 / 4680 | **25 / 36** | 620.8 | **382.0** |
| screw/manipulated_front | ok | 1 | 147 / 3250 | 25 / 25 | 443.0 | 65.3 |
| screw/scratch_head | ok | 1 | 645 / 3250 | 25 / 25 | 1651.2 | 63.3 |
| screw/scratch_neck | ok | 1 | 102 / 3250 | 25 / 25 | 328.0 | 65.1 |
| screw/thread_side | ok | 1 | 1025 / 3250 | 25 / 25 | 2582.4 | 72.6 |
| screw/thread_top | ok | 1 | 44 / 3250 | 25 / 25 | 193.3 | 65.4 |
| **Total** | 8/8 ok | 13 | 2430 / 27430 | 200 / 211 | **6666.9** | **932.3** |

Generation s includes all rejected stage-1 attempts and QC-rejected pairs; model loads are
separate (`diffusion_internal.json`, `totals`). Mean per accepted pair: **33.3 s**
(**38.0 s** including model loads), over 200 pairs. No type underfilled.

**The 130 x budget did not affect the formal pool.** Every one of the 13 worker calls stayed
within 50 x its request (closest: screw/thread_side, 1025 of 1250). Stage 1 stops as soon as
the request is filled, so a 50 x cap would not have stopped any call.

**metal_nut/scratch: shared QC 25/36, 11 x `mask_too_large`.** The diffusion-internal
`too_large` rule limits mask area as a share of the whole 256 x 256 image (<= 4 x the largest
support mask, `mask_filter_v2`). Shared QC limits mask area relative to the estimated object
foreground (<= 0.20, `src/generation/qc.py`). A mask can pass the first and fail the second.
The adapter refilled the 11 rejected pairs with 5 more worker calls (requests 5, 3, 1, 1, 1),
each with its own model load. That raised the load time to 382 s, against 63–73 s for the
other single-call types. (metal_nut/bent, the first call of job 43376, loaded in 156 s.)
36 attempts are within the protocol cap of 250 per type. No rule was changed.

**Known issue: thread_side mask fragments on the background.** In the thread_side contact sheet
(e.g. row #0), small mask fragments lie on the background next to the screw. `placement_v2`
only requires >= 95 % of the mask area inside the placement foreground, and thread_side masks
are fragmented (§5 and `docs/proposals/placement_v3.md` §7), so a few small fragments can
lie outside. Recorded, not filtered.

Outputs on TC2 (home `/home/mcaai/yuchen027`, outside Git):

- Bundles: `outputs/diffusion/formal_s42/<category>_<defect>/` (`batch.npz`, `metadata.json`,
  `provenance.json`, `qc_report.json`, `diffusion_internal.json`, `summary.json`)
- Raw worker runs: `outputs/diffusion/raw_runs/formal/<category>_<defect>/seed42/`
  (request, result, worker log, foregrounds, `rejects/`, `rejects.json`). For metal_nut/scratch,
  the 5 refill calls are in `seed43/` to `seed47/` (`generate_with_qc` passes seed 42 + call number).
- Contact sheets, all 25 pairs per type, none selected:
  `outputs/diffusion/formal_s42/contact_sheets/<category>_<defect>.png`. 7 types were made with
  `make_contact_sheet.py` on a copy of the single worker run dir, after checking pixel by pixel
  that it holds exactly the bundle's 25 pairs. metal_nut/scratch was made with
  `bundle_contact_sheet.py`, because its worker dirs also hold the QC-rejected pairs.
- Job scripts and logs: `jobs/p3_formal_body.sh`, `jobs/p3_formal_{slow,rest}.sbatch`,
  `jobs/logs/formal_slow_43375.out`, `jobs/logs/formal_rest_43376.out`

## 9. Inpaint sampling-steps study (validation only)

**Design.** SLURM job 43442, code `60c4969` (`code_dirty: false`, `purpose: validation`; never a
formal pool). metal_nut/bent and screw/scratch_neck, 8 pairs each, seed 3042. Only
`sampling.image_ddim_steps` varies: 50, 100, 200. Everything else is as in the formal run (mask
steps 100, budget 130 x, `mask_filter_v2`, `placement_v2`, shared QC). The three configs are
copies of `configs/diffusion/generation.json` outside Git
(`~/outputs/diffusion/steps_study/configs/generation_steps{50,100,200}.json`); they differ from it
only in `image_ddim_steps` and `work_dir`, which the job checked before running. All 6 bundles
`ok`, shared QC 8/8, 1 worker call each. **Masks and sources are pixel-identical across the three
step counts** (0 differing pixels), so the step count is the only variable. This is expected
from the code: stage 1 completes before any inpainting, and shared QC reads only the mask and
the source foreground.

**Time** (same node, NVIDIA A40; seconds):

| | 50 steps | 100 steps | 200 steps |
|---|---|---|---|
| Inpaint per pair | 0.74 | 1.40 | 2.74 |
| Generation per accepted pair, metal_nut/bent | 5.50 | 6.17 | 7.50 |
| Generation per accepted pair, screw/scratch_neck | 10.31 | 11.02 | 12.35 |

Stage-1 time is the same at every step count (bent 38 s, scratch_neck 77 s for 8 pairs).
Per-pair costs with model loads are not compared: the first run of the job (bent, 50 steps)
loaded from a cold cache (94 s against about 60 s for the others).

**Pixel difference from the 200-step result** (mean absolute difference, 0–255, averaged over
RGB). It only describes how different the outputs are; **it is not a quality measure**.

| | 50 steps, inside mask | 50 steps, outside mask | 100 steps, inside mask | 100 steps, outside mask |
|---|---|---|---|---|
| metal_nut/bent | 21.0 | 3.6 | 17.2 | 2.5 |
| screw/scratch_neck | 31.8 | 1.9 | 20.2 | 1.4 |

Even at 100 steps the defect inside the mask is a different result, not a close approximation
of the 200-step one. The non-zero difference outside the mask matches §5.1: inpainting edits
whole 8 px latent cells.

**Visibility (blind).** `zoom_review.py` (unchanged) was run on a copy per type with all 24 rows
(8 pairs x 3 step counts) shuffled with a fixed seed and renumbered; the sheets show only the
row number. Key: `~/outputs/diffusion/steps_study/blind_key.json`. One annotator (yuchen)
labelled all 48 rows on the delivery-doc scale (V / B / N). Labels:
`docs/diffusion_evidence/labels_yuchen.json`, SHA-256
`323183f3c85165a03140b5873dcaf4f223e4d6f218b27a2a9005c50470988fa0`. The labels were given in
full in one message before the key was opened. The first saved copy
(`steps_study/blind_labels.json`) was written immediately before unblinding, in the same script.
`labels_yuchen.json` was written afterwards with identical labels, so its hash proves the content,
not the time of writing.

| | 50 steps | 100 steps | 200 steps |
|---|---|---|---|
| metal_nut/bent (V / B / N) | 5 / 3 / 0 | 3 / 5 / 0 | 4 / 4 / 0 |
| screw/scratch_neck (V / B / N) | 6 / 2 / 0 | 4 / 4 / 0 | 3 / 5 / 0 |

Same label at all three step counts: bent 6/8 pairs, scratch_neck 5/8. Differing pairs: bent #6
(V/B/V for 50/100/200), #7 (V/B/B); scratch_neck #4 (V/V/B), #6 (V/B/B), #7 (V/B/B). Paired
50 vs 200: bent 1 pair better at 50 and 0 worse (sign test p = 1.0); scratch_neck 3 better and
0 worse (p = 0.25). No evidence that fewer steps reduce visibility; the small lead for 50 steps
is not significant.

**Conclusion.** Lowering the inpaint steps saves little: 200 -> 50 saves about 2 s per pair,
27 % of the generation cost per accepted pair for bent and 17 % for scratch_neck, and less for
low-yield types (e.g. thread_side, 103 s per accepted pair in §8). The cost is driven by the
stage-1 yield, not by inpainting. The formal pool stays at 200 steps; no change is proposed.

**Limits.** One annotator only. The zoom crop is centred on the mask, and the masks are
identical, so the three versions of a pair show the same crop: the annotator could tell which
rows belong to the same pair, though not which step count each row had. 8 pairs per type and
two types only.

Outputs on TC2 (outside Git): bundles `~/outputs/diffusion/steps_study/<category>_<defect>_steps<N>/`,
raw runs `~/outputs/diffusion/steps_study/raw_steps<N>/validation/`, blind sheets
`~/outputs/diffusion/steps_study/blind/<category>_<defect>/zoom_review.png`, job script
`~/jobs/p4_steps_study.sbatch`, log `~/jobs/logs/steps_study_43442.out`.

## 10. Failure taxonomy

Read-only analysis of existing evidence (formal run §8, validation runs §4, steps study §9);
nothing was rerun. **No rule was changed in this round**; every "direction" below is a
possible next step, not implemented.

### 10.1 Stage-1 rejections in the formal run

Source: `totals.rejected_by_reason` in each bundle's `diffusion_internal.json`. "Produced" is
the number of pairs sent to shared QC (metal_nut/scratch: 36, of which 25 passed, §8).
Stage-1 yield = produced / attempts.

| Type | Attempts | Produced | Yield | check_mask | placement_failed | too_small_or_empty | dirty_bg | inverted | latent_too_small | too_large |
|---|---|---|---|---|---|---|---|---|---|---|
| metal_nut/bent | 56 | 25 | 44.6 % | 9 | 11 | 0 | 0 | 5 | 3 | 3 |
| metal_nut/color | 225 | 25 | 11.1 % | 174 | 25 | 0 | 0 | 0 | 1 | 0 |
| metal_nut/scratch | 186 | 36 | 19.4 % | 22 | 87 | 0 | 0 | 35 | 6 | 0 |
| screw/manipulated_front | 147 | 25 | 17.0 % | 14 | 5 | 33 | 30 | 0 | 40 | 0 |
| screw/scratch_head | 645 | 25 | 3.9 % | 431 | 57 | 62 | 56 | 0 | 8 | 6 |
| screw/scratch_neck | 102 | 25 | 24.5 % | 57 | 7 | 4 | 1 | 3 | 3 | 2 |
| screw/thread_side | 1025 | 25 | 2.4 % | 118 | 415 | 206 | 161 | 60 | 29 | 11 |
| screw/thread_top | 44 | 25 | 56.8 % | 2 | 1 | 13 | 1 | 0 | 2 | 0 |
| **Total** | **2430** | **211** | **8.7 %** | **827** | **608** | **318** | **249** | **103** | **92** | **22** |

**Where the cost goes.** Stage 1 is 90 % of the generation time (5977 of 6667 s; thread_side
97 %, scratch_head 95 %). Every stage-1 attempt costs about the same sampling time whatever its
fate, so each reason's share of attempts approximates its share of the stage-1 cost:
`check_mask` 34 % of all attempts (827), `placement_failed` 25 % (608), `too_small_or_empty`
13 %, `dirty_bg` 10 %, `inverted` 4 %, `latent_too_small` 4 %, `too_large` 1 %. The two largest
sources are concentrated: `check_mask` mostly in scratch_head (431) and metal_nut/color (174),
`placement_failed` mostly in thread_side (415). Those three types use 78 % of all attempts.

**Examples** (first 6 rejected masks per reason in attempt order, no selection; each tile is
raw grey sample | judged mask). Made with `scripts/diffusion/analysis/rejects_sheet.py` (unchanged,
k = 6) on copies of the formal raw runs, so the raw runs were not written to:
`~/outputs/diffusion/formal_s42/rejects_sheets/screw_thread_side.png` and
`~/outputs/diffusion/formal_s42/rejects_sheets/screw_scratch_head.png` (outside Git).

Directions: `check_mask` — find out why the scratch_head and metal_nut/color mask models
produce masks covering the image border (mask-model training or guidance). `placement_failed` —
rotation-aligned placement (`docs/proposals/placement_v3.md`, not adopted: it would admit more
fragmented masks).

### 10.2 Problems that remain in accepted pairs

**a) Shared QC and the internal area rule disagree.** metal_nut/scratch: shared QC 25/36,
11 x `mask_too_large` (§8). The internal `too_large` rule limits mask area as a share of the
whole image (<= 4 x the largest support mask, `mask_filter_v2`); shared QC limits it relative
to the object foreground (<= 0.20, `src/generation/qc.py`). The 11 rejected pairs cost 5 extra
worker calls and 382 s of model loads. Evidence: `formal_s42/metal_nut_scratch/qc_report.json`,
`diffusion_internal.json`. Rule not changed in this round. Direction: an object-relative area
check inside the generator (a future `mask_filter_v3`), so these masks are rejected before
inpainting.

**b) Masks are more fragmented than the real defects.** Connected components per mask
(8-connectivity, 256 px), formal pool vs the 5 support masks of the same type:

| Type | Support masks (median; values) | Formal pool, 25 masks (median; range) |
|---|---|---|
| metal_nut/bent | 1; 1,1,1,2,2 | 2; 1–4 |
| metal_nut/color | 1; 1,1,1,2,2 | **6; 1–14** |
| metal_nut/scratch | 1; 1,1,1,2,3 | 2; 1–5 |
| screw/manipulated_front | 1; all 1 | 2; 1–4 |
| screw/scratch_head | 1; all 1 | 2; 1–10 |
| screw/scratch_neck | 1; all 1 | 2; 1–7 |
| screw/thread_side | 2; 1,1,2,3,3 | **7; 3–19** |
| screw/thread_top | 1; all 1 | 3; 1–6 |

Every type is more fragmented than its support masks; thread_side and metal_nut/color most.
Rule not changed in this round (accepted as a known limit, see the proposal §7).
Direction: a component-count rule in `mask_filter_v3` (large yield cost, proposal §7) or
mask-model changes.

**c) Mask fragments on the background.** `placement_v2` needs >= 95 % of the mask inside the
placement foreground (the shared-QC foreground minus the shadow halo), so up to 5 % may lie
outside it. Share of each accepted mask's area outside:

| Type | Outside placement foreground: masks with any / max | Outside shared-QC foreground: masks with any / max |
|---|---|---|
| metal_nut/bent | 18/25 / 4.8 % | 18/25 / 4.0 % |
| metal_nut/color | 16/25 / 4.9 % | 15/25 / 4.0 % |
| metal_nut/scratch | 22/25 / 5.0 % | 22/25 / 4.7 % |
| screw/manipulated_front | 14/25 / 4.7 % | 1/25 / 1.1 % |
| screw/scratch_head | 12/25 / 4.9 % | 0/25 / 0 % |
| screw/scratch_neck | 10/25 / 4.8 % | 1/25 / 0.8 % |
| screw/thread_side | 19/25 / 4.9 % | 4/25 / 4.7 % |
| screw/thread_top | 12/25 / 2.9 % | 1/25 / 0.4 % |

No mask is more than 5 % outside either foreground. On screw, most of the outside area lies on
the shadow halo: it is outside the placement foreground but inside the shared-QC foreground.
This is the halo visible around the screw in the thread_side contact sheet (§8). Shared-QC
foreground: `estimate_object_foreground` on the source with the formal loader's convention;
placement foreground: `foregrounds/` in each raw run. Rule not changed in this round.
Direction: require 100 % inside the placement foreground, or drop fragments outside it (this
changes the defect shape and needs its own validation).

**d) The edited region extends beyond the mask.** Upstream inpaints whole 32 x 32-latent cells
(8 x 8 px), so the edit is larger than the saved pixel mask (§5.1; run 2 annotators saw it in
#2, #10, #12, #13, `labels_claude_pr_s42.json`). §9 gives a quantitative sign: outside the mask,
the 50- and 100-step results differ from the 200-step result by 1.4–3.6 (0–255). If the outside
region were untouched, it would equal the source in every run and that difference would be 0.
Saved masks therefore under-label part of the change. Rule not changed in this round.
Direction: snap the mask to its latent cells before inpainting, so the label equals the
editable region (§5.1).

**e) Labelled pairs whose defect is not clearly visible.** Same annotator (yuchen), same scale:

| | Not V (B or N) | N |
|---|---|---|
| thread_side, validation run 2 (§4) | 8/16 = 50 % | 2/16 |
| bent + scratch_neck, steps study (§9) | 23/48 = 48 % | 0/48 |

(With both annotators, 6/16 thread_side pairs got no V from either, §4.) The share of pairs that
are visible only when compared with the source (B) is about the same, around half, in all three
types, so this is not specific to thread_side. Pairs with no visible defect (N) appeared only in
thread_side, but 2/16 vs 0/48 is weak evidence (Fisher p = 0.06), and the steps-study rows are
3 versions of 8 pairs per type, not 48 independent pairs. Recorded, not filtered (the agreed
rule for this round). Direction: more annotators and pairs per type before any visibility
filter; the automatic visibility measure is not validated (§4).

