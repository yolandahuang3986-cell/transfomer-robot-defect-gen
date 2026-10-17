# Member C delivery: AnomalyDiffusion generator (validation stage)

Status on 2026-10-14: code path complete and run end to end on TC2 for one defect type
(screw/thread_side, 16 validation samples). **No formal pool has been generated** and
no downstream (E3) result exists. The 16 samples are evidence only and must not enter
any formal pool.

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
