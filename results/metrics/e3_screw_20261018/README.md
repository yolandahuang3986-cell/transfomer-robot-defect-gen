# E3: Real + Diffusion, screw

TC2 job **43675** completed successfully on **NVIDIA A40**, with exit code **0:0**
and a total elapsed time of **5 minutes 48 seconds**, including GPU preflight. All
three runs used clean main commit `dd653319958f875f209885aabe8d81e6912b77e6`
(same commit as E3 metal_nut; `src/inspection/formal_bundle.py` and
`scripts/train_inspector.py` are byte-identical to the `77ebf7f` commit used for
E0/E1, so the downstream training/eval code is unchanged across all experiments).

## Protocol and validation

- Frozen protocol `v1.0-frozen`; downstream seeds 42/43/44.
- 345 real training images plus the accepted AnomalyDiffusion pool: 5 in-scope screw
  defect types (manipulated_front, scratch_head, scratch_neck, thread_side,
  thread_top), 25 pairs each, 125 total.
- 2,200 optimizer steps per seed; batch 8; 256-pixel inputs; final-step checkpoint.
- Evaluation on 135 real test images.
- CUDA Gate 0 passed. No dataset, generated batch or model weights are included here.
- Model settings, step budgets, frozen evaluation partition and recorded package
  versions match E0/E1.
- No seed selection, threshold tuning, extra training or protocol changes were made
  after observing results.

## Results

Values are mean ± sample SD across the three downstream seeds. The synthetic pool
is fixed; this SD does not measure generator variability. Thresholded metrics use 0.5.

| Metric | E0: real only | E1: real + Procedural | E3: real + Diffusion |
|---|---:|---:|---:|
| Pixel AUROC | 0.5201 ± 0.2718 | 0.6913 ± 0.1515 | 0.5705 ± 0.2347 |
| Dice | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| IoU | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Anomalous-image recall | 0.0000 ± 0.0000 | 0.3333 ± 0.5774 | 0.0284 ± 0.0325 |
| Normal-image FPR | 0.2846 ± 0.4929 | 0.3333 ± 0.5774 | 0.2927 ± 0.5069 |

Paired delta, E3 − E0: Pixel AUROC +0.0504 ± 0.0558 — a small average gain, clearly
smaller than Procedural's +0.1712 ± 0.1464 on the same category. Dice and IoU stay at
0 for both E0 and E3 at the fixed threshold (the screw real-only baseline is already
degenerate, recorded as-is since E0). Recall rises only slightly, from 0 to a mean of
0.0284 (a handful of positive pixels in at most one anomalous image per seed, not a
reliable detection change); FPR is essentially unchanged.

Screw shows the smallest synthetic-data effect of the four E1/E3 runs completed so
far. This does not establish that Diffusion is worse than Procedural in general: one
fixed pool per method, three seeds and one category do not support a significance
claim or an overall method ranking. The known screw-specific diffusion limitations
(mask fragmentation, the shadow-halo foreground overlap, and a few samples with no
clearly visible defect — see `docs/member_c_delivery.md` §5/§10) are candidate,
not confirmed, explanations for the smaller gain.

## Evidence

- [Seed 42](diffusion_screw_seed42.json)
- [Seed 43](diffusion_screw_seed43.json)
- [Seed 44](diffusion_screw_seed44.json)
- [CUDA Gate 0](gate0.json)
- [Training-only timing](timing.json)
- [Run plan and command record](run_plan.json)
- [Formal generation record and failure taxonomy](https://github.com/yolandahuang3986-cell/transfomer-robot-defect-gen/blob/main/docs/member_c_delivery.md)
- [Leakage evidence PR](https://github.com/yolandahuang3986-cell/transfomer-robot-defect-gen/pull/11)

Checkpoints and detailed logs remain in A's TC2 result directory
(`~/robot-results/e3_diffusion_screw_20261018_v1/`). The accepted pool and its
hash-bound `acceptance.json` are kept outside Git in A's local review archive
(`c_diffusion_formal_20261017_received_20261018/review_screw/`), synced from the
OneDrive project review folder.
