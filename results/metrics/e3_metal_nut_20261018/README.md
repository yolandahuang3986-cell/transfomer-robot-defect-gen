# E3: Real + Diffusion, metal_nut

TC2 job **43672** completed successfully on **NVIDIA A40**, with exit code **0:0**
and a total elapsed time of **4 minutes 2 seconds**, including GPU preflight. All
three runs used clean main commit `dd653319958f875f209885aabe8d81e6912b77e6`
(current main at the time of acceptance, includes PR #9's diffusion stage-1 fix;
`src/inspection/formal_bundle.py` and `scripts/train_inspector.py` are byte-identical
to the `77ebf7f` commit used for E0/E1, so the downstream training/eval code is
unchanged across all three experiments).

## Protocol and validation

- Frozen protocol `v1.0-frozen`; downstream seeds 42/43/44.
- 235 real training images plus the accepted AnomalyDiffusion pool: 3 in-scope
  metal_nut defect types (bent, color, scratch), 25 pairs each, 75 total.
- 1,500 optimizer steps per seed; batch 8; 256-pixel inputs; final-step checkpoint.
- Evaluation on 77 real test images (22 normal, 55 held-out anomalies), excluding flip.
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
| Pixel AUROC | 0.6559 ± 0.1177 | 0.8202 ± 0.0995 | 0.8306 ± 0.0321 |
| Dice | 0.0821 ± 0.1391 | 0.0591 ± 0.0644 | 0.2198 ± 0.2824 |
| IoU | 0.0467 ± 0.0792 | 0.0312 ± 0.0345 | 0.1443 ± 0.1970 |
| Anomalous-image recall | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9879 ± 0.0210 |
| Normal-image FPR | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9242 ± 0.1312 |

Paired delta, E3 − E0: Pixel AUROC +0.1748 ± 0.1140 (3/3 seeds improve; substantially
lower cross-seed spread than E1's +0.1644 ± 0.2171); Dice +0.1377 ± 0.3513 and IoU
+0.0976 ± 0.2371 (both driven mostly by one seed, 43, which jumps from near-zero to
0.54 Dice — not a consistent per-seed gain); recall −0.0121 ± 0.0210 and FPR
−0.0758 ± 0.1312 (small, one-seed-driven decreases from E0's saturated 1.0).

The average AUROC gain is consistent across seeds and larger in mean than Procedural's,
but the Dice/IoU gain is not: two of three seeds show little or negative pixel-overlap
change, so this is a descriptive ranking result for one category and one fixed pool,
not evidence of a reliable segmentation improvement or an established Procedural-vs-
Diffusion ranking. No protocol was changed after observing these results.

## Evidence

- [Seed 42](diffusion_metal_nut_seed42.json)
- [Seed 43](diffusion_metal_nut_seed43.json)
- [Seed 44](diffusion_metal_nut_seed44.json)
- [CUDA Gate 0](gate0.json)
- [Training-only timing](timing.json)
- [Run plan and command record](run_plan.json)
- [Formal generation record and failure taxonomy](https://github.com/yolandahuang3986-cell/transfomer-robot-defect-gen/blob/main/docs/member_c_delivery.md)
- [Leakage evidence PR](https://github.com/yolandahuang3986-cell/transfomer-robot-defect-gen/pull/11)

Checkpoints and detailed logs remain in A's TC2 result directory
(`~/robot-results/e3_diffusion_metal_nut_20261018_v1/`). The accepted pool and its
hash-bound `acceptance.json` are kept outside Git in A's local review archive
(`c_diffusion_formal_20261017_received_20261018/review_metal_nut/`), synced from the
OneDrive project review folder.
