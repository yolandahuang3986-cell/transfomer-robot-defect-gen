# E1: Real + Procedural, screw

TC2 job **43671** completed successfully on **NVIDIA A40**, with exit code **0:0**
and a total elapsed time of **6 minutes 19 seconds**, including GPU preflight.
All three runs used clean main commit `77ebf7f5870cc21a97d98373c2f03ac83871219b`
(the same code commit as the accepted metal_nut E1 run, for comparability).

## Protocol and validation

- Frozen protocol `v1.0-frozen`; downstream seeds 42/43/44.
- 345 real training images plus the accepted 125-pair Procedural pool
  (`screw_dark_core_colored_polygon_v4`, acceptance recorded 2026-10-18).
- 2,200 optimizer steps per seed; batch 8; 256-pixel inputs; final-step checkpoint.
- Evaluation on 135 real test images, excluding flip (not applicable to screw).
- CUDA Gate 0 passed. No dataset, generated batch or model weights are included here.
- E0/E1 model settings, step budgets, frozen evaluation partition and recorded package
  versions match. The training/model/data/metric implementations are unchanged from the
  E0 and metal_nut E1 code commit.
- No seed selection, threshold tuning, extra training or protocol changes were made
  after observing results.

## Results

Values are mean ± sample SD across the three downstream seeds. The synthetic pool
is fixed; this SD does not measure generator variability. Thresholded metrics use 0.5.

| Metric | E0: real only | E1: real + Procedural | Paired delta (E1 − E0) |
|---|---:|---:|---:|
| Pixel AUROC | 0.5201 ± 0.2718 | 0.6913 ± 0.1515 | 0.1712 ± 0.1464 |
| Dice | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| IoU | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Anomalous-image recall | 0.0000 ± 0.0000 | 0.3333 ± 0.5774 | 0.3333 ± 0.5774 |
| Normal-image FPR | 0.2846 ± 0.4929 | 0.3333 ± 0.5774 | 0.0488 ± 0.0845 |

Pixel AUROC rises on average (2 of 3 seeds improve, one is roughly flat), while Dice,
IoU and recall stay at or near 0 at the fixed 0.5 threshold for both E0 and E1: the
screw real-only baseline is already degenerate (it predicts almost no positive pixels
at this threshold, recorded as-is since E0), and Procedural training does not change
that at the pixel-overlap level. Anomalous-image recall moves from 0/3 to 1/3 seeds
detecting at least one positive pixel per anomalous image — a small, single-seed
change, not a reliable detection improvement. Three seeds and one category do not
establish statistical significance or an overall method ranking.

## Evidence

- [Seed 42](procedural_screw_seed42.json)
- [Seed 43](procedural_screw_seed43.json)
- [Seed 44](procedural_screw_seed44.json)
- [CUDA Gate 0](gate0.json)
- [Training-only timing](timing.json)
- [Run plan and command record](run_plan.json)
- [Acceptance review (screw_dark_core_colored_polygon_v4)](https://github.com/yolandahuang3986-cell/transfomer-robot-defect-gen/pull/10)

Checkpoints and detailed logs remain in A's TC2 result directory
(`~/robot-results/e1_procedural_screw_20261018_v1/`). The accepted pool and its
hash-bound `acceptance.json` are kept outside Git in A's local review archive
(`b_screw_revised_20261017_received_20261018/`), synced from the OneDrive project
review folder.
