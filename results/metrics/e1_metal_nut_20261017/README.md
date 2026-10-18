# E1: Real + Procedural, metal_nut

TC2 job **43178** completed successfully on **NVIDIA A40**, with exit code **0:0**
and a total elapsed time of **4 minutes 45 seconds**, including GPU preflight.
All three runs used clean main commit `77ebf7f5870cc21a97d98373c2f03ac83871219b`.

## Protocol and validation

- Frozen protocol `v1.0-frozen`; downstream seeds 42/43/44.
- 235 real training images plus the same accepted 75-pair Procedural pool.
- 1,500 optimizer steps per seed; batch 8; 256-pixel inputs; final-step checkpoint.
- Evaluation on 77 real test images (22 normal, 55 held-out anomalies), excluding flip.
- All 88 tests and CUDA Gate 0 passed. No dataset, generated batch or model weights are included here.
- E0/E1 model settings, step budgets, frozen evaluation partition and recorded package versions match.
  The training/model/data/metric implementations are unchanged from the E0 code commit.
- No seed selection, threshold tuning, extra training or protocol changes were made after observing results.

## Results

Values are mean ± sample SD across the three downstream seeds. The synthetic pool
is fixed; this SD does not measure generator variability. Thresholded metrics use 0.5.

| Metric | E0: real only | E1: real + Procedural | Paired delta (E1 − E0) |
|---|---:|---:|---:|
| Pixel AUROC | 0.6559 ± 0.1177 | 0.8202 ± 0.0995 | 0.1644 ± 0.2171 |
| Dice | 0.0821 ± 0.1391 | 0.0591 ± 0.0644 | -0.0231 ± 0.1611 |
| IoU | 0.0467 ± 0.0792 | 0.0312 ± 0.0345 | -0.0155 ± 0.0912 |
| Anomalous-image recall | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Normal-image FPR | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 |

Pixel AUROC increases on average, while mean Dice and IoU decrease at the frozen
0.5 threshold. Normal-image FPR remains 1.0: every normal test image has at least
one predicted positive pixel. The results therefore show better average pixel
ranking for this category, but do not establish improved segmentation or usable
image-level defect detection. Three seeds and one category do not establish an
overall method ranking or statistical significance.

Screw E1 remains on hold for placement quality. GAN and Diffusion have no accepted
formal pools in this comparison. The summary marks these entries incomplete and
does not calculate a two-category Procedural macro score.

## Evidence

- [Seed 42](procedural_metal_nut_seed42.json)
- [Seed 43](procedural_metal_nut_seed43.json)
- [Seed 44](procedural_metal_nut_seed44.json)
- [Summary with paired E0 differences](summary.json)
- [Execution record and file hashes](execution_record.json)
- [CUDA Gate 0](gate0.json)
- [Training-only timing](timing.json)
- [Acceptance review and fixed sheets](../../../docs/reviews/b_delivery_20261017/README.md)

Checkpoints and detailed logs remain in A's TC2 result directory. Acceptance,
legacy-purpose review and the exact run plan are also saved in the OneDrive
project review folder. The acceptance records identify Codex-assisted review and
do not claim an additional independent human visual rating.
