# Secondary ratio analysis: metal_nut, Diffusion, r=1/5/10 — 19 October 2026

Completes the ratio analysis alongside [the Procedural results](../ratio_analysis_metal_nut_20261019/README.md).
Per `docs/formal_evaluation_protocol.md` §4: r=1 is a deterministic stratified
prefix of the primary pool, r=10 keeps the primary pool and appends.

## Construction

- **r=1 (15 pairs, 5/type)**: a literal, byte-identical prefix of C's already-
  accepted primary pool (first 5 of each type's 25 entries). Built directly by
  A from C's already-delivered bundles — **no new generation from C needed**.
- **r=5 (75 pairs, 25/type)**: the primary pool itself, unchanged (already
  accepted and evaluated as E3, `results/metrics/e3_metal_nut_20261018/`).
- **r=10 (150 pairs, 50/type)**: the primary pool (25/type) kept unchanged,
  plus 25/type newly generated and appended by C under a disclosed, non-primary
  seed (142, to avoid colliding with the primary pool's own TC2 output
  directory). Delivered 2026-10-19 (SLURM job 44097), with per-type stage-1
  attempts/QC detail in the delivery package; C independently verified zero
  pixel-level duplication between the append pool and the primary pool (and
  within the append pool itself) before delivery.

Loading a type's quota split across two bundles (primary + append) required a
code fix (PR #17): the per-bundle check originally required each bundle to
itself equal the exact per-type quota, which made a split impossible under any
single `--ratio-tier-count` value. C found this while preparing her delivery
and described the exact fix needed; it was implemented, tested, verified
against all 4 real accepted pools, and merged before this analysis ran — its
first real-world use was this exact delivery, and it worked as designed on the
first attempt.

Both tiers' acceptance: automated checks (checksum, source/leakage, shared-QC
recheck, aggregate-count) via `scripts/audit_formal_pool.py --ratio-tier-count
... [--ratio-append-seed 142]`, then a reduced human-audit pass (per the agreed
secondary-analysis process: 5–8 pairs instead of the primary 20, drawn
preferentially from the newly-generated portion) by A. Full acceptance records
(outside Git, hash-bound): `review_diffusion_r1/acceptance.json`,
`review_diffusion_r10/acceptance.json` in A's local review archive, synced
from TC2.

## Results

Jobs **44092** (r=1, exit 0:0, 5:01) and **44176** (r=10, exit 0:0, 3:24), both
NVIDIA A40, seeds 42/43/44, 1,500 optimizer steps, evaluated on the same frozen
77-image metal_nut test set as E0/E1/E3/the Procedural ratio analysis.

| Ratio (n) | Pixel AUROC | Δ vs E0 | Dice | IoU |
|---|---:|---:|---:|---:|
| r=1 (15, A's prefix, no new generation) | 0.7098 ± 0.0549 | +0.0540 ± 0.0629 | 0.0324 ± 0.0501 | 0.0169 ± 0.0263 |
| r=5 (75, primary, unchanged) | 0.8306 ± 0.0321 | +0.1748 ± 0.1140 | 0.2198 ± 0.2824 | 0.1443 ± 0.1970 |
| r=10 (150, primary + C's seed-142 append) | 0.8801 ± 0.0176 | +0.2242 ± 0.1284 | 0.6523 ± 0.0190 | 0.4842 ± 0.0207 |

Unlike Procedural (diminishing returns past r=5), Diffusion's pixel AUROC rises
**monotonically and with shrinking cross-seed spread** across all three ratios
(SD: 0.0549 → 0.0321 → 0.0176). Dice/IoU jump sharply at r=10 (0.65/0.48) — the
largest segmentation gain observed anywhere in this project (compare: the
primary E3 metal_nut Dice was 0.22, E0 was 0.08, every Procedural ratio stayed
below 0.06). This is reported as-is, flagged for attention rather than
explained: one ratio point, three seeds, no isolated mechanism identified. A
plausible but unverified factor is that 150 synthetic pairs (vs 75) gives the
segmentation head enough examples to learn the metal_nut defect geometry more
reliably; this is not tested here.

## Combined comparison (both methods)

| Method | Ratio (n) | Pixel AUROC | Δ vs E0 | Dice | IoU |
|---|---|---:|---:|---:|---:|
| Procedural | r=1 (15) | 0.6918 ± 0.1208 | +0.0360 ± 0.0046 | 0.0190 ± 0.0305 | 0.0097 ± 0.0157 |
| Procedural | r=5 (75, primary) | 0.8202 ± 0.0995 | +0.1644 ± 0.2171 | 0.0591 ± 0.0644 | 0.0312 ± 0.0345 |
| Procedural | r=10 (150) | 0.8000 ± 0.1298 | +0.1442 ± 0.2139 | 0.0429 ± 0.0383 | 0.0222 ± 0.0199 |
| Diffusion | r=1 (15) | 0.7098 ± 0.0549 | +0.0540 ± 0.0629 | 0.0324 ± 0.0501 | 0.0169 ± 0.0263 |
| Diffusion | r=5 (75, primary) | 0.8306 ± 0.0321 | +0.1748 ± 0.1140 | 0.2198 ± 0.2824 | 0.1443 ± 0.1970 |
| Diffusion | r=10 (150) | 0.8801 ± 0.0176 | +0.2242 ± 0.1284 | 0.6523 ± 0.0190 | 0.4842 ± 0.0207 |

The two methods respond differently to more synthetic data in this range: a
descriptive observation from one ratio sweep, one category, three seeds per
point — not a significance claim or a general scaling law.

## Evidence

- [r=1 seed 42](ratio_diffr1_diffusion_metal_nut_seed42.json) /
  [43](ratio_diffr1_diffusion_metal_nut_seed43.json) /
  [44](ratio_diffr1_diffusion_metal_nut_seed44.json)
- [r=10 seed 42](ratio_diffr10_diffusion_metal_nut_seed42.json) /
  [43](ratio_diffr10_diffusion_metal_nut_seed43.json) /
  [44](ratio_diffr10_diffusion_metal_nut_seed44.json)
- [r=1 run plan](run_plan_r1.json) / [CUDA Gate 0](gate0_r1.json)
- [r=10 run plan](run_plan_r10.json)

Checkpoints and detailed logs remain in A's TC2 result directories
(`~/robot-results/eratiodiff{1,10}_diffusion_metal_nut_20261019_v1/`). The
accepted pools and their hash-bound `acceptance.json` records are kept outside
Git in A's local review archive, synced from the OneDrive project review folder.
