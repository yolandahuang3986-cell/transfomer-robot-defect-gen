# Secondary ratio analysis: metal_nut, Procedural, r=1/5/10 — 19 October 2026

Per `docs/formal_evaluation_protocol.md` §4: "A's ratio analysis... r=1 uses the
primary pool's deterministic stratified prefix, r=10 keeps the primary pool and
appends." metal_nut only, scoped first, per the agreed decision of 18 October.

## A construction bug was caught and corrected same-day

The first attempt (18 October, **not** in this directory — never committed)
generated r=1 (15 samples) and r=10 (150 samples) by independently re-invoking
`generate_procedural.py` with `--seed 42` at each target count. This was found
**not** to be a prefix/superset of the already-accepted primary pool (75
samples, seed 42): a shared-QC rejection partway through a run shifts the RNG
state for every sample generated afterward, so a different `--num-samples` does
not reproduce the same sequence past the first rejection.

Direct comparison against the primary pool's own metadata: the independent r=1
matched the primary pool's first 15 entries on 14/15 (diverging only on the
last); the independent r=10 matched the primary pool's first 75 on 67/75
(diverging from #68 onward, i.e. all of entries 68-75 and all 75 "new" entries
75-149 are not comparable to a clean quantity-only manipulation). That result
was never committed to this repo; it was shown in the OneDrive report draft
for a few hours on 19 October, explicitly marked `[CORRECTION IN PROGRESS]`,
then replaced with the corrected numbers below once they were ready.

## Corrected construction

- **r=1 (15 pairs)**: a literal, byte-identical prefix of the already-accepted
  primary pool's own `batch.npz`/`metadata.json` (first 15 of 75 entries). No
  regeneration. See `docs/member_a_delivery.md` §8 for the build method.
- **r=5 (75 pairs)**: the primary pool itself, completely unchanged (not rerun
  here — already accepted and evaluated as part of E1, `results/metrics/e1_metal_nut_20261017/`).
- **r=10 (150 pairs)**: the primary pool (75) kept unchanged, plus 75 newly
  generated pairs appended from a second bundle under a disclosed, non-primary
  seed (142, chosen only to avoid colliding with the primary pool's own output
  directory on TC2). 80 attempts for 75 QC passes (93.75%).

This required a small code change (`ratio_append_seeds` / `--ratio-append-seed`,
PR #15) to let `load_formal_pool` admit a bundle whose provenance records a
disclosed non-primary seed, only in ratio-analysis mode. PR #15's own real-data
regression check confirmed this does not affect any already-accepted primary
pool; this ratio analysis is further confirmation in actual use.

Both tiers' acceptance: automated checks (checksum, source/leakage, shared-QC
recheck) via `scripts/audit_formal_pool.py --ratio-tier-count ... [--ratio-append-seed 142]`,
then a reduced human-audit pass (per the agreed secondary-analysis process:
5-8 pairs instead of the primary 20) by A. Full acceptance records (outside Git,
hash-bound): `review_r1_v2/acceptance.json`, `review_r10_v2/acceptance.json` in
A's local review archive, synced from TC2.

## Results

Jobs **44078** (r=1, exit 0:0, 5:06) and **44079** (r=10, exit 0:0, 3:57), both
NVIDIA A40, code commit `ed5a727` (includes PR #15), seeds 42/43/44, 1,500
optimizer steps, evaluated on the same frozen 77-image metal_nut test set as E0/E1/E3.

| Ratio (n) | Pixel AUROC | Δ vs E0 | Dice | IoU |
|---|---:|---:|---:|---:|
| r=1 (15, true prefix) | 0.6918 ± 0.1208 | +0.0360 ± 0.0046 | 0.0190 ± 0.0305 | 0.0097 ± 0.0157 |
| r=5 (75, primary, unchanged) | 0.8202 ± 0.0995 | +0.1644 ± 0.2171 | 0.0591 ± 0.0644 | 0.0312 ± 0.0345 |
| r=10 (150, primary + seed-142 append) | 0.8000 ± 0.1298 | +0.1442 ± 0.2139 | 0.0429 ± 0.0383 | 0.0222 ± 0.0199 |

Pixel AUROC rises from r=1 to r=5, then dips slightly at r=10 rather than
continuing to improve — diminishing, not negative, returns in this range.
r=10's paired gain (+0.1442 ± 0.2139) is close to r=5's (+0.1644 ± 0.2171) and
still noisy across seeds; r=1's gain is smaller but markedly more consistent
(± 0.0046). Dice/IoU fall below E0 at all three ratios, least so at r=10.
Recall and FPR at r=10 are driven by one seed (43: 0.84 recall, 0.09 FPR vs the
other two seeds' saturated 1.0/1.0), a single-seed effect, not a general
pattern. Three ratio points and three seeds per point do not establish
statistical significance or a confirmed optimum.

Diffusion's side of this analysis is unaffected by the construction bug (it was
never run) and is now clearer for C: r=1 needs no new generation (buildable
directly from her already-accepted bundles); r=10 only needs the 25/type append
portion under a disclosed fresh seed, not a full 50/type regeneration.

## Evidence

- [r=1 seed 42](ratio_r1v2_procedural_metal_nut_seed42.json) /
  [43](ratio_r1v2_procedural_metal_nut_seed43.json) /
  [44](ratio_r1v2_procedural_metal_nut_seed44.json)
- [r=10 seed 42](ratio_r10v2_procedural_metal_nut_seed42.json) /
  [43](ratio_r10v2_procedural_metal_nut_seed43.json) /
  [44](ratio_r10v2_procedural_metal_nut_seed44.json)
- [r=1 run plan](run_plan_r1.json) / [CUDA Gate 0](gate0_r1.json)
- [r=10 run plan](run_plan_r10.json) / [CUDA Gate 0](gate0_r10.json)

Checkpoints and detailed logs remain in A's TC2 result directories
(`~/robot-results/eratio{1,10}v2_procedural_metal_nut_20261019_v1/`). The
accepted pools and their hash-bound `acceptance.json` records are kept outside
Git in A's local review archive, synced from the OneDrive project review folder.
