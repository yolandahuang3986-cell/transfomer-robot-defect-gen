# Proposal: placement_v3 (rotation-aligned placement)

**Status: DRAFT, for A's approval. Nothing is implemented or frozen. `placement_v2` stays in use.**
Author: Member C. Branch `fix/diffusion-review`.

The design and the adoption criteria (§1–§5) were written before the s2042 result (job 43341);
§6 was added afterwards.

## 1. Problem

screw/thread_side does not fill its target under `placement_v2`. Placement onto the thin
screw core is the largest single reject reason, as it was in runs 1 and 2
(`docs/member_c_delivery.md` §4: 87 % and 88 % of shape-passing masks failed placement).

`placement_v2` translates the mask's bounding-box crop onto the placement foreground and
accepts the first of up to 200 random positions with overlap >= 0.95. It never rotates.

### 1.1 Funnel, validation job 43273, seed 1042 (budget 50 x, 4 samples -> 200)

| Step | Left | Rejected here |
|---|---|---|
| Stage-1 masks judged | 200 | |
| Pass generator checks (check_mask, mask_filter_v2 shape rules) | 94 | 106 (dirty_bg 39, too_small_or_empty 29, check_mask 23, inverted 14, too_large 1) |
| Placed (`placement_v2`) | 18 | 76 `placement_failed` |
| Pass `latent_too_small` | 3 | 15 |

Result: `underfilled_stage1`, 3/4 at 200/200 attempts; yield 1.5 % (95 % CI 0.3–4.3 %).
Placement pass rate among masks that reached placement: 18/94 = 19 %.

Placement failures per target image (the target changes when a sample is filled):

| Target (index, source) | Attempts while active | placement_failed | / attempts | / masks reaching placement |
|---|---|---|---|---|
| 0, `screw/train/good/288.png` | 1–81 (81) | 34 | 42 % | 34/39 = 87 % |
| 1, `127.png` | 82–101 (20) | 10 | 50 % | 10/12 = 83 % |
| 2, `271.png` | 102–119 (18) | 7 | 39 % | 7/11 = 64 % |
| 3, `006.png` | 120–200 (81) | 25 | 31 % | 25/32 = 78 % |

Failures are spread over all four targets, so they are not caused by one bad image.

screw/scratch_head in the same job used 197/200 attempts, mostly rejected by upstream
`check_mask` (125). That is a stage-1 mask-shape problem; **placement_v3 does not address it**.

### 1.2 Mechanism (offline probe, development data only)

Probe: `~/outputs/diffusion/proposals/placement_v3/rot_probe.py`, `rot_probe_fair.py`.
It re-places the 76 saved `placement_failed` masks of s1042 on the same 4 saved placement
foregrounds, CPU only. It uses only train/good foregrounds and generated masks, with no
held-out data.

- **Orientation.** The median angle between the mask's principal axis and the screw's
  principal axis (PCA of pixel coordinates) is 30°; 50 % of the masks are more than 30° off.
- **Equal try budget (2 x 200 tries, 5 RNG seeds):** translation only places 1–2 / 76;
  rotation-aligned (Δ and Δ+180°) places **36–39 / 76**. The gain comes from the
  rotation, not from the extra tries.
- **Where it helps:** elongation (sqrt of the PCA eigenvalue ratio) >= 4: 31.8 / 45 placed
  (mean of 5 seeds); elongation < 4: 6.0 / 31.
- Expected effect on the s1042 funnel, if nothing else changes: placed 18 -> about 56 of
  94. This is an estimate, not a run. The `latent_too_small` rule still applies afterwards.

### 1.3 Findings that limit the claim (for A)

1. **The generated masks are fragmented.** The masks are mostly not one long defect but a
   line of small fragments. Their "elongation" is the spread of those fragments.
   Connected components (8-connectivity, 256 px):
   - `placement_failed` masks: median 11 (p75 14, max 23); none has 1 component;
   - the 3 accepted masks: 11, 6, 9;
   - the 5 thread_side support masks (ground truth): 3, 3, 1, 2, 1.

   So the mask model already produces masks that are far more fragmented than the real
   defects, including the masks accepted today. v3 would let more of these masks through.
   It does not make them more realistic. This is a separate mask-quality question (a
   possible `mask_filter_v3` component rule), and **it is not part of this proposal**.
2. **Target 0 (`288.png`) has holes in its placement foreground** inside the screw body
   (see the example sheet, last row). This adds to its high failure rate (87 %). It is a
   foreground issue, not orientation.
3. **metal_nut** is roughly round. Its foreground axis is ill-defined, so "align to the
   axis" would amount to a random rotation there. See §3, decision D2.

## 2. Evidence (TC2, outside Git)

Raw run: `~/outputs/diffusion/raw_runs/validation/screw_thread_side/seed1042/`
(`result.json`, `rejects.json`, `request.json`, `rejects/`, `foregrounds/`).

Examples. Each `rejects/*_placement_failed_mask.png` is shown with the foreground it failed on:

| Attempt | Mask | Foreground | Elongation | Angle off | Aligned placement |
|---|---|---|---|---|---|
| 92 | `rejects/00092_placement_failed_mask.png` | `foregrounds/127.png` | 20.3 | 79° | placed |
| 127 | `rejects/00127_placement_failed_mask.png` | `foregrounds/006.png` | 7.5 | 76° | placed |
| 87 | `rejects/00087_placement_failed_mask.png` | `foregrounds/127.png` | 8.5 | 69° | placed |
| 5 (counter-example) | `rejects/00005_placement_failed_mask.png` | `foregrounds/288.png` | 4.4 | 17° | not placed (foreground holes) |

Sheet (mask | foreground | overlay at the original position):
`~/outputs/diffusion/proposals/placement_v3/examples_s1042.png`. Per-mask probe output:
`rot_probe_s1042.json`, `rot_probe_fair_s1042.json` (same folder).

## 3. Design

`placement_v3` = `placement_v2` + one rotation step before the translation search.
The foreground variant (`core_thr3_fill_and_qc`) and `min_overlap` 0.95 stay the same.

1. Principal axis of the binary mask θm and of the placement foreground θf (PCA of pixel
   coordinates, angle mod 180°).
2. Rotate the mask by Δ = θm − θf with nearest-neighbour interpolation (stays binary;
   canvas expanded so nothing is cut off), and try the translation search as in v2.
   If no position reaches 0.95, try again at Δ + 180°.
3. If the rotated crop is larger than 256 px in either direction, that orientation fails.
4. Record per sample: `placement_rotation_deg`, the orientation used, tries and overlap.
5. Order of rules unchanged: shape rules (on the unrotated mask) -> placement ->
   `latent_too_small` (on the placed mask). Shared QC unchanged.

Decisions for A (my recommendation first):

- **D1, tries per mask:** 200 per orientation (up to 400 per mask, as in the probe), or
  100 + 100 to keep v2's 200. Placement is CPU-only and cheap next to one stage-1 sample.
  Recommendation: 200 + 200.
- **D2, scope:** screw only, metal_nut keeps v2 (recommended, because metal_nut's axis is
  ill-defined), or all categories.
- **D3, fragmented masks (§1.3.1):** out of scope here; flag it in the report as a known
  limit; decide separately whether to propose `mask_filter_v3`.

Disclosure: rotation is not in upstream AnomalyDiffusion. It changes the orientation
distribution of the placed defects relative to the learned mask model. The alignment rule
was chosen during development after looking at s1042 rejects (train/good foregrounds and
generated masks only).

## 4. Validation plan

1. Unit tests (no GPU): the rotated mask stays binary; area changes by <= 5 % from
   nearest-neighbour rotation; same seed gives the same result; Δ = 0 gives the v2
   result; rotation angle and orientation are recorded; an oversized rotated crop fails
   cleanly.
2. Offline replay (no GPU) on the saved s2042 `placement_failed` masks with the frozen v3
   parameters, to check that the s1042 probe result holds on new data.
3. GPU validation, pre-declared before submission: new seed (not 42, 1042 or 2042), budget
   130 x, 4 pairs per type, all 8 in-scope types (screw types with v3 and, if D2 = screw
   only, metal_nut types with v2 as a no-change control). All bundles kept, no rerun.
4. Report next to the v2 runs: funnel per type, placement pass rate, `latent_too_small`
   share, shared-QC result, components per accepted mask, and an example sheet for a
   visual check that the defects lie on the screw body.

## 5. Adoption criteria (fixed before the v3 validation run)

_Status: C1–C5 are proposed values, not yet confirmed. They must be confirmed by A before
any v3 GPU run and are not changed after it._

v3 is adopted only if **all** of these hold in the GPU validation run of §4.3:

- **C1:** screw/thread_side fills 4/4 within the 130 x budget.
- **C2:** thread_side placement pass rate (placed / masks reaching placement) is at least
  2 x the v2 rate pooled over s1042 and s2042, and the lower 95 % CI bound of the v3 rate
  is above the pooled v2 point estimate.
- **C3:** every type that filled under v2 at 130 x also fills under v3 (no regression);
  shared QC accepts every exported pair, as it does under v2.
- **C4:** the share of placed masks rejected by `latent_too_small` is not more than
  10 percentage points above v2 (pooled), so rotation does not just move the loss
  downstream.
- **C5:** the share of accepted thread_side masks with more than 3 connected components
  (8-connectivity, 256 px; 3 = the maximum over the 5 thread_side support masks) is not
  higher than under v2, pooled over s1042 and s2042. Reason: §1.3.1 shows that rotation
  mainly rescues fragmented masks, and C1–C4 measure pass rates only, so they cannot
  detect a shift towards more fragmented accepted masks.

If any criterion fails, v3 is not adopted and v2 stays. The rotation rule is not re-tuned
on the same runs. Any further change becomes a new proposal with new seeds.

## 6. s2042 results (job 43341, budget 130 x, placement_v2)

Filled in after job 43341. Sections 1–5 were written before this result and are unchanged.
Run: commit `741d235`, `code_dirty: false`, factor 130, seed 2042, 4 pairs per type, elapsed 18 min.

Pre-declared rule (written before the result): both types fill -> formal runs use 130 x;
thread_side still underfilled at 520/520 -> no further budget increase, push this proposal.
**Outcome: both types filled -> formal runs use 130 x.**

| | screw/thread_side | screw/scratch_head |
|---|---|---|
| Status / produced | ok, 4/4 | ok, 4/4 |
| Attempts / budget | 166 / 520 | 130 / 520 |
| Pass generator checks | 81 | 17 (check_mask 85) |
| Placed (v2) | 10 (placement_failed 71) | 5 (placement_failed 12) |
| Pass latent_too_small | 4 (6 rejected) | 4 (1 rejected) |
| Yield (95 % CI, Clopper–Pearson) | 2.4 % (0.7–6.1 %) | 3.1 % (0.8–7.7 %) |
| Shared QC | 4/4 | 4/4 |
| Components per accepted mask | 10, 15, 9, 11 | 2, 4, 1, 2 |

thread_side placement failures per target (active attempts, failed / masks reaching placement):
0 `256.png` 1–10, 4/5; 1 `267.png` 11–20, 3/5; 2 `008.png` 21–127, **50/56**;
3 `153.png` 128–166, 14/15. Unlike s1042, one target (`008.png`) took 107 of the 166 attempts.

Notes:

1. Both runs would also have filled under the old 50 x budget (166 and 130 < 200). This run
   alone does not show that 130 x was needed. Pooled over s1042 and s2042 (v2), thread_side
   yields 7/366 = 1.9 %. At that rate, P(fewer than 4 of 4 | 200 attempts) = 0.47 and
   P(... | 520 attempts) = 0.01 (binomial). For scratch_head (8/327 = 2.4 %) the values are
   0.28 and 0.001.
2. v2 baselines for the proposed criteria (thread_side, pooled s1042 + s2042):
   placement pass rate 28/175 = 16 %; `latent_too_small` share of placed masks 21/28 = 75 %;
   accepted masks with > 3 components 7/7 = 100 %. **With a 100 % baseline, C5 cannot fail.**
   A different measure may be needed. Open for A, not changed here.
3. Offline v3 probe on the s2042 thread_side `placement_failed` masks (probe parameters, not
   frozen v3 parameters; 2 x 200 tries, 5 RNG seeds): translation only 0–1 / 71,
   rotation-aligned 45 / 71 (elongation >= 4: 40 / 40; < 4: 5 / 31). Output:
   `~/outputs/diffusion/proposals/placement_v3/rot_probe_fair_s2042.json`.
   This agrees with s1042 (§1.2).

Raw evidence: `~/outputs/diffusion/raw_runs/validation/screw_{thread_side,scratch_head}/seed2042/`;
bundles: `~/outputs/diffusion/smallbatch_s2042_budget130/`; log `~/jobs/logs/budget130_43341.out`.
