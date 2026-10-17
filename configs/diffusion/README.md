# Diffusion-internal rules (Member C)

These rules run **inside** the diffusion generator, before shared QC.
They do not change shared QC (`src/generation/qc.py`). Their rejection counts are
reported separately from shared-QC counts, and their cost is included in the
per-sample generation time.

Rule files are frozen: never edited after freezing; a change is a new version.
Each generation request records the rule file's SHA-256.

| File | SHA-256 (first 16) | Status |
|---|---|---|
| `mask_filter_v1.json` | `1f135ce97708cf6f` | frozen 2026-10-08 |
| `placement_v1.json` | `464806203f535606` | frozen 2026-10-08 |
| `placement_v2.json` | `51159d583f2821c7` | frozen 2026-10-10, approved by A |
| `mask_filter_v2.json` | `07de291f8010989e` | frozen 2026-10-10, approved by A |

## What each rule does

- `mask_filter_v1`: after upstream `check_mask`, reject sampled masks that are empty/too
  small, inverted (white on the border or > 20 % area), on a non-black background, or
  too large. Area window = [1/4 of the smallest, 4x the largest] support mask of that
  defect type at 256 px.
- `mask_filter_v2`: v1 plus a latent-resolution rule. After placement, the mask is resized
  exactly as upstream `ddpm.log_images` does (`torchvision.transforms.Resize((32, 32))`,
  then `>= 0.5`). The number of kept latent cells must be
  `>= max(2, min latent cells over the 5 support masks of this defect type)`.
- `placement_v1`: translate the mask's bounding box onto the shared-QC foreground of the
  chosen train/good source image; accept the first position with >= 95 % overlap.
- `placement_v2`: same, but the foreground is the shared-QC foreground minus the soft
  shadow halo: (same background model, distance threshold x3, holes filled) AND (QC foreground).

## Where the rule values come from (disclosure)

- Area window and support latent cells: computed per defect type from that type's 5
  frozen support masks only (training data). No cross-category pooling; flip excluded.
- Thresholds of the shape filter, the x3 foreground multiplier and the latent floor of 2
  were chosen **during development**, after looking at generated samples and train/good
  images. They were not set in advance. No held-out image, mask or metric was used.
