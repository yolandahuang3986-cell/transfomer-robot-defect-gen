# AnomalyDiffusion runtime (Member C)

The upstream code runs in its own environment and is reached only through
`scripts/diffusion/ad_worker.py` (a subprocess + JSON/PNG file boundary), in the
same way as the GAN worker. Upstream code is never imported by project code.

## Upstream

- Repository: https://github.com/sjtuplayer/anomalydiffusion
- Pinned commit: `b85297f`
- Clone location used on TC2: `~/external/anomalydiffusion` (outside Git)

## Environment (`Anomalydiffusion` conda env)

- Python 3.8, torch 1.12.1+cu113, run on TC2 A40 (CUDA driver 570).
- Pins needed to build/run on TC2 (`PIP_CONSTRAINT`):
  `lmdb==1.4.1`, `opencv-python<4.9`, `opencv-python-headless<4.9`, `numpy==1.23.1`, `Pillow==9.5.0`.
- Extra installs: `clip` (git, `--no-deps`), `imgaug==0.4.0`,
  `taming-transformers` at commit `3ba01b2` installed with `pip install --no-deps -e .`
  (a plain install produces an empty wheel).
- Base weights: `models/ldm/text2img-large/model.ckpt` (upstream's generic LDM weights).

## Local patches (`patches/`, apply with `git apply` on `b85297f`)

| Patch | Why |
|---|---|
| `support_list_dataset.patch` | With `SUPPORT_LIST=<json>` set, training reads only the frozen 5-shot support images/masks of each defect type, instead of upstream's "first third of the test set". Validation uses the first support image only. Prints `[patch] SUPPORT_LIST ...` so a job log proves it was active. |
| `mask_max_steps.patch` | `train_mask.py` hard-codes 30001 steps; `MASK_MAX_STEPS` makes it configurable. |
| `spatial_encoder_per_step.patch` | Upstream overwrites `spatial_encoder.pt` at every save; keep `spatial_encoder_gs-<step>.pt` so intermediate checkpoints are complete. |
| `stylegan2_op_fallback.patch` | The fused StyleGAN2 CUDA ops JIT-compile with ninja at import; TC2 lacks it. The import is wrapped; the ops raise if actually called (generation does not call them). |

All four patches are checked to apply cleanly together on `b85297f`.

## Checkpoint policy

- Official AnomalyDiffusion defect checkpoints were used for feasibility only.
  They were trained on test-set images and are never used for project generation.
- Project anomaly model: trained jointly on the 8 in-scope groups (metal_nut/flip
  excluded), 5 support pairs each, `SUPPORT_LIST=support_core_n5_seed42.json`.
  The trained `embeddings.pt` holds 73 name slots (upstream `name-anomaly.txt`);
  exactly the 8 in-scope names differ from their initial value (checked with
  `scripts/diffusion/analysis/check_anomaly_embeddings.py`).
- Project mask models: one per defect type, trained on that type's 5 support masks.
- Weights stay outside Git.
