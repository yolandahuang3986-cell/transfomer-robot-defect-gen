# GAN progress — 2026-10-02

## What has actually run

- Isolated local Python 3.12.14 environment; PyTorch 2.8.0, torchvision 0.23.0, NumPy 1.26.4, albumentations 1.3.1, OpenCV headless 4.10.0.84, einops 0.8.2 and torchsummary 1.5.1.
- Pinned official source: SyntheticDefectGeneration at `5a8099204bc0858da314f8e0b5e8b6770c298e98`. Tracked upstream source is unchanged. The project wrapper uses its CycleGAN_Mask networks, losses, update functions and training augmentations.
- Local CPU runs at 256px, batch size 1, full ngf=64 and 9 residual blocks: 1 step (3.34s training loop) and 10 steps (32.57s training loop), with finite reported losses. Times exclude dependency installation and model setup, and are not GPU estimates.
- Both runs use only 220 metal_nut normal-training images and the 5 frozen scratch support images. No held-out examples train the model.
- A2B generator checkpoint saved, checksummed and reloaded by subprocess adapter. Images normalized from tanh [-1,1] to [0,1]; learned sigmoid masks thresholded at the explicit upstream option default 0.2 (no extra sigmoid).
- Twenty image/mask pairs exported from each run, with source, split, checkpoint and code hashes.
- **Quality acceptance FAILED for these smoke batches:** all 20 masks cover 100% of the image at threshold 0.2; outputs after 1 step are near-gray, after 10 steps still poor. These are undertrained engineering checks, not usable defects or evidence of mode collapse after convergence.
- Default adapter refuses smoke checkpoints unless `allow_smoke=True` is explicitly passed. Exported metadata always preserves smoke status; no formal downstream result has been computed.
- 27 project pytest tests passed (original A tests plus B procedural, bundle and GAN input safeguards).
- Generator is deterministic in evaluation for a given normal input; seed chooses source order, not latent defect variants. Repeating one normal image does not create independent new defects.

## Remaining before Gate 1

1. Check user's TC2 setup job 39173 (last observed downloading dependencies; completion unknown).
2. Transfer B code overlay to the existing `~/projects/defect-gen` clone; this work has not been pushed to GitHub.
3. Submit the provided GPU smoke job to measure real GPU model, VRAM and runtime.
4. Agree on a longer training budget and validation protocol with A. The wrapper's checkpoint is inference-only; optimizer/replay/RNG resume support must be added before long, restartable six-hour training chunks.
5. Train and inspect real image/mask quality; do not cure full-image masks merely by choosing a threshold on held-out data.
6. Extend accepted training to agreed defect types and screw, then integrate into A's formal runner. Gate 1 remains pending.

## Local commands

Run from project root, using separate project and GAN Python environments:

```bash
../envs/procedural/bin/python scripts/train_gan.py --root ../datasets/mvtec_ad --manifest data/splits/fewshot_n5_seed42.json --stage ../datasets/gan_support/metal_nut_scratch --upstream ../upstream/SyntheticDefectGeneration --python ../envs/gan/bin/python --out ../outputs/gan_new_smoke --image-size 256 --max-steps 10 --device cpu
../envs/procedural/bin/python scripts/generate_gan.py --root ../datasets/mvtec_ad --manifest data/splits/fewshot_n5_seed42.json --upstream ../upstream/SyntheticDefectGeneration --python ../envs/gan/bin/python --checkpoint ../outputs/gan_new_smoke/generator_A2B.pt --category metal_nut --defect-type scratch --num-samples 20 --allow-smoke --out ../outputs/gan_new_smoke_samples
```

`--max-steps 0` completes requested epochs rather than stopping early. This flag does not establish quality acceptance. Current configs/gan.yaml documents upstream defaults; the runner is controlled by CLI flags, not YAML parsing.

## TC2 command once B files are transferred

```bash
cd ~/projects/defect-gen
sbatch --output="$HOME/job_logs/gan-%j.log" scripts/tc2_gan_smoke.sbatch
```

The job creates a separate `~/envs/defect-gan`, stages a fresh frozen support set, runs 10 GPU updates, exports 20 smoke samples and records environment versions under `~/defect_runs/gan-smoke-JOBID`. It does not modify A's environment. Script syntax is checked locally; TC2 execution is not yet verified.
