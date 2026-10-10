# Member B: local Phase 1 handoff (2026-10-02)

## Scope and status

Procedural baseline: colored polygon CutPaste-style editing, not full CutPaste/DRAEM reproduction. B branch is based on A commit `0b4d06298721748ae475c7dda31a55240e01503a`; fetched main `a4fcdf4` has an identical tree. No frozen split or A trainer/evaluator files were changed.

- Two 30-sample batches (metal_nut and screw) generated from **normal training images only**.
- Frozen manifest and freeze attestation verified, including file hashes.
- Foreground estimated from each normal image by Otsu brightness threshold (bright metal nut / dark screw), then eroded. This is a category-specific heuristic, not ground-truth object segmentation.
- Each binary mask labels exactly the changed float32 pixels. Edits are clipped to estimated foreground. Background errors from v1 uniform placement are addressed; conservative foreground may exclude dark nut surfaces or reflective screw surfaces.
- Output is generic `synthetic_patch`; no claim to synthesize particular real defect types such as bent, flip, or thread_top.
- Images are NCHW float32 [0,1], masks N1HW binary float32, metadata includes normal source and donor IDs, configuration and generation time.
- Both batches completed one CPU optimization step using A's unchanged U-Net builder, generated loader and evaluator. 64px functional check only, not formal 256px training or held-out evaluation.
- Formal downstream improvement, A acceptance, GPU timing and GAN training remain pending.
- Verification on 2026-10-02: **22 pytest tests passed** (A's original 13 plus 9 new B tests). Both category smoke checks passed.
- Final local outputs are `../outputs/procedural_v2_checked_metal_nut_seed42` and `../outputs/procedural_v2_checked_screw_seed42`; earlier v1/v2 folders are development artifacts.
- Full installed local environment versions are recorded outside Git at `../outputs/procedural_local_macos_packages.txt`.
- GAN stage executed and verified: metal_nut/scratch contains 220 normal images, exactly 5 frozen support defects and 5 matching masks, no held-out images. Not a GAN training result.

## Reproduce procedural export

Use the project requirements in an isolated environment. From repository root, with data and outputs outside Git:

```bash
python -m pytest -q
python scripts/generate_procedural.py --root ../datasets/mvtec_ad --manifest data/splits/fewshot_n5_seed42.json --category metal_nut --num-samples 30 --image-size 256 --seed 42 --out ../outputs/procedural_metal_nut_run1
python scripts/generate_procedural.py --root ../datasets/mvtec_ad --manifest data/splits/fewshot_n5_seed42.json --category screw --num-samples 30 --image-size 256 --seed 42 --out ../outputs/procedural_screw_run1
python scripts/smoke_procedural_inspector.py --manifest data/splits/fewshot_n5_seed42.json --category metal_nut --bundle ../outputs/procedural_metal_nut_run1 --out ../outputs/procedural_metal_nut_run1/inspector_smoke.json
```

Existing output folders are not overwritten. Bundles contain `batch.npz`, `metadata.json`, `provenance.json`, and `preview.png`. Source file checksums record uncommitted implementation versions; `code_dirty` is disclosed. Timestamps/runtime are not expected to be bit-identical across reruns.

For A's formal runner, call `load_bundle(directory, manifest, category)` in `src/generation/bundle.py`, then pass the returned GeneratedBatch to existing `generated_dataset`. The reader verifies split identity, bundle checksums and source/donor membership in normal training rows. A must still wire this into the formal runner and settle training budget, synthetic ratio and validation protocol.

## GAN source audit

Selected official repo: https://github.com/pasqualecoscia/SyntheticDefectGeneration
Pinned commit: `5a8099204bc0858da314f8e0b5e8b6770c298e98`.

Observed directly in source:

1. `data/create_mvtec_dataset.py` takes the first 80% of official test defects for GAN training. **Do not execute that splitter for this project.** It conflicts with our frozen 5-shot protocol.
2. Use `scripts/prepare_gan_support.py` to export `train/A` normal images, `train/B` frozen support images, `train/mask` aligned masks. Original names map to sequential filenames in provenance. Held-out and test-normal files are not copied.
3. `src/utils_datasets.py` pairs B and mask by sorted order; our stage gives matching numeric names. Train length is max(A,B), so each of five support defects can be sampled repeatedly.
4. Upstream imports `albumentations` and `torchsummary`, which are absent from its requirements list; torch/torchvision are also not listed. Its ElasticTransform call uses legacy arguments. Exact compatibility must be tested in a separate GAN environment; do not blindly install latest dependencies.
5. `--mask` is defined with `action=store_false`: passing that flag disables masks. Leave it absent for the default masked model.
6. Defaults include 256px, batch size 1, 300 epochs, decay start 100. For a short test, epochs must remain greater than decay_epochs. These are upstream defaults, not an approved project training budget.
7. The generator emits a continuous learned mask. The adapter needs a documented binary threshold and normalized RGB output. Visualization PNGs saved with matplotlib or automatic contrast normalization must not serve as raw training masks.
8. No runtime/GPU/VRAM claim is made from source inspection. The project YAML's generic latent_dim is not yet a verified upstream control.

First training-only stage command:

```bash
python scripts/prepare_gan_support.py --root ../datasets/mvtec_ad --manifest data/splits/fewshot_n5_seed42.json --category metal_nut --defect-type scratch --out ../datasets/gan_support/metal_nut_scratch
```

Next: validate actual GAN imports and one optimization step in an isolated environment; then implement the real checkpoint adapter and produce reviewable samples. GPU execution on TC2 remains subject to the user's network/account access and SLURM allocation.


## GAN execution update

See [member_b_gan_progress.md](member_b_gan_progress.md) for actual 1/10-step CPU runs, 27 tests, failed smoke sample quality, reproducible commands and the unexecuted TC2 GPU job. Earlier GAN-pending statements above describe the initial procedural milestone.
