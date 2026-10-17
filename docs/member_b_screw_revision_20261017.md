# Screw revised candidate pool — 17 October 2026

Replaces the ON HOLD screw candidate for review only. It does not replace metal_nut,
which A reports is accepted and has completed E1 training/evaluation. GAN remains paused.

Generation commit2914399, clean working tree, base main6d2256f. Algorithm
screw_dark_core_colored_polygon_v4. On each normal source/donor, Otsu is applied
again within the initial dark population to isolate a conservative metal core,
then the existing5px erosion is applied. Shared normal foreground determines
patch size independently of the smaller placement core. Placement is restricted
to the core; sharedQC code/thresholds are unchanged. Generic synthetic_patch only.

This was developed after inspecting the prior generated examples and A PR7;
it is NOT a pre-registered method. No held-out images or performance guided it.
A first correction (100d932) used the core for sizing too and underfilled at
120/125 after1250 attempts (1130 too-small rejects). Its complete QC statistics
are retained in prior_attempt_underfilled_qc.json. No images from that attempt
were reused. New sizing was frozen before generating this complete fresh pool.

125/125 accepted from134 attempts;9 too-small rejects; acceptance93.28%.
Base seed42, retry seeds increment via unchanged A helper; attempts limited1250.
CPU generation2.542s, QC0.505s, measured helper wall3.209s. Failed development
attempt cost26.080s generation +0.656s QC is separate, not concealed or pooled
into the production acceptance rate. Environment is recorded in environment.txt.

batch.npz is canonical float32, metadata.json and provenance.json preserve
source/donor IDs, file hashes, purpose=formal and source hashes. Full125-pair
changed-pixel alignment, loader/contract and formal-loader QC recheck pass.
PNG images/masks and metadata.jsonl are exports; no manual deletion/replacement.

review/ is produced by the actual A audit_formal_pool.py. The20 selected indices
use seed42 shuffle/pop. review/acceptance.json remains pending_review, without
A approval. B's Codex-assisted inspection saw no obvious detached shadow patches
in these20 examples; this is not independent human review or final acceptance.
Potential limitation: conservative dark regions bias placement away from bright
metal highlights; cannot guarantee named defect realism or downstream benefit.

Reproduce from2914399 using project environment and a new output directory:
python scripts/generate_procedural.py --root "$HOME/datasets/mvtec_ad" --manifest data/splits/fewshot_n5_seed42.json --category screw --num-samples 125 --seed 42 --shared-qc --purpose formal --out "$HOME/defect_runs/screw_revised"

MVTec-derived illustrations/data: MVTec AD, CC BY-NC-SA4.0. Data stays outside Git.

Tests:94 passed using the existing environment via a no-space symlink path. Two initial diffusion fixture subprocess failures were due to spaces in its shebang interpreter path; no diffusion source was changed.

## Uploaded artifact
OneDrive: transfomer/CA6127_Technical_Review/b_screw_revised_20261017.tar.gz
Upload confirmed17October2026.22,589,576bytes;285checksum entries.
SHA256:17838774afb4604d19a6f231ec4a141c01ba23e70a3d3a40b303b4e82a88e4a1
A must independently review and issue acceptance; no E1 screw run is claimed.
