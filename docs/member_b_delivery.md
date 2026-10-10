# Member B code delivery

## What is implemented
- Foreground-constrained colored-polygon procedural baseline; labels actual modified pixels. Not full CutPaste/DRAEM reproduction and not semantically controlled.
- Frozen-support staging and strict file checks; isolated official GAN worker, checkpoint adapter, raw export and failure-mode tests.
- Optional --shared-qc procedural export using A PR#2 shared base/QC/scope implementation (now merged into main). Existing default exporter remains a pre-QC engineering path, not formal training data.
- Independent matched bent loss ablation scripts, 50epochs per arm, fixedseed42, pinned upstream and compatible augmentation dependency. Original outputs preserved. Observed results in member_b_results_20261010.md.

## Verified
27 local tests pass; isolated integration with A PR39a0800 also passes27 tests. Shared QC achieved30/30 requested for each category after32/33 attempts. Actual downloaded nine300epoch GAN checkpoints run but manual preview quality is not accepted. Completed40711 ablation does not establish successful repair.

## Interface limitations to resolve with A
Procedural emits defect_type=synthetic_patch and rejects semantic labels: do not claim it produces bent or thread_side. Agree on a common category-level quantity budget or an explicit allocation protocol without relabeling generic patches. Confirm downstream optimizer-step budget for E0/E1 to avoid unequal optimization being mistaken for data utility.
GAN adapter currently expects worker-generated training_report.json beside its checkpoint. TC2 legacy state dictionaries can be evaluated by diagnostics/export scripts but are not yet imported through the report-validated adapter; do not claim full legacy integration.
A's shared QC is merged into main and synchronized into this branch; B does not duplicate or overwrite A's shared modules in this patch. Scope excludesflip according to that PR, not a unilateral B change.

## Formal experiment status
No final E1/E2 downstream metrics have been produced. Underfilled GAN results must be recorded, not silently backfilled. A failed generator can support failure analysis, not a successful utility claim.

## Reproduction
Local B dependencies: team requirements plus requirements-gan.txt for isolated GAN runtime. GAN upstream5a8099204bc0858da314f8e0b5e8b6770c298e98.
Scripts accept data/output paths; datasets, weights and environments stay outside Git. Historical TC2 environment differed in augmentation behavior; keep that evidence separate from matched ablation.

## Main integration verification
Synchronized main f26c5ce (A PR#2). Kept A scope-aware inspector and shared QC; retained B concrete GAN adapter. Full combined suite: 37 passed. Formal downstream metrics and legacy checkpoint adapter import remain outstanding.
