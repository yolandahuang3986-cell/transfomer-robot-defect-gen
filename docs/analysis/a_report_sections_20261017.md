# Member A report sections

## 4.5 Matched Synthetic Method Comparison

The original 75-pair Procedural metal_nut pool passed A's whole-pool acceptance after file, source, mask-alignment and shared-QC checks and a Codex-assisted review of the fixed 20-pair sample. This review is not an additional independent human rating. The legacy-purpose record binds the unchanged bundle files to their original formal-candidate use. The 125-pair screw pool remains on hold for patches on shadow or background, so only metal_nut enters E1.

Three E1 runs completed on TC2 A40 with the same 235 real training images and one fixed 75-pair synthetic pool. Each used 1,500 updates, the final-step checkpoint and the frozen 77-image evaluation set. Model settings and recorded package versions match E0. Table 6 reports means, sample SDs and same-seed differences; Figure 2 shows the individual paired results.

Mean pixel AUROC rises from 0.6559 to 0.8202, with paired change 0.1644 ± 0.2171. Two seeds improve and one declines. Dice decreases from 0.0821 to 0.0591 and IoU from 0.0467 to 0.0312 at threshold 0.5. Normal-image FPR stays at 1.0 for every seed: all 22 normal test images trigger at least one positive pixel. Recall of 1.0 therefore does not establish useful inspection.

For RQ1, this category shows a descriptive gain in average pixel ranking, without an improvement in thresholded overlap or image-level false alarms. The runs do not isolate a causal mechanism. Generic synthetic_patch labels do not establish named-defect realism. Screw E1 and E2/E3 are unavailable, so no two-category Procedural macro score or overall method ranking is reported. The SD measures downstream randomness with one pool; it does not cover support-split or generator variability. No thresholds, seeds or training budgets were changed after observing these results.

## 4.6.1 GAN Failure Evidence and Background Loss Ablation

B's original in-scope previews comprise five normal training sources for each of three metal_nut and five screw defect types. At threshold 0.5, 23 of the 25 screw masks are empty; the other two contain only 2 and 8 pixels. B's shared-QC report records zero passing screw previews, while metal_nut scratch and color pass geometrically. These small preview sets are diagnostic, not full pools generated under the primary retry budget or evidence of visual acceptance.

The bent ablation compares background-loss weights 0.1 and 0 at recorded seed 42, with 50 epochs per arm, the same 220 normal images and five frozen bent support pairs. Packaged provenance, common settings and environment lists match, and both logs reach epoch 50. A rethresholded the four supplied probability arrays and verified the empty-mask counts. Table 7 retains B's reported shared-QC counts separately from those independent checks.

Disabling background loss did not yield a stable repair: the zero-weight arm has no geometric QC passes at either checkpoint, and four of five masks are empty by epoch 50. The default arm also deteriorates, from five geometric passes at epoch 25 to one at epoch 50. Visible texture/shape artifacts and weak image-mask correspondence remain. This bounded single-seed experiment cannot establish a unique cause or general failure of the published method. No accepted GAN pool is available; E2 is recorded as not_run_quality_gate, with no downstream score substituted.

## 4.6.2 Quality and Cost Accounting

Procedural generation needed 83 attempts for 75 metal_nut QC passes and 136 for 125 screw passes; all 19 rejections were too-small masks. Geometric pass rates are 90.36% and 91.91%. The first pool is accepted and evaluated; the second remains on hold. Thus the passing count is distinct from the number admitted to downstream training. Table 8 reports B's CPU generation and shared-QC timers, including rejected attempts. Their sum excludes normal loading/resizing and later serialization and review production; it is not a controlled GPU latency benchmark.

The three E1 trainer runtimes are 68.19, 69.38 and 68.18 seconds. These timers include integrity checks, dataset setup, training and evaluation up to result assembly; they exclude Python imports, checkpoint writing and queue time. The Slurm allocation takes 285 seconds including preflight. A separate training-only probe measures 0.04701 seconds per update after five warmup steps and reports 0.514 GiB peak PyTorch allocation. Neither quantity measures generator cost or total process VRAM.

Procedural needs no learned fitting stage. Comparable GAN/Diffusion fitting, rejected-attempt and memory costs remain incomplete, so the ledger uses missing/inapplicable entries rather than measured zeros. Cost per formally accepted screw pair is undefined while the batch is on hold. Current CPU/GPU figures do not support a controlled speed ranking. Generator quality, semantic coverage, accepted-pair cost and downstream utility remain separate outcomes.

## 4.6.3 Secondary Analyses Still Pending

The primary comparison uses synthetic ratio five. Ratios one and ten have not been run through a separately documented secondary entry. They must preserve traceable pool construction and disclose the observed primary results. C's placement, mask-filter and sampling-step evidence is also pending final integration. These analyses cannot be marked complete or used to replace missing formal E3 results.

## 4.7 Threats to Validity and Future Extensions

Two products, one support split and three downstream seeds limit generalization. The E1 comparison is currently restricted to metal_nut and one synthetic pool. Repeats measure inspector-training variability rather than support or generator randomness. Flip remains excluded. Development outputs informed some placement/filter choices; that timing must be disclosed. The fixed threshold and update budget were not changed after held-out evaluation.

Equal accepted counts and inspector updates do not equalize generator fitting cost, semantic coverage or positive-pixel balance. Foreground estimates can include screw shadow, and fixed-sample acceptance cannot certify every generated pair. The metal_nut review was Codex-assisted, not an additional independent human rating. Possible future controls include independent validation for threshold calibration, multiple support splits, real-support oversampling, improved object masks and robot-view perturbations. These are proposed extensions, not repairs applied to the reported primary comparison.

## 4.8 Conclusions and Outstanding Comparisons

The project has a frozen comparison protocol, six completed real-only runs and a three-seed metal_nut Procedural comparison. In the latter, average pixel AUROC increases while Dice/IoU decrease and every normal test image still triggers an alert. Generated patches therefore show a limited ranking benefit under this recipe, without demonstrated improvement in thresholded segmentation or usable image-level detection. The GAN diagnostics and bounded loss ablation document a quality-gate failure rather than a successful E2 comparison.

RQ1 is partly answered for metal_nut only. For RQ2, the received evidence shows that geometric QC can admit shadow placement and cannot establish semantic or visual correctness. RQ3 remains incomplete because comparable GAN/Diffusion costs and formal pools are unavailable. B still owes a repaired complete screw pool, and C's formal Diffusion handoff must pass the same admission process. The overall method ranking and two-category synthetic conclusions remain open.
