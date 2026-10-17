# Member A analysis and evidence

As of 17 October 2026, all six E0 runs and the three-seed metal_nut E1 comparison
are complete. The original metal_nut 75-pair pool is accepted. **B's screw 125-pair
pool remains on hold and must be repaired and redelivered.** C's adapter and
validation evidence are merged, but the complete formal Diffusion pools are still
pending. GAN has failure-analysis evidence and no accepted formal pool.

## Completed analysis

- [Paired metal_nut E0/E1 results and figure](e0_e1_metal_nut_20261017.md)
- [GAN failure analysis and bounded bent ablation](gan_failure_analysis_20261017.md)
- [Quality and cost ledger with measurement boundaries](quality_cost_ledger_20261017.md)
- [English text integrated into A's report sections](a_report_sections_20261017.md)
- [Raw E1 metrics and execution evidence](../../results/metrics/e1_metal_nut_20261017/README.md)

E1 raises average pixel AUROC on metal_nut, while average Dice/IoU decrease and
normal-image FPR remains 1.0. This is a descriptive ranking result, not evidence
of improved segmentation or usable image-level detection. The three downstream
seeds use one fixed synthetic pool; no significance claim or overall method
ranking is made.

The OneDrive working report retains four main parts: A's background/framework,
B's Procedural/GAN methods, C's Diffusion methods, and A's common evaluation.
The completed metal_nut comparison, current GAN evidence, and current quality/cost
accounting have scoped finalized labels. The full cross-method comparison,
secondary ratios and final overall conclusion remain open.

## Remaining inputs and dependencies

| Owner | Required input or task | Enables |
|---|---|---|
| B | Repaired complete screw 125-pair candidate pool, provenance, QC and costs | A's whole-pool review, then screw E1 |
| C | Formal pools: 25 pairs per in-scope defect type (75/125 by category), training/checkpoint evidence and rejection/cost records | A's review, then E3 |
| A | Review each new complete pool and run frozen three-seed evaluation after acceptance | Completed matched primary comparison |
| A | Disclosed secondary ratio experiments and final quality/utility/cost synthesis | Final cross-method analysis and report conclusion |

C's previously reported validation-code issues also remain relevant at the
merged adapter head: internal attempt batches can exceed the configured cap,
zero-output exports omit accounting fields, and the development summary can omit
shared-QC rejected-sample costs. These are implementation/accounting corrections
for C to resolve before formal handoff; merging the validation adapter does not
close them. The formal A trainer already sums the bundle QC reports rather than
using that development summary.

No accepted GAN pool means E2 is recorded as not_run_quality_gate, with no score
imputed. This analysis does not authorize extra GAN training or changes to the
frozen shared QC, thresholds, evaluation split or training budget.
