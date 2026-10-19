# Member A analysis and evidence

As of 18 October 2026, all six E0 runs and the primary comparison are complete for
both accepted methods and both categories: metal_nut and screw E1 (Procedural) and
E3 (Diffusion). B's screw pool was repaired (`screw_dark_core_colored_polygon_v4`,
PR #10) and accepted; C's formal Diffusion pools (8 in-scope types × 25) were
independently leakage-checked and visually audited, and accepted for both
categories. GAN still has only failure-analysis evidence and no accepted formal
pool; E2 remains `not_run_quality_gate` by design.

## Completed analysis

- [Paired metal_nut E0/E1 results and figure](e0_e1_metal_nut_20261017.md)
- [GAN failure analysis and bounded bent ablation](gan_failure_analysis_20261017.md)
- [Quality and cost ledger, cross-method (18 Oct)](quality_cost_ledger_20261018.md),
  superseding the [17 Oct Procedural/GAN-only version](quality_cost_ledger_20261017.md)
- [English text integrated into A's report sections](a_report_sections_20261017.md)
- [Raw E1 metal_nut metrics and execution evidence](../../results/metrics/e1_metal_nut_20261017/README.md)
- [Raw E1 screw metrics and execution evidence](../../results/metrics/e1_screw_20261018/README.md)
- [Raw E3 metal_nut metrics and execution evidence](../../results/metrics/e3_metal_nut_20261018/README.md)
- [Raw E3 screw metrics and execution evidence](../../results/metrics/e3_screw_20261018/README.md)
- [Full primary-comparison writeup](../member_a_delivery.md) (§4–§6)

Across the four E1/E3 run-sets, average pixel AUROC rises over E0 in every case
(metal_nut: Procedural +0.1644±0.2171, Diffusion +0.1748±0.1140, 3/3 seeds up and
markedly more consistent than Procedural; screw: Procedural +0.1712±0.1464, Diffusion
+0.0504±0.0558, the smallest effect of the four). Dice/IoU gains are inconsistent
across seeds and, on screw, both methods stay at 0 at the fixed threshold because
the screw real-only baseline is already degenerate. These are descriptive ranking
results from one fixed pool per method/category and three downstream seeds; none
of them establish statistical significance or a general method ranking.

The OneDrive working report retains four main parts: A's background/framework,
B's Procedural/GAN methods, C's Diffusion methods, and A's common evaluation. B and
C's method/process/cost/limitations sections can now be finalized independently of
further primary runs; the cross-method effectiveness conclusion stays in A's part.
The cross-method quality/cost synthesis is complete (18 Oct ledger, above); secondary
ratio analysis and the final overall conclusion remain open (see `member_a_delivery.md` §8).

The 18 Oct cost ledger independently recomputed Diffusion's totals from each
bundle's raw `diffusion_internal.json`/`qc_report.json` (2026-10-18) and found them
to match C's reported figures exactly on attempts, accepted counts and generation
seconds. Cost per accepted Diffusion pair spans 7.7–103.3 s across the 8 defect
types, driven by stage-1 mask yield rather than inpainting cost. No controlled
cross-method speed ranking is possible: Procedural (CPU), Diffusion (GPU) and GAN
(no formal pool) were measured with different timers on different hardware paths.

## Remaining inputs and dependencies

| Owner | Required input or task | Enables |
|---|---|---|
| — | **Primary comparison: complete.** E0/E1/E3 done for both methods, both categories. | — |
| — | **Quality/cost synthesis: complete** (18 Oct cross-method ledger). | — |
| A | Decide the secondary ratio-analysis mix tiers (open since hand-off; candidates r=1, r=10 against the frozen primary r=5) | Secondary ratio experiment can be planned |
| A | Final cross-method conclusion once the above is available | Unified report, §4.8-equivalent |
| B / C | Finalize method, failure-analysis and cost sections (content already largely written) | Unified report assembly |

C's PR #9 fixed the three validation-code issues previously flagged at the merged
adapter head (stage-1 budget checked once per 8-mask batch instead of per mask;
zero-output exports omitting accounting fields; the development cost summary
omitting shared-QC-rejected-sample costs). The formal A trainer already summed the
bundle QC reports rather than using that development summary, so the formal pool's
numbers were unaffected; see PR #9's description for the fix-by-fix mapping.

No accepted GAN pool means E2 is recorded as not_run_quality_gate, with no score
imputed. This analysis does not authorize extra GAN training or changes to the
frozen shared QC, thresholds, evaluation split or training budget.
