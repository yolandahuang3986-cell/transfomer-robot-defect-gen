# Member A evaluation delivery: E0, E1 (Procedural) and E3 (Diffusion)

更新：2026-10-18；E0 完成于 2026-10-10。对应 [冻结协议](formal_evaluation_protocol.md) 与 [执行说明](formal_evaluation_runbook.md)。

## 1. 已完成

- 冻结范围、配额、三 seed 和等优化步数：只排除 flip、保留 color；正式使用 40 张异常 support。
- 实现各方法通用合成池读取、数据来源/哈希/配额校验、共享 QC 重验、绑定整批数据的人工验收记录。
- 实现正式训练、假阳性率、计划/执行入口、配对指标汇总和 TC2 提交脚本。
- 本地 Python 3.11 与 TC2 Python 3.10 均为 49 tests passed；CPU/GPU 实际 U-Net Gate 0 均通过。GitHub PR #4 的 CPU CI 也已通过。
- 六次 E0 已全部完成：两个类别，各 seeds 42/43/44。任务 40765，NVIDIA A40，COMPLETED，exit 0，调度任务总时长 00:11:36。

训练代码 commit：`ec8fc9408e683360ebfb8fe3814891ace27e6c5d`，六份结果均记录干净工作区。后续文档/结果提交不改变该次训练的代码来源。

## 2. 正式 E0 结果

每个数值为三个下游 seed 的均值 ± 样本标准差；该方差不是生成器随机性的估计。Dice/IoU 使用类别全部测试像素汇总，阈值固定 0.5。

| 类别 | Pixel AUROC | Dice | IoU | 异常图 Recall | 正常图 FPR |
|---|---:|---:|---:|---:|---:|
| metal_nut | 0.6559 ± 0.1177 | 0.0821 ± 0.1391 | 0.0467 ± 0.0792 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| screw | 0.5201 ± 0.2718 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.2846 ± 0.4929 |

metal_nut：235 张真实训练图、77 张测试图、1500 优化步；screw：345 张真实训练图、135 张测试图、2200 优化步。测试图含正常图，异常测试图均来自冻结 held-out 并按相同 scope 排除 flip。

解释边界：

- metal_nut 的 image-level Recall=1 同时伴随 FPR=1，不能将其解释为检测成功；“任意阳性像素”规则会把局部误报放大为整图告警。
- screw 在固定阈值下 Dice/IoU/Recall 为 0。AUROC 的跨 seed 波动较大，当前 real-only 下游基线较弱且不稳定。
- 本次只证明固定协议下的 E0 真实表现；尚不能证明任何生成方法改善检测。
- 没有根据这些 held-out 结果重新挑 seed、调阈值、延长训练或更换模型。后续 E1/E3 使用同一预算；任何新协议实验须另行批准并披露已观察测试结果。

## 3. 成本与原始证据

单次训练器记录的运行时长：metal_nut 约 80–82 秒，screw 约 120 秒；包含数据校验、训练与评估等程序内工作，不等同于纯 GPU 训练时间。整个 Slurm 任务还包括测试、Gate 0、测速、进程启动和结果保存。

训练测速使用 5 步预热 + 20 步计时，约 0.0481 秒/步。PyTorch peak allocated 显存为 551,783,424 bytes（约 0.514 GiB）；这不是进程总 GPU 占用，也不是生成器显存成本。

原始结果与复现配置：

- [metal_nut seed 42](../results/metrics/e0_20261010/real_only_metal_nut_seed42.json)
- [metal_nut seed 43](../results/metrics/e0_20261010/real_only_metal_nut_seed43.json)
- [metal_nut seed 44](../results/metrics/e0_20261010/real_only_metal_nut_seed44.json)
- [screw seed 42](../results/metrics/e0_20261010/real_only_screw_seed42.json)
- [screw seed 43](../results/metrics/e0_20261010/real_only_screw_seed43.json)
- [screw seed 44](../results/metrics/e0_20261010/real_only_screw_seed44.json)
- [机器可读汇总](../results/metrics/e0_20261010/summary.json)；[训练测速](../results/metrics/e0_20261010/timing.json)。

模型权重、日志和完整任务计划保留在 A 的 TC2 目录 `/home/mcaai/yuanyuan010/robot-evaluation/e0-ec8fc94/`。该路径位于个人账户，不宣称 B/C 已有访问权限；Git 中只保留小型指标和说明文件。

## 4. Completed metal_nut E1 — 17 October 2026

The original complete 75-pair Procedural metal_nut pool has been accepted after
technical checks and Codex-assisted visual review, following the user's
instruction to start formal training. The record does not claim an additional
independent human rating. The hash-bound legacy-purpose review preserves the
original data and provenance. Screw remains on hold for placement quality.

Job **43178** completed on NVIDIA A40 with **exit 0:0**, elapsed **4:45**. Seeds
42/43/44 each completed the frozen 1,500 optimizer steps using the same accepted
pool. All 88 tests and CUDA Gate 0 passed. Model settings, training budgets,
evaluation partitions and recorded package versions match E0. E0 was not rerun.

| Metric | E0 metal_nut | E1 metal_nut |
|---|---:|---:|
| Pixel AUROC | 0.6559 ± 0.1177 | 0.8202 ± 0.0995 |
| Dice | 0.0821 ± 0.1391 | 0.0591 ± 0.0644 |
| IoU | 0.0467 ± 0.0792 | 0.0312 ± 0.0345 |
| Normal-image FPR | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

The average AUROC gain does not establish improved segmentation: Dice and IoU
fall at the fixed 0.5 threshold, and every normal test image still triggers at
least one positive pixel. Three downstream seeds with one fixed synthetic pool
and one completed category do not establish overall superiority or statistical
significance. No protocol was changed after observing these results.

[Full results, paired differences and execution evidence](../results/metrics/e1_metal_nut_20261017/README.md).
Weights remain outside Git in A's TC2 result storage; acceptance and run records
are also saved in the OneDrive project review folder.

## 5. Completed screw E1 (Procedural) — 18 October 2026

B's revised screw pool (`screw_dark_core_colored_polygon_v4`, commit `2914399`,
PR #10) replaces the held original: a second Otsu split isolates a conservative
dark material core for placement, separately from the shared-foreground patch-size
reference. 125/125 accepted from 134 attempts (93.28%), 9 `mask_too_small` rejects;
the earlier underfilled development attempt (120/125 at 1250 attempts) is kept as a
separate, unused record. A verified all 285 package checksums, confirmed every
source/donor image is `screw/train/good/*` (frozen split hash match, no held-out or
support use), and visually audited all 20 fixed review pairs: every patch lands on
the physical screw, resolving the prior hold (patches 0009/0082 on background/shadow).
Accepted 2026-10-18.

Job **43671** completed on NVIDIA A40 with **exit 0:0**, elapsed **6:19**. Seeds
42/43/44 each completed the frozen 2,200 optimizer steps using the accepted pool,
code commit `77ebf7f` (same as E0/metal_nut E1). CUDA Gate 0 passed.

| Metric | E0 screw | E1 screw |
|---|---:|---:|
| Pixel AUROC | 0.5201 ± 0.2718 | 0.6913 ± 0.1515 |
| Dice | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| IoU | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Anomalous-image recall | 0.0000 ± 0.0000 | 0.3333 ± 0.5774 |
| Normal-image FPR | 0.2846 ± 0.4929 | 0.3333 ± 0.5774 |

Pixel AUROC improves on average (2 of 3 seeds up, one roughly flat), but Dice/IoU
stay at 0 for both E0 and E1: the screw real-only baseline is already degenerate at
the fixed 0.5 threshold (recorded as-is since E0), and this does not change under
Procedural training. Recall moves from 0/3 to 1/3 seeds — a single-seed change, not
a reliable detection gain. No protocol was changed after observing these results.

[Full results, paired differences and execution evidence](../results/metrics/e1_screw_20261018/README.md).

## 6. Completed E3 (Diffusion), metal_nut and screw — 18 October 2026

C's formal AnomalyDiffusion pool (8 in-scope defect types × 25 = 200 pairs, code
commit `6e2d490`/main `dd65331`, PR #9 + PR #11) was independently leakage-checked
and visually audited by A, then accepted for both categories on 2026-10-18:

- **Leakage**: the 40 anomaly-model training records and both mask-model support
  lists match the frozen in-scope support (`fewshot_n5_seed42.json`) exactly by path
  and SHA-256, with zero hash intersection against the 149 held-out images (checked
  both directions). `metal_nut+flip`'s mask model was trained but is not referenced
  by `configs/diffusion/generation.json` and was not used for formal generation.
- **Code**: all 7 unique source files referenced in the 8 bundles' `provenance.json`
  (56 entries) match `origin/main` @ `dd65331` byte-for-byte.
- **Quota**: 75 metal_nut (bent/color/scratch × 25) + 125 screw (manipulated_front/
  scratch_head/scratch_neck/thread_side/thread_top × 25), no flip.
- **Visual audit**: 20/20 metal_nut pairs and 20/20 screw pairs show the defect
  primarily on the physical object; 3 of the 20 screw pairs show a few stray mask
  pixels on background next to the screw head, matching the already-disclosed,
  quantified limitation in `member_c_delivery.md` §10.2c (≤5% of mask area outside
  the placement foreground) — not a new or undisclosed defect, and not comparable to
  the full-patch-on-background issue that held the original Procedural screw pool.

Jobs **43672** (metal_nut, 4:02) and **43675** (screw, 5:48) both completed on
NVIDIA A40 with **exit 0:0**, seeds 42/43/44, frozen optimizer steps (1,500 /
2,200), code commit `dd65331` (its `src/inspection/formal_bundle.py` and
`scripts/train_inspector.py` are byte-identical to the `77ebf7f` commit used for
E0/E1, so downstream training/eval code is unchanged across experiments). CUDA
Gate 0 passed for both.

| Metric | E0 metal_nut | E1 metal_nut (Procedural) | E3 metal_nut (Diffusion) |
|---|---:|---:|---:|
| Pixel AUROC | 0.6559 ± 0.1177 | 0.8202 ± 0.0995 | 0.8306 ± 0.0321 |
| Dice | 0.0821 ± 0.1391 | 0.0591 ± 0.0644 | 0.2198 ± 0.2824 |
| IoU | 0.0467 ± 0.0792 | 0.0312 ± 0.0345 | 0.1443 ± 0.1970 |
| Normal-image FPR | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9242 ± 0.1312 |

| Metric | E0 screw | E1 screw (Procedural) | E3 screw (Diffusion) |
|---|---:|---:|---:|
| Pixel AUROC | 0.5201 ± 0.2718 | 0.6913 ± 0.1515 | 0.5705 ± 0.2347 |
| Dice | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| IoU | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| Normal-image FPR | 0.2846 ± 0.4929 | 0.3333 ± 0.5774 | 0.2927 ± 0.5069 |

On metal_nut, Diffusion's average AUROC gain (+0.1748 ± 0.1140, 3/3 seeds improve)
is slightly larger and markedly more consistent across seeds than Procedural's
(+0.1644 ± 0.2171, 2/3 improve); its Dice/IoU gain is larger in mean but driven
mostly by one seed (43), not a consistent per-seed pattern. On screw, Diffusion's
gain (+0.0504 ± 0.0558) is clearly smaller than Procedural's (+0.1712 ± 0.1464) —
the smallest effect of the four E1/E3 runs so far. These are descriptive results
from one fixed pool per method/category and three downstream seeds: they do not
establish statistical significance or a general Procedural-vs-Diffusion ranking.
No protocol was changed after observing these results.

[metal_nut full results and evidence](../results/metrics/e3_metal_nut_20261018/README.md);
[screw full results and evidence](../results/metrics/e3_screw_20261018/README.md).
Weights remain outside Git in A's TC2 result storage; both categories' hash-bound
`acceptance.json` records are kept in A's local review archive, synced from the
OneDrive project review folder.

## 7. Completed analysis and report integration

The [analysis index](analysis/README.md) links the paired seed table/figure, GAN
failure and matched bent-ablation evidence, and the quality/cost ledger. The
English working report on OneDrive retains the four-part ownership structure.

As of 2026-10-18, the primary comparison is complete for both accepted methods and
both categories: E0 vs E1 (Procedural, metal_nut + screw) and E0 vs E3 (Diffusion,
metal_nut + screw), 24 downstream training runs total (4 run-sets × 3 seeds, plus
the 6 E0 runs). GAN has no accepted formal pool; E2 is recorded as
`not_run_quality_gate`, not a zero score, per the approved decision to treat it as
a negative result (existing failure/ablation evidence stands, no new GAN training
is planned). The analysis preserves raw result precision and source hashes and
identifies AI-assisted visual review explicitly throughout (B's and A's own).

## 8. Secondary ratio analysis (Procedural, metal_nut) — 19 October 2026

Decided 18 October: test r=1 and r=10 against the frozen primary r=5 (§4 of the
protocol), scoped first to metal_nut, both accepted methods, with a reduced
human-audit pass (5–8 pairs per tier instead of the primary 20) given the
generation code itself is already accepted at r=5.

**A construction bug was found and corrected same-day.** §4 of the protocol
requires the tiers to be nested ("r=1 uses the primary pool's deterministic
stratified prefix, r=10 keeps the primary pool and appends"). The first attempt
independently re-invoked `generate_procedural.py` at each target count (`--seed
42`, `--num-samples 15` / `150`) — this was *not* a prefix/superset of the
accepted primary pool: a shared-QC rejection partway through a run shifts every
sample generated afterward, so a different `--num-samples` at the same seed
does not reproduce the primary pool's sequence past the first rejection.
Verified directly: the independent r=1 matched the primary pool's first 15 on
only 14/15; the independent r=10 matched the primary pool's first 75 on only
67/75. That result was shown in the shared OneDrive report draft for a few
hours, marked `[CORRECTION IN PROGRESS]`, and was never committed to this repo.

**Corrected construction:** r=1 (15 pairs) = a literal, byte-identical prefix
of the accepted primary pool's own `batch.npz`/`metadata.json` — no
regeneration. r=10 (150 pairs) = the primary pool (75) kept completely
unchanged, plus 75 newly generated pairs appended from a second bundle under a
disclosed, non-primary seed (142, chosen only to avoid a TC2 output-directory
collision with the primary pool's own seed-42 run; 80 attempts, 93.75%
acceptance). This needed a small code change admitting a bundle with a
disclosed non-primary seed in ratio-analysis mode only (PR #15; its own
real-data regression check, and this analysis's real use of it, both confirm no
effect on any already-accepted primary pool).

Jobs **44078** (r=1) and **44079** (r=10), both exit 0:0, NVIDIA A40, code
commit `ed5a727` (includes PR #15), seeds 42/43/44, 1,500 steps.

| Ratio (n) | Pixel AUROC | Δ vs E0 | Dice | IoU |
|---|---:|---:|---:|---:|
| r=1 (15, true prefix) | 0.6918 ± 0.1208 | +0.0360 ± 0.0046 | 0.0190 ± 0.0305 | 0.0097 ± 0.0157 |
| r=5 (75, primary, unchanged) | 0.8202 ± 0.0995 | +0.1644 ± 0.2171 | 0.0591 ± 0.0644 | 0.0312 ± 0.0345 |
| r=10 (150, primary + seed-142 append) | 0.8000 ± 0.1298 | +0.1442 ± 0.2139 | 0.0429 ± 0.0383 | 0.0222 ± 0.0199 |

Pixel AUROC rises from r=1 to r=5, then dips slightly at r=10 rather than
continuing to improve — diminishing, not negative, returns in this range.
r=10's paired gain is close to r=5's and still noisy across seeds; r=1's gain
is smaller but markedly more consistent. Dice/IoU fall below E0 at all three
ratios, least so at r=10. Three ratio points and three seeds per point do not
establish statistical significance or a confirmed optimum.

Diffusion's side is unaffected by the bug (never run) and is now a smaller ask
for C: r=1 needs no new generation (A builds it directly from her already-
accepted bundles); r=10 only needs the 25/type append portion under a
disclosed fresh seed, not a full 50/type regeneration.

[Full results and evidence](../results/metrics/ratio_analysis_metal_nut_20261019/README.md).

## 9. Remaining work

The B and C delivery/acceptance/primary-comparison chain (steps 1–6 of the team's
linear workflow) is complete for both methods and both categories. What remains is
secondary analysis and the final report, not new primary comparisons:

- **Ratio analysis:** Procedural side complete (§8 above). Diffusion side
  (r=10's 25/type append only; r=1 needs no generation) requested from C, not
  yet delivered.
- **Quality/cost synthesis: complete** (18 Oct 2026). B's and C's generation
  timers, QC statistics and failure taxonomies are compiled into one cross-method
  [ledger](analysis/quality_cost_ledger_20261018.md); Diffusion's totals were
  independently recomputed by A from the raw bundle JSON and match C's reported
  figures exactly. No controlled cross-method speed ranking is possible (different
  timers, different hardware paths); Diffusion's per-type cost spans 7.7–103.3 s
  per accepted pair, driven by stage-1 mask yield.
- **Final report and PPT:** A assembles the unified report (background/framework +
  common evaluation); B and C contribute their method, failure-analysis and cost
  sections, which can be finalized now independent of the ratio analysis. Only the
  final cross-method conclusion section depends on the ratio analysis, if it runs.
- No accepted GAN pool means E2 remains unrun by design, not a gap to fill.
