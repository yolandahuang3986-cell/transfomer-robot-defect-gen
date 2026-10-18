# Member A evaluation delivery: E0 and metal_nut E1

更新：2026-10-17；E0 完成于 2026-10-10。对应 [冻结协议](formal_evaluation_protocol.md) 与 [执行说明](formal_evaluation_runbook.md)。

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

## 5. Completed analysis and report integration

The [analysis index](analysis/README.md) links the paired seed table/figure, GAN
failure and matched bent-ablation evidence, and the quality/cost ledger. The
English working report on OneDrive retains the four-part ownership structure.
The metal_nut E0/E1 result, current GAN evidence and current quality/cost account
are marked finalized within their stated scope; missing B/C experiments are not.

The analysis preserves raw result precision and source hashes, identifies
Codex-assisted visual review explicitly, and leaves unavailable costs null.
C's adapter/validation PR is merged; its formal candidate pools remain pending.

## 6. Remaining work

- **B:** Fix screw placement and deliver a new complete 125-pair pool with
  provenance, QC/cost records and fixed review evidence. The current pool is on
  hold. The GAN failure/ablation evidence has been received; it is not qualified
  formal E2 training data.
- **C:** Deliver 25 samples per in-scope defect type, with shared QC, training-data
  and checkpoint provenance, and the required formal-purpose declaration.
- **A:** Review new complete pools and execute the remaining accepted comparisons.
  No accepted GAN pool means E2 remains unrun, not a zero score.
- Complete ratio analysis, quality/utility/cost comparisons and the report once
  the required primary comparisons are available. E0 and metal_nut E1 completion
  do not mean that all four experiment chains or all A responsibilities are done.
