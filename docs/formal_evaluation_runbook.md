# A 正式评估执行说明

对应 [v1.0-frozen 协议](formal_evaluation_protocol.md)。代码接入已实现；真实样本的人工验收和最终实验结果不由单元测试代替。

## 1. 环境和基础验证

使用 Python 3.10 或以上的独立项目环境。上游 GAN/Diffusion 环境通过文件交付，不直接 import 项目代码。

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/run_gate0.py --device cpu --out /tmp/robot-gate0.json
```

Gate 0 使用独立 fixture，仅验证实际 U-Net 优化、推理和结果 schema；不代表真实数据效果。正式 runner 要求干净的已提交代码、原始冻结 manifest 和相同数据哈希。

## 2. B/C 合成数据交付

主实验：metal_nut 75、screw 125；基础生成 seed 42。共享 QC 的重试 seed 按现有 helper 递增，记录实际 seed，不将每条 metadata 强制改写成 42。

Procedural 每类别一个 bundle。GAN/Diffusion 每个有效缺陷一个 25 张 bundle，类别训练时拼接；不能使用 flip。统一文件接口：

- `batch.npz`：`images` 为 N×3×256×256 float32 [0,1]，`masks` 为 N×1×256×256 float32 {0,1}。
- `metadata.json`：项目 GeneratedBatch metadata；必须准确记录 method/category/defect_type/source_image_id/seed/generation_config/generation_seconds。Procedural 还须有合法 train/good donor。
- `provenance.json`：`category`、基础 `seed: 42`、`num_samples`、`split_sha256`、`file_sha256`（batch.npz 和 metadata.json 的 SHA-256）、`shared_qc`（原始完整 QC report）。
- 另附生成器训练数据/权重来源、配置、上游与项目 commit、内部过滤统计、总 wall time、环境和失败说明，供 A 检查训练泄漏及成本。

共享 QC 必须在生成的 256px 网格上运行，源 train/good 使用 RGB 双线性缩放至相同网格；不得裁剪或改变几何对齐。A 用相同方式重算前景和 QC。全池只做通过/拒绝判断，不在读取阶段悄悄筛掉坏样本。

B 的现有脚本可直接生成所需格式，必须显式启用共享 QC。示例（将输出路径换成新的共享目录）：

```bash
python scripts/generate_procedural.py --root ~/datasets/mvtec_ad \
  --manifest data/splits/fewshot_n5_seed42.json --category metal_nut \
  --num-samples 75 --image-size 256 --seed 42 --shared-qc \
  --out ~/datasets/robot-generated/procedural_metal_nut_primary
python scripts/generate_procedural.py --root ~/datasets/mvtec_ad \
  --manifest data/splits/fewshot_n5_seed42.json --category screw \
  --num-samples 125 --image-size 256 --seed 42 --shared-qc \
  --out ~/datasets/robot-generated/procedural_screw_primary
```

## 3. A 整批验收

```bash
python scripts/audit_formal_pool.py --root ~/datasets/mvtec_ad \
  --category metal_nut --method procedural \
  --bundle ~/datasets/robot-generated/procedural_metal_nut_primary \
  --out ~/datasets/robot-reviews/procedural_metal_nut_primary
```

条件方法重复 `--bundle`，一次提交同类别所有有效缺陷。工具校验哈希、来源、配额、QC 版本和尝试上限，重新运行共享 QC；按 seed 42 分层抽出 20 对 image/mask/overlay，写出 `acceptance.json`，初始状态始终为 `pending_review`。

A 查看图像并检查生成器训练来源、checkpoint、scope 和无 held-out 泄漏证据后，填写 `reviewed_at`、`notes`、`training_provenance_verified: true`；只有整批通过才将 `status` 改为 `accepted`，`reviewer` 保持 `A`。失败时记录 `rejected` 及原因，不挑样。不可仅为让脚本运行而改为 accepted。

验收绑定 pool、协议、QC 与配置哈希；换数据或换规则后必须重新审查。记录是一份可追溯团队决定，不是加密签名系统。

## 4. 训练与成本测速

先申请学校 GPU，在分配到的计算节点运行训练。测速只使用训练数据，5 步预热后计时 20 步，不产生 held-out 分数：

```bash
python scripts/benchmark_inspector.py --root ~/datasets/mvtec_ad \
  --category metal_nut --device cuda --out /tmp/robot-training-timing.json
```

单次 E0：

```bash
python scripts/train_inspector.py --formal --root ~/datasets/mvtec_ad \
  --manifest data/splits/fewshot_n5_seed42.json --category metal_nut \
  --method real_only --seed 42 --device cuda \
  --out ~/datasets/robot-results/e0_metal_nut_seed42.json \
  --checkpoint ~/datasets/robot-results/e0_metal_nut_seed42.pt
```

E1 增加 `--method procedural`（替换 real_only）、`--bundle` 和 `--acceptance`，输出名也相应改变。E2/E3 同理。正式入口从协议获取 1500/2200 steps；`--epochs` 仅作用于旧非正式模式。完整 batch=8、等权有放回采样，各方法统一。

用 deliveries JSON 列出已交付的方法；路径写绝对路径。例：

```json
[
  {"category":"metal_nut", "method":"procedural",
   "bundles":["/shared/robot/procedural_metal_nut"],
   "acceptance":"/shared/robot/reviews/procedural_metal_nut/acceptance.json"},
  {"category":"metal_nut", "method":"gan", "status":"not_run_quality_gate",
   "reason":"A review: systematic mask/image mismatch"}
]
```

这里的 `/shared/robot` 是示例，不代表已存在或已有权限；GAN 的例示拒绝记录也须以实际 A 审查为准。

```bash
python scripts/run_formal_evaluation.py --root ~/datasets/mvtec_ad \
  --deliveries /absolute/path/deliveries.json --out ~/datasets/robot-results/primary-plan
```

默认仅生成计划；确认路径正确后换一个新的输出目录，加 `--execute` 开始顺序执行。无 deliveries 时只计划 E0 的两个类别、三个 seed，其他条件记录 `not_run_missing_delivery`。不要在 GPU 登录节点执行训练。

运行结果、日志和 checkpoint 保存在外部目录。runner 拒绝覆盖已有输出，遇到失败即停止后续训练并记录状态；已完成运行不要重复。修复后可使用计划中的未完成单次命令运行到新路径。

TC2 提交入口（不传 deliveries 时仅运行六次 E0；日志目录须已存在）：

```bash
sbatch --output=/absolute/path/robot-eval-%j.log \
  scripts/tc2_formal_evaluation.sbatch \
  /absolute/path/clean-checkout /home/mcaai/yuanyuan010/datasets/mvtec_ad \
  /absolute/path/new-run-directory
```

任务先执行测试、GPU Gate 0 和训练测速，再运行正式计划。任务失败时查看日志，不修改 held-out 驱动的参数重试。

## 5. 汇总与解释

```bash
python scripts/summarize_formal_results.py /absolute/path/completed_results/*.json \
  --out /absolute/path/primary_summary.json
```

只传训练结果 JSON，不传 run_plan、acceptance 或其它日志 JSON。汇总器校验正式协议、split、步数、配额及重复 seed；三个训练 seed 必须使用相同 pool。输出每类别均值/样本标准差、配对 E0 差值和两类别等权宏平均。缺失方法标记 incomplete，不填零，不自动降为单 seed。

`defect_recall` 使用任意阳性像素判断异常图，须与 test/good 的 image_false_positive_rate、定位 Dice/IoU 一同解释。三个 seed 的误差仅表示下游随机性，不覆盖生成器 seed 方差。

## 6. 当前边界

- A 基础设施可以执行正式主实验；实际执行状态以结果/任务记录为准。
- B 已合并的 Procedural 每类 30 张记录是接入证据，不是 75/125 主配额交付。
- GAN 仍需整批质量验收；不因训练完成而视作成功方法。
- C 必须提供满足本页文件接口的合成池和训练来源证据。
- r=1/10 次级比例分析尚未接入此严格主实验入口，不能修改主协议配置来绕过配额。须使用单独标记的次级运行入口后再执行。
