# B 正式样本交接（2026-10-10）

A 已明确正式协议：仅排除 flip，保留 color，共40张异常support；Procedural 保留 synthetic_patch，按物体类别总量对齐。生成 seed42，依共享 helper 的确定顺序保留最先通过者，最多10倍尝试，不人工挑选。重试种子按共享实现确定性递增，不改变其实现。A 负责正式下游训练：seeds42/43/44，metal_nut1500步，screw2200步。

## 本次产物

生成代码 bf798af，干净工作区，共享 QC synthetic-qc-v1。冻结清单不变。目标批次已生成，尚待人工验收：

|类别|目标/通过|尝试|拒绝原因|通过率|生成CPU秒|QC秒|
|---|---:|---:|---|---:|---:|---:|
|metal_nut|75/75|83|8张mask过小|90.36%|1.179|0.358|
|screw|125/125|136|11张mask过小|91.91%|2.310|0.514|

200对通过contract、正常训练source/donor归属、文件哈希和下游generated_dataset读取验证；重新执行共享QC全部通过。mask与实际修改像素一致。以上不等于人工验收或下游收益。提供完整float32 NPZ、PNG、metadata、来源记录、环境和每类固定20张审阅图。

## 人工审核发现：screw 暂停验收

固定20张审阅图中，0009、0082的贴片看起来位于螺丝外侧阴影/背景。虽然通过共享QC，前景估计可能把阴影计入物体。整批screw暂不进入正式训练，交A审核是否为系统性放置失败；没有逐张删除或补数，也没有修改共享规则。Metal_nut同样待A按统一流程人工验收。该发现必须与自动通过率同时报告。

## 交接包

b_delivery_20261010_v1.tar.gz，52,401,531 bytes。
SHA256: 4e62995040b589fba306180f2993d61ced0c9f883c2ff1d83b12d3ae629b589d
包含500个校验条目，以及GAN预览/质量审计和40711消融日志、环境。权重不重复打包，图像不进GitHub。

计划上传至B账号 `/home/mcaai/weiwei003/defect_deliveries/b_20261010_v1`。**尚未确认上传，不保证A跨账号可访问。** 必须以上传脚本的UPLOAD_VERIFIED为准，并通过学校允许的共享位置或文件交接供A读取；不扩大整个home的权限。

GAN历史权重在B账号 `~/projects/sdg/weights/fewshot_*`；40711在 `~/defect_runs/bent_ablation_40711`。原始证据保留，本次不追加训练。E2是否标为生成质量未通过、下游未执行，由A统一验收记录。

## 复现与载入

在PR#3代码目录，使用项目环境及MVTec数据路径；输出路径须不存在：

```bash
python scripts/generate_procedural.py --root "$HOME/datasets/mvtec_ad" --manifest data/splits/fewshot_n5_seed42.json --category metal_nut --num-samples 75 --seed 42 --shared-qc --out "$HOME/defect_runs/procedural_metal_nut_75"
python scripts/generate_procedural.py --root "$HOME/datasets/mvtec_ad" --manifest data/splits/fewshot_n5_seed42.json --category screw --num-samples 125 --seed 42 --shared-qc --out "$HOME/defect_runs/procedural_screw_125"
```

下游接口：`load_bundle(directory, manifest, category)` → `generated_dataset(batch)`。生成包README含完整示例和环境。下游正式结果未运行，由A负责。
