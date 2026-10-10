# B 实验阶段结果（2026-10-10）

## 结论
Procedural已完成本地功能验证，现有60对样本中55对通过A共享QC。官方GAN九组训练均已完成，但默认eval推理存在空mask、错位、伪影和重复模式。关闭background损失的单因素对照没有解决这些问题。尚无正式下游效果指标，不能宣称提升检测效果。

## 40711单因素实验
同一固定seed42，metal_nut/bent，220正常训练图+5support，256px，batch1，两组50轮、25轮起衰减；仅lambda_background为0.1/0。两组使用同一隔离环境，完整依赖已归档。任务COMPLETED0:0，整体耗时01:21:46（包括依赖补装、两组训练和CPU预览，不应当作纯训练时间）。

每个checkpoint使用相同5张正常训练图，eval模式，mask阈值0.5：

|组别|轮次|空mask|共享QC通过|
|---|---:|---:|---:|
|原损失0.1|25|0/5|5/5|
|原损失0.1|50|3/5|1/5|
|关闭损失0|25|0/5|0/5，均前景重叠不足|
|关闭损失0|50|4/5|0/5|

第25轮非空mask仍伴随物体变形、固定位置缺陷及重复模式。移除此项损失不是已验证的修复。本实验单seed、5张预览，不足以排除该损失的所有影响或证明唯一根因。不同环境、种子、训练预算下的原300轮结果不是这个单因素实验的匹配对照。

## 现有样本共享QC审计
采用A PR#2 commit39a0800e3809f7318de5e505ddd65ae3a4152628的原样代码，阈值未修改。该PR本次读取时仍OPEN。最新scope统一排除metal_nut/flip，未更改冻结manifest。

|方法/类别|通过/审计样本|
|---|---:|
|Procedural metal_nut|27/30|
|Procedural screw|28/30|
|GAN300 metal_nut scratch|5/5|
|GAN300 metal_nut bent|0/5|
|GAN300 metal_nut color|5/5|
|GAN300 screw各5组|均0/5|

Procedural拒绝均为mask过小。GAN scratch/color虽通过几何QC，人工观察仍有失真与多样性不足：不能将几何通过等同于生成质量合格，也不能私自筛掉单张再冒充统一协议。
这些是既有样本审计，不是完整generate_with_qc的10倍尝试协议，不是正式填充率或生成成本实验。Screw推理耗时未用于此次成本比较。正式数据应通过共享入口重新生成并记录尝试、拒绝与成本。

## 数据与证据
全部诊断仅使用冻结support和正常训练源图，不依据held-out调参。早期bent/flip实际TC2训练B图像像素和mask哈希核对通过；Screw目前仅验证provenance，未取回实际staged B/mask核验。
原九组权重、日志及40711完整结果位于Mac Desktop对应下载目录，保持在Git之外。
详细QC：shared_qc_audit_20261010.json。

## 剩余工作
- B adapter仍需与A最新scope/QC入口整合、测试并提交；Procedural是generic synthetic_patch，不得伪称能控制真实缺陷语义，需要与A明确配额映射。
- 通过统一入口生成正式数据，按预定统一预算完成Real+Procedural。GAN若在10倍尝试内不足，报告underfilled，不放宽规则或用坏标签凑数。
- 正式下游比较与团队汇总尚未完成；本报告为可追溯的阶段交付，不是完整作业验收。

## 共享QC完整接入补充
固定A共享代码39a0800版本，与B代码在隔离目录集成，原27项测试通过。通过generate_with_qc正式入口生成generic synthetic_patch：metal_nut32次得到30张，screw33次得到30张；下游generated_dataset验证通过。此处为类别层面的接口验收，不是语义缺陷分层正式结果。新增scripts/generate_procedural.py的--shared-qc选项，要求先合入A PR#2。
