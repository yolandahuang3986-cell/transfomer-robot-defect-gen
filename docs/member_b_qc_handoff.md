# Procedural共享QC接入（2026-10-10）

隔离集成测试使用A PR#2 commit39a0800e3809f7318de5e505ddd65ae3a4152628的base/qc/scope/config与B本地代码，不修改共享QC规则。27项原有测试通过。

生成→共享QC重试→require_target→A generated_dataset闭环：
- metal_nut：目标30，尝试32，通过30，拒绝2（过小）；通过率93.75%。
- screw：目标30，尝试33，通过30，拒绝3（过小）；通过率90.91%。

这是generic synthetic_patch类别层面的接入验收，不是正式按语义缺陷分层的实验。Procedural没有bent/scratch等语义控制，不能把贴片改名冒充特定缺陷。正式配额协议需A明确；正式下游指标仍未运行。

B实际仓库scripts/generate_procedural.py新增--shared-qc，要求先具有A PR#2的共享代码。未启用时保留旧行为。失败时写qc_report并拒绝把不足样本当完成。暂未推送GitHub。

b_qc_integration_snapshot.zip是可审查的完整集成源码快照，含A的固定版本共享代码、B实现与测试、冻结清单配置，不含图像数据/权重/环境。不是对A分支的独立PR，不应整包覆盖正在工作的仓库。数据另保存在procedural_shared_qc_gate中，切勿上传GitHub。
