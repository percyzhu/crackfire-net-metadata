# 完整60次结果独立检查器：准备记录与执行步骤

`audit_complete_extension_matrix.py`已准备，最终候选SHA为`6757565747f31bf30f599aff2f2fe7c12b3762d9bedfbcf225783e8fa5f24165`。它不导入聚合器、学习模型或其统计函数。当前只完成语法、数学恒等式及真实缺矩阵门槛检查，**完整结果分支尚未运行，未创建`independent_full_matrix_review.json`或任何新结果通过文件**。另一智能体的最终API审阅见`independent_checker_api_review_by_execution.md/.json`，没有未解决阻断问题。

`independent_checker_preparation.json`记录真实子进程返回码：解析测试0、未完成门槛2，两次stderr均为空。原始两个缺矩阵证据分别保留在`independent_checker_missing_matrix_gate.json`及`..._v2.json`；读取成绩JSON、预测数组、保存张量及模型推理计数全为零。这里的“解析测试”是明确构造的数学例子，用于检查程序，不是替代真实数据的科学结果。

旧准备证据保持原样；最终版本复测与变更范围见`independent_checker_preparation_revision2.json`。相对旧版只补强失败报告拒绝、未注册结果路径拒绝，以及全部次要CI的次数/有效比例/可估计状态/有限有序端点检查，没有改变任何统计定义。

## 检查范围

- 首先核验固定计划及原/新源SHA、所有60注册元数据、完成状态和无活动任务的完整队列，再要求唯一正式聚合快照包含全12协议结果与最后报告。未完成时立即返回等待。
- 60新模型+120原比较模型的实际文件哈希、审计身份、原完成审计及新日志验证最小值/停止/学习率规则；CPU严格`<3e-6`或原GPU逐位一致的分支逐条核对，不重复GPU推理。
- 实际180个预测、原有序测试ID/61点时间/共同真值，全部逐例与逐时间CSV，全部16主度量及五种子均值/样本SD，固定火灾/数量/批次/保留-恢复/恒值-非恒值分层。
- 两个固定差值均为原比较模型MAE−set-attention MAE，正值有利注意力。独立复算每协议两对比、5000次几何/种子/双向配对bootstrap，保持几何顺序和两维随机权重，未把5N作为独立人口。
- 独立核对全部工程点定义、排序及平局、0.8/0.6事件、删失、回穿、条件时间及共同事件覆盖；独立复算每协议四项final/J工程CI的1000次共同重抽样。全部48工程点差值/JSON-CSV/有效数量与NA规则均核对。其余分类/时间/排序CI算法已在软件审查中确认，检查器不谎称重新计算了全部这些CI的重抽样。
- 绑定先前独立GPU/CPU的软件、梯度及掩码检查。完成后另检查全部997个输入的1–15节点、11节点特征、3全局特征、61×4时间特征和非空valid-key mask。这不是新推理、重新测试模型置换等变性或任意节点数量外推。

## 唯一执行步骤

1. AI执行者完成全60之后，仅运行一次正式`aggregate_extension.py --replay`，向根任务提供新的聚合快照路径及report/run_integrity的SHA。不要并行重复汇总。
2. 独立审阅者先运行`audit_complete_extension_matrix.py --aggregate-directory <实际聚合快照> --check-ready`。只有`READY_FOR_INDEPENDENT_FULL_MATRIX_REVIEW`才进入下一步。
3. 运行`audit_complete_extension_matrix.py --aggregate-directory <实际聚合快照>`。默认只在全部检查成功后，以独占方式创建扩展根目录的`independent_full_matrix_review.json`。任何失败均需诊断，不可删除旧证据或改容差后自动重跑。
4. 实际审查通过文件与图表门槛已统一字段：`status=PASS_INDEPENDENT_NUMERICAL_REVIEW_60_NEW_PLUS120_REUSED`，`numerical_review_passed=true`，固定plan/report/run_integrity SHA，60新/120复用，完整protocols/models/seeds及`submission_pass=false`。这是未来成功分支的合同，当前没有该文件。
5. 最后结合完整结果审阅注意力与两基线的真实正负结果、条件事件覆盖、阴燃人口、输入/来源范围并核对图表和论文。新模型仍属原结果已知后的探索性扩展，不取代原420主比较，不重写原七模型固定预算分析；第八模型计时也不能用原七模型成本代填。

检查器的实际完整分支与图表编译仍待未来60全齐后执行，软件准备通过不预示任何新模型的科学优越性。
