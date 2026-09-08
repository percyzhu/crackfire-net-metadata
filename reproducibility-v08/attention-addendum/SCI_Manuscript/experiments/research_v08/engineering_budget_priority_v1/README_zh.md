# 固定预算优先分析：原七模型探索性结果

本目录属于 **post-review exploratory analysis**：原420次准确性和工程结果已知后，先固定协议，再经根任务批准，才计算预算效用。原七模型隔离计时完成后执行；没有训练、模型推理或有限元计算。原420数据、预测和确认性主要统计未修改；未来第八模型只能在另一个明确计划下追加。

协议见 `SCI_Manuscript/review/eaai_editor/research_v08/engineering_budget_priority_protocol.md/.json`。MD SHA为 `77a8c53efe548b3cfde9620e81b36c089558e9a0e52cc7c86dddb152d12ba24d`，JSON SHA为 `efeaf945c10c7f7563b460f34c82bf7658cb11ff01df1467f1e2c71685d4257d`；根授权记录在 `adoption_record.json`。

## 分析与来源

全部12协议、7模型、5种子，固定终值和J两个端点、10/25/50%三档案例名额。每档选择k=floor(bN)，按最低预测响应优先；精确边界平局使用均匀选例的期望权重，名额总量仍为k。报告预期被选参考均值U、同预算oracle均值、随机无放回选k例的精确期望、oracle regret及相对随机提升，另给可用oracle改善比例L。L不是风险降低或节省时间的百分比。

所有测试集中均未发现置换不变的节点特征多重集、全局特征和时间特征完全相同的不同案例；997个完整图的连通关系逐一核验，节点规范化函数对这997个输入的节点倒序均生成相同规范序列。这不是模型经过997次置换推理的测试，本分析没有重新推理。Fire-only和global-statistics的相同有效输入组按预定规则平均端点预测分数；所有模型使用同一精确平局规则。`input_equivalence_gate.json`是在首次读取预算预测前产生的记录。其较早的 `input_equivalence_gate_before_metadata_path_fix.json` 仅保留检查点文件名修正前的代码绑定证据，当前正式门槛为不带该后缀的文件。

生成器 `budget_priority.py` 固定SHA为 `ef67d81d988545f3fd93ddc715901691a4daa5ff7ea908d3d33ae3d69be6fa32`。命令分别为 `python budget_priority.py --input-gate`、`--analytic-tests`、`--run`。已存在的正式门槛和结果拒绝覆盖；不应为重复执行删除旧记录。

## 已完成的独立核对

`results/independent_arithmetic_audit.json` 使用另一套排序块、精确求和和标量梯形公式，未调用生成器函数。重新检查420个保存预测的端点、99,470条案例端点记录、298,410条选择权重、2,520条种子效用、504条五种子汇总和全部配对差值。生成阶段还逐一核验420个实际检查点SHA、预测SHA、测试顺序及真值/时间；全12协议既有独立审查与最终封存身份保持一致。

解析检查覆盖名额守恒、预测平局的期望选择、穷举小例子的无放回随机均值/标准差、恒值预测等于随机、恒值参考的零regret/零提升及L为NA、负提升的保留。当前真实12个测试人口的两个端点都非恒值，因此实际2,520条L均有定义；这不取消其他阈值分析的删失/NA限制。

## 文件接口

| 文件（均在results） | 行数 | 含义 |
|---|---:|---|
| reference_budgets.csv | 72 | N/k、oracle、随机期望/策略SD、L分母 |
| model_seed_utility.csv | 2,520 | 原7模型每个种子的U、regret、提升、L、平局计数 |
| model_five_seed_summary.csv | 504 | 五种子均值、样本SD、最小/最大值及有效数量 |
| paired_seed_contrasts.csv | 720 | 同种子匹配集合−sum GNN、sum−mean GNN的U差值 |
| paired_five_seed_summary.csv | 144 | 上述差值的描述性种子汇总，不是CI |
| raw_uncanonicalized_baseline_diagnostics.csv | 720 | 两简单基线未规范化排序的诊断，不能择优替换主分析 |
| case_endpoint_scores.csv | 99,470 | 原始/规范化分数、真值、有效输入组、改变量 |
| case_selection_probabilities.csv | 298,410 | 每个案例的选择概率，精确重建平局期望效用 |

`analysis_report.json` 保留执行/来源/输出SHA；该原始完成记录中的“pending independent arithmetic review”是执行结束时的状态，随后独立审查记录单独追加，不改写原报告。

所有效用均使用原始无量纲响应单位；图表若乘100应标为响应百分点。L可按百分数显示，原始数据不应裁剪为0–1。单输入策略的随机提升仅有不超过2.23×10^-16的舍入残差；47条微小负regret保留原值且只在显示层记零，绝不是超越oracle。正文解释和所有固定预算的结果见 `results_interpretation_zh.md`。

十个家族留出试验对应不同拟合权重；本目录未构造跨十族的一个统一筛选模型或预算总效用。IID/数量外推和家族测试人口重叠；五种子不等于五倍独立几何。这里不是整稿送审通过结论。
