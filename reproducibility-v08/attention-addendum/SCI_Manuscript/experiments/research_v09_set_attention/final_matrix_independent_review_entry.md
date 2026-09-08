# 新60次探索性扩展：最终独立结果审查入口

**当前只有软件/API审查；这里不给结果或整稿通过。** 原420次主要比较和原七模型预算结果保持独立。新模型在查看原结果之后提出，所有比较必须标为探索性扩展。

最终独立检查器为`audit_complete_extension_matrix.py`，SHA `6757565747f31bf30f599aff2f2fe7c12b3762d9bedfbcf225783e8fa5f24165`。最终候选已由聚合器作者另做有界API审查，记录在`independent_checker_api_review_by_execution.md/.json`；其明确不替代独立数值结果审阅。既有完整聚合器SHA保持`aa06a3280fed95be571f1b39677c84c20d193804897d4f4d55ec7a99053a2e7a`，学习计划SHA为`8d21e13af116dc97556b37060daaf86afed5f7ba645a5bdae18e3d4ffb3b05a9`。

根和执行者已确定唯一正式聚合目录为`evaluation_complete60_v1`。完整聚合及进程结束后，依次调用`python audit_complete_extension_matrix.py --aggregate-directory evaluation_complete60_v1 --check-ready`和同一命令去掉`--check-ready`；这些相对路径须在本扩展目录中执行，也可全部使用绝对路径。检查器默认独占写`independent_full_matrix_review.json`，当前不预写该文件。

与新图表、独立计时采用同一个未来真实通过合同：`status=PASS_INDEPENDENT_NUMERICAL_REVIEW_60_NEW_PLUS120_REUSED`、`numerical_review_passed=true`、`extension_plan_sha256`、`aggregate_report_sha256`、`aggregate_run_integrity_sha256`、`completed_new_runs=60`、`reused_comparator_runs=120`，以及完整匹配的protocols/models/seeds。必须同时保留`submission_pass=false`和探索性状态。完整数值分支实际执行后生成的哈希才可用于后续授权，软件/API审阅记录不满足此门槛。

根任务安排唯一执行者在全部60注册单元和队列完成后运行真实聚合器，并提供新输出快照路径。审阅者应先读取该快照 `report.json` 与完整队列的身份信息。若缺少全部60、最后完整报告或任何回放失败，停止结果解读，保留失败记录，不用部分家族结果代替全矩阵。

## 最终入口所需条件

- 60个新身份=12协议×1新模型×5个固定种子；队列无活动单元且完成。所比较的180个运行由60新模型和120个原匹配集合/sum GNN组成，不是新训练180次，更不是独立几何180倍。
- `report.json`状态为`COMPLETE_EXPLORATORY_60_NEW_PLUS120_REUSED_MATRIX`，绑定固定计划、原完成审计、执行授权、候选聚合器及其实际调用统计源、所有输出SHA。此前软件候选SHA为`aa06a3280fed95be571f1b39677c84c20d193804897d4f4d55ec7a99053a2e7a`；若执行时改变，应先审明确差异，不能沿用旧软件决定。
- `run_integrity.json`全部新60状态通过，保存权重/预测/元数据/日志身份一致；CPU最大差严格小于3e-6或保留差异并有原GPU逐位相同。不存在容差放宽、失败隐藏或伪造回放。原120比较文件须保持原计划/审计/实际文件SHA。
- 各协议都有完整3模型×5种子统计、逐例CSV、时间CSV、固定分层CSV、工程JSON和CSV，保持原有测试ID顺序/61时刻/参考值。

## 完成后有界独立数值审查

参照原420已经使用的独立算术流程，对12协议逐一核对实际预测与CSV、五种子均值/SD/逐例尾部；独立重算两项固定主要MAE差值及5000次几何/种子配对区间。必须保留全部族和正负差值，不能因为注意力是新增模型而自动改变原主要结论。

工程核对覆盖终值/J误差与排序定义、0.8/0.6事件/删失/漏报/误报/回穿/晚报、各模型自身参考/预测共同事件覆盖。按原流程独立重算四项终值/J主要工程差值的1000次CI，核对全部工程CSV与有效重抽样数量。条件时间对比不是两个模型共享事件交集；原来零事件、单事件和Smoldering的86恒值/11非恒值边界应按新实际结果保留。

核验身份、统计和解释后才能发**新60全矩阵结果开发阶段审查决定**。这个决定仍不等于论文通过；需另核对实际图表、主稿与补充材料，且第八模型推理效率必须独立测量后才能填值。原七模型预算结果不在本入口内改写，第八模型预算若追加须明确新计划与版本。

现有软件审查：`independent_aggregate_code_review.md/.json`。强化缺矩阵检查：`independent_missing_matrix_probe.py`与`independent_missing_matrix_probe_evidence.json`，它只触发真实未完成门槛，禁止预测数组/成绩/模型构造/推理/数据载入，不代表完整分支已执行。
