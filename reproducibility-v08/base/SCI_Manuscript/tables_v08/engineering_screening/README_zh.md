# 工程指标两张简表：准备入口

本目录准备补充材料中的两张可读简表，没有生成未完成数据的结果表。目标是尽量在一页展示；若真实编译需要更多空间，优先可读性，不缩成难以辨认的小字。主稿可以只引用这两张表。

第一表12行，列出IID、裂缝数量外推及十个火灾家族留出协议的独立测试几何数，以及**容量匹配集合−求和GNN**的终值MAE、时间平均指标J的MAE差和现有95%次要区间。正值有利于求和GNN。显示值乘1000、保留三位小数；CSV保留原始尺度和精度。区间沿用1000次几何×种子共同重采样，不重新计算或选择区间，不对多个次要指标作确认性显著声明。

第二表24行（12协议×两个阈值），显示参考事件E/删失C以及两模型各自五种子中的最大漏报、最大误报和最小共同事件覆盖。共同事件记为`n_common`，不使用已表示时间平均的J。它是某一种子内**参考与该模型**都越界的案例，不是两个模型的共同交集，也不是跨种子共同交集。最大FN、最大FP与最小覆盖可能来自不同种子，不能拼成一个最差checkpoint。事件按0、60、…、3600秒网格首个响应≤阈值定义；删失不填成3600秒事件。

五种子共享同一组测试几何。均值先在种子内按案例计算，再跨五种子平均；不将5N作为独立样本量，也不将不同协议的N相加。参考E/C仅记一次。E=0时覆盖记NA，未定义的FPR、recall和条件时间也保持NA。计数为零可以是真实观察，但没有相应类别时不能把比率填成零。全部七模型均保留伴随CSV：

- `scalar_by_seed.csv`：420行，终值/J的MAE与bias，逐模型逐种子。
- `screening_by_seed.csv`：840行，两个阈值的全部原始计数、覆盖、条件时间与回穿指标。
- `scalar_model_summary.csv`：84行，五种子均值与样本SD；SD不是置信区间。
- `screening_model_summary.csv`：168行，极值、五种子均值及有效种子数；只要有种子未定义，`mean_over_all5_seeds`保持NA，不悄悄平均其余种子。
- `matched_set_vs_sum_gnn_secondary_ci.csv`：24行，原尺度差值及两个端点，显式记录次要1000次区间。

不导出Spearman/rho。External fire中fire-only在同一火灾输入下的微小浮点差异不能表示几何排序能力；原冻结排序结果不在此处修改或重新计算。

## 执行门槛

从项目根目录运行：

```powershell
python SCI_Manuscript/tables_v08/engineering_screening/generate_engineering_tables.py
python SCI_Manuscript/tables_v08/engineering_screening/generate_engineering_tables.py --generate
```

前一条只检查门槛；后一条也必须先过同一门槛。要求全部12协议各35个结果、最终420回放汇总、每协议工程CI JSON/CSV、独立统计复算JSON及短评均齐备。第一阶段只读计划/总报告元数据与文件存在性。联合齐备后再读取CI/复算文件的元数据头，停在数值载荷之前；全部状态合格后才进入成绩读取与逐源绑定。IID沿用原独立审阅文件名，其余协议使用`_independent_results_audit_v2.json`。

缺完整条件返回exit2并写`gate_status.json`，成绩载荷读取数为0，不创建`generated/`。完整时仍须核验计划、数据、12个冻结源、划分、最终回放、420个预测/检查点SHA以及独立复算对CI/主要统计的来源绑定。只读取既有工程指标，不打开预测数组，不重新预测、bootstrap、训练、计时或解FE。

成功生成的`generated/`不覆盖旧版本；包含两张LaTeX表、caption、独立`preview.tex`、五个CSV和来源清单。状态仍是`TABLES_GENERATED_PENDING_INDEPENDENT_AND_LATEX_QA`，不能据此称为最终表格通过。随后应从该目录编译`preview.tex`，独立复算导出的CSV/LaTeX数字并检查真实页数、溢出与可读性，再由根任务纳入稿件。当前完整数据分支、生成结果复核、LaTeX实际编译及图面排版均待验；没有示例数字或占位结果。
