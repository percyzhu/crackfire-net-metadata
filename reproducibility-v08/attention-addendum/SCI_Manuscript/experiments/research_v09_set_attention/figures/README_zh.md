# 新集合注意力控制：独立探索性图表

此目录当前为图表程序与方法文字准备，不含结果图或伪造数据。它补充原420次证据，不覆盖七模型主图、统计、工程例子或预算表。

固定输出仅两图、两张12行表和三份CSV：三模型MAE均值/种子/SD，以及matched set−attention、sum GNN−attention的配对差和95%区间。全部12协议保留原计划顺序，正值有利attention，数据单位统一为归档响应误差×10³；5种子不扩大独立几何数。显示设计、来源门槛和复核风险见`FIGURE_CONTRACT.md`，英文图注见`captions_en.tex`，可供整合的方法段落见`methods_extension_draft.tex`。

生成器先只读计划、队列和身份元数据。只有60新运行全部完成、完整聚合报告与其全部来源绑定通过，并且真实独立数值结果审查`independent_full_matrix_review.json`具有约定的PASS状态及该聚合报告/完整性报告SHA，才允许读取MAE/CI。软件代码审查或先前420通过文件均不能替代本门槛。图表生成时还从逐例MAE CSV复核每种子的均值、三模型五种子均值/样本SD和两个配对点差；不在绘图时重新选择协议或改变统计方法。

当前检查命令：

```powershell
python SCI_Manuscript/experiments/research_v09_set_attention/figures/generate_extension_figures.py check-ready
```

完整60聚合与独立结果审阅实际完成后，由执行者给出真实`--audit-dir`及一个不存在的输出目录。下列命令只说明接口，不代表该完整快照当前存在：

```powershell
python SCI_Manuscript/experiments/research_v09_set_attention/figures/generate_extension_figures.py generate --audit-dir SCI_Manuscript/experiments/research_v09_set_attention/evaluation_final60 --review SCI_Manuscript/experiments/research_v09_set_attention/independent_full_matrix_review.json --output SCI_Manuscript/experiments/research_v09_set_attention/figures/complete_reviewed_v1
```

真实未完成矩阵测试显式调用`generate`入口，同时禁止成绩/预测/checkpoint读取、绘图数据准备、绘图、表格写入和创建输出目录。`incomplete_gate_probe_evidence.json`保留第一次实测；第二次包含实际队列计数的证据为`incomplete_gate_probe_evidence_v2.json`。不得为测试完整分支而造假数据、假PASS审查或假图。重复检查必须写新证据文件。

最终导出PDF、可编辑SVG与300dpi PNG，固定180×136mm；自动检查文字越界并保存检查记录。真实PNG/PDF视觉核对、图表独立算术核对、LaTeX表格编译和原主稿中的摆放仍待完整数据后执行。生成记录保持`visual_QA_pass=false`，直至另有实际检查证据；没有自动承诺终稿或期刊送审通过。
