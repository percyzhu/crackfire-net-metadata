"""Format only the completed, independently audited extension into a review note."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = HERE / 'evaluation_complete60_v1'
audit_path = HERE / 'independent_full_matrix_review.json'
audit = json.loads(audit_path.read_text(encoding='utf-8'))
assert audit['status'] == 'PASS_INDEPENDENT_NUMERICAL_REVIEW_60_NEW_PLUS120_REUSED'
assert audit['numerical_review_passed'] and audit['completed_new_runs'] == 60
assert audit['reused_comparator_runs'] == 120
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

rows = []
table = ['| 协议 | 几何数 | 注意力 MAE | 集合−注意力 [95% CI] | sum−注意力 [95% CI] |',
         '|---|---:|---:|---:|---:|']
for reviewed in audit['protocol_reviews']:
    protocol = reviewed['protocol']
    source = EVAL / protocol / 'summary.json'
    summary = json.loads(source.read_text(encoding='utf-8'))
    attention = next(m for m in summary['model_summaries'] if m['model'] == 'set_attention')
    row = dict(protocol=protocol, independent_geometries=reviewed['independent_geometries'],
               attention_equal_case_MAE=attention['metrics']['equal_case_MAE'],
               effects=reviewed['effects'], source_summary_sha256=sha(source))
    rows.append(row)
    formatted = []
    for key in ['attention_vs_matched_set', 'attention_vs_sum_GNN']:
        effect = reviewed['effects'][key]
        lo, hi = effect['two_way95']
        formatted.append(f"{effect['point']*1000:+.6f} [{lo*1000:+.6f}, {hi*1000:+.6f}]")
    table.append(f"| {protocol} | {row['independent_geometries']} | {attention['metrics']['equal_case_MAE']['mean']*1000:.6f} | {formatted[0]} | {formatted[1]} |")

note = '''# 新60次集合注意力扩展：独立完整结果短评

数值开发阶段核查通过，可进入结果图表和正文整合；**不是整稿或期刊送审通过**。新增模型是在原420次结果已知后提出的探索性扩展，不能替换原主要对比或将本轮未校正的区间当作跨12协议的确认性优越性检验。原七模型预算与原七模型计时保持独立。

已实际检查60个新运行与120个复用对照，180个保存预测文件、21315逐例行、3720分层行；独立重算24项主差值的5000次几何/种子/双向配对区间，以及48项终值/J差值的1000次区间。其他工程点指标与全部工程CSV、有效样本和NA一致；其余排序、时间、分类区间未逐一独立重抽样，不能扩大该核查范围。身份、权重与预测哈希、既有真实回放记录均一致；新增60中35次CPU超原容差均保留且原GPU逐位一致，审阅者未重复GPU推理。

## 完整协议结果

下表全部误差和差值**乘1000**，不是百分数或百分点；正差值表示注意力误差更低。区间为5000次种子与几何共同重抽样的95%分位区间，未做跨协议多重比较校正。每列均为5个相同种子的配对结果，几何数不是五倍预测行数。

TABLE

IID的注意力MAE为0.009173537，匹配集合为0.008502947，5个种子均为集合误差更低，差值区间完整低于零。与sum比较的IID区间仍跨零，其上端为+0.00000177654，不能将接近零或绘图舍入解释成显著负值。

数量外推中注意力MAE为0.011775828，sum为0.011139067；sum−注意力为−0.000636761，95%区间[−0.001214508,−0.000007922]，4/5个配对种子sum误差更低。注意力与匹配集合的区间跨零。这支持当前实现和固定复合分布变化下sum的误差优势；不能归结为图结构不可替代，或声称已隔离纯节点数量、源批次及显式边信息的因果贡献。注意力能处理变长节点集合，但本轮未在注意力模型上重新做稀疏邻接实验；原GNN同权重稀疏图结果仍是连接方式敏感性检查，不是有限元网格迁移验证。

十个留一火灾族中，Decay的两项区间均完整正向，Plateau的两项均完整负向；其余八族两项均跨零。Bilinear有注意力更低的点估计，Linear则有更高的点估计和较大种子波动，均不宜挑选点估计宣布总体优胜。必须保留全部族，不能把十个独立训练的留族模型写成一个通用模型。

## 工程误差与可用场景

0.8/0.6仅为档案响应代理的审计阈值，越界时间位于60秒输出网格；它们不表示结构失效、耐火极限或规范安全等级。首达时间误差只覆盖参考与该模型均越界的病例，条件覆盖须与误差一起解释。

- IID注意力在两阈值均无漏报/误报；数量外推0.8有0–1例漏报，对应225–226/226个参考事件共同覆盖，0.6无漏报/误报。数量外推0.6仍有1–3例预测越界后回穿，低平均误差不保证曲线单调。
- Bilinear注意力在0.6的73个参考事件中无漏报，但4个参考未越界病例中误报3–4个；0.8/0.6分别有4–11/9–15例预测回穿。整体MAE下降的点估计不能消除这些曲线与筛查缺陷。
- Linear注意力在0.8漏报10–33/122个参考事件，实际共同覆盖89–112；在0.6漏报0–6/92例并有0–1例误报。0.8的条件时间MAE为202.5–548.1秒，必须同时显示上述覆盖，不能当作全部病例的时间精度。
- Smoldering的86恒值/11非恒值结构未变。注意力全97例MAE为0.012516568，86恒值为0.008001549，11非恒值为0.047815812；五个种子没有正向响应过预测，主要为响应低估。0.8只有1个参考事件，各种子误报12–26例；840–1020秒的条件时间误差仅来自该1例。0.6参考零事件且预测零事件，FPR=0及共同事件占总人口比例0可定义，召回率、平衡准确率、参考事件共同覆盖率和条件时间误差均应为NA，不能报告完美识别。原3655/4656个参考响应相等对被排序指标排除（78.5009%）的边界继续成立。

本轮适合支撑“在固定档案模拟响应上比较可变规模集合/图模型，并显式披露外推和筛查失败”的工程应用论点。现有证据不支持注意力普遍优于原方法，也不支持未经现场、试验或修正温度场校准的安全判断。新增注意力尚无预算优先级分析和独立计时数值；原七模型结果不能借给第八模型。真实计时需另按已审计划在根授权后完成。

## 绑定记录

- 独立检查器：`audit_complete_extension_matrix.py`，SHA CHECKER。
- 实际完整检查结果：`independent_full_matrix_review.json`，SHA AUDIT。
- 唯一完整统计：`evaluation_complete60_v1/report.json`，SHA REPORT；身份审计SHA INTEGRITY。
- 新计划SHA PLAN；状态`AFTER_ORIGINAL_TEST_RESULTS_REVIEW_EXPLORATORY`。

本短评只格式化已经通过逐例独立算术核对的数据，不进行训练、模型推理或追加假设检验。
'''
for key, value in {'TABLE':'\n'.join(table), 'CHECKER':audit['checker_sha256'], 'AUDIT':sha(audit_path),
                   'REPORT':audit['aggregate_report_sha256'], 'INTEGRITY':audit['aggregate_run_integrity_sha256'],
                   'PLAN':audit['extension_plan_sha256']}.items():
    note = note.replace(key, value)
out = HERE / 'independent_full_matrix_results_review.md'
with out.open('x', encoding='utf-8') as f:
    f.write(note)
record = dict(status='DEVELOPMENT_RESULTS_REVIEW_COMPLETE_WITH_SCOPE_LIMITS', submission_pass=False,
              numerical_review_passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
              epistemic_status=audit['epistemic_status'], independent_review_sha256=sha(audit_path),
              report_sha256=audit['aggregate_report_sha256'], run_integrity_sha256=audit['aggregate_run_integrity_sha256'],
              extension_plan_sha256=audit['extension_plan_sha256'], source_formatter_sha256=sha(Path(__file__)),
              markdown_sha256=sha(out), units='Raw dimensionless archival response; markdown table multiplies all errors and differences by1000.',
              effect_sign='comparator_minus_set_attention; positive favors set_attention for MAE',
              primary_intervals='5000 paired two-way bootstrap; exploratory, no multiplicity correction',
              rows=rows, blocking_numerical_issues=[], manuscript_required_boundaries=[
                  'No universal attention superiority; preserve IID negative, LCRO negative against sum, Plateau negative, Decay positive, and all crossing intervals.',
                  'LCRO is compound count/source shift, not isolated causal topology benefit.',
                  'Retain threshold FN/FP/recross and model-specific conditional event coverage.',
                  'Smoldering 86flat/11nonflat, single0.8 event and no0.6 reference events; preserve NA.',
                  'No transfer of old7 budget/timing values to new8th model.',
                  'No whole-manuscript or journal submission PASS.'])
with (HERE/'independent_full_matrix_results_review.json').open('x', encoding='utf-8') as f:
    json.dump(record, f, indent=2, ensure_ascii=False)
print(json.dumps({'markdown':str(out), 'markdown_sha256':sha(out), 'status':record['status']}, indent=2))
