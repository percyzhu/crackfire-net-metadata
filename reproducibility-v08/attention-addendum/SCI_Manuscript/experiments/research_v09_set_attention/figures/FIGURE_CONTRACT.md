# 交互集合扩展图表合同

本目录只准备探索性扩展的图表，不替代原420次七模型主证据。当前没有绘图成绩，也不预设哪个模型更好。待展示的问题是：在原12协议和5种子下，增加可跨节点交互的集合控制后，三种已选模型的归档响应误差及配对差值如何变化。图不主张交互机制因果、物理承载力或任意拓扑成功迁移。

后端沿项目现有Python/matplotlib工作流。图型为定量比较；先完整展示三模型，再展示配对不确定性，不增加装饰性示意图。尺寸均180 mm ×136 mm，白背景、8–9 pt字体、PDF/可编辑文本SVG和300dpi PNG。全部12协议按冻结计划顺序保留，IID/LCRO与十个留一家族协议之间设分隔线；不按成绩排序、选族或删负结果。

第一图 `extension_mae`：三个并排、共享横轴的面板分别为matched set、sum GNN和set attention。每行一个协议，左侧标注独立测试几何数n；五个细小点代表五个种子的equal-case MAE，实心大点为五种子均值，误差线为样本SD，不是置信区间。相同线性横轴包括0，以全部均值±SD及种子点的全局范围自动取轴范围，无断轴或误差线截断；若均值−SD小于0，保留该完整SD线段。单位为归档响应误差×10³。

第二图 `extension_effects`：两个共享坐标的森林图，分别显示matched set−attention与sum GNN−attention的MAE差及冻结5000次几何×种子配对bootstrap的95%区间。正值有利attention，负值有利各面板的baseline；零线可见，两面板统一对称线性横轴，端点不截断，不使用显著性星号。五种子仍对应n个独立几何，区间不是逐时点重抽样或5n个独立算例。

另输出两张12行LaTeX表：三模型MAE均值±SD；两个配对差与95%区间。CSV保留原始无量纲数值和×10³显示单位，并保留几何、种子、交叉三类区间。图只展示交叉区间。原420结果继续保留；本扩展复用120个旧比较运行，并新增60个运行，不能称新训180次。

读成绩的必要门槛依次为：固定源/计划；60注册身份及COMPLETE队列；真实完整聚合report；独立数值结果审查PASS且绑定该report/integrity SHA；实际统计输出与来源SHA匹配。软件代码审查PASS不能替代结果审查。未达到门槛时不读summary、CSV、NPZ或checkpoint，不创建成图目录，不生成假图。

最终独立结果审查JSON合同：`status=PASS_INDEPENDENT_NUMERICAL_REVIEW_60_NEW_PLUS120_REUSED`、`numerical_review_passed=true`，并有`extension_plan_sha256`、`aggregate_report_sha256`、`aggregate_run_integrity_sha256`、`completed_new_runs=60`、`reused_comparator_runs=120`、完整`protocols/models/seeds`。这是未来真实审查的输入合同，不是预先写入的通过决定。

成图前从逐例MAE CSV独立复核五种子均值、样本SD和两个配对点差；只采用已通过审查的CI，不在绘图时另调抽样。最终还要真实渲染、检查全部文本边界、阅读实际PDF/PNG、核对图表数值和LaTeX编译，不能把缺矩阵测试当成完整图表验收。
