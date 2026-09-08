# v08应用研究稿：方法与定位开发评阅

2026-09-07，DEVELOPMENT_REVIEW。审阅对象为`latex_v08/main.tex`及各节源文件，结合冻结420次计划、997例独立审计、评价实现与AGENT_v08；绑定文件见`development_prose_review_snapshot.json`。这不是最终PDF版式或完整研究审阅。未把预期中的结果占位计作本阶段缺陷，未阅读训练预测/成绩，也不授予论文通过。

**总体判断：新定位与方法主线成立，可继续训练与成稿。** 正文已经准确区分档案指数、11例同源定义诊断与11项受控FE设置；N9–15与批次变化、smoldering恒定状态分布、同一checkpoint的稀疏视图均如实说明。相关工作没有残留0.005或20%内部数值门槛作为档案研究的无条件阻断。未发现需要停训的错误。

## 一项需要立即改正的实质措辞

**D08-P01｜相关工作仍有一处把物理几何历史赋予当前档案量。** `01_related_work.tex`第10行：

> For the present quantity ... an early crossing can change later section geometry even after cooling.

当前监督量只对归档标量取累积最小，不能因某点早先越过阈值就称其后续几何被永久去除。建议改为：“For a physically reconstructed section quantity, a threshold crossing can alter the subsequent retained domain. The archival index studied here preserves a scalar envelope rather than that material-point history; the distinction motivates the source-matched diagnostic and limits its physical interpretation.” 文献段的物理动机可保留，但需明确其与α_archive的差别。

## 两项精确性与交付对齐提醒

**D08-P02｜旧验证误差的名称和来源。** `02_engineering_data.tex`第19行的36/43/40°C应明确为本科论文所报告的**RMSE**，不要只称errors。已经注明未重新复现实验误差是正确的。实验来源Zheng等与本科论文生成的旧模型RMSE应保持分开；建议在补充材料或来源表绑定本科论文章节与表号，避免读者误以为这三个RMSE出自引用的实验论文。本项不要求新增FE计算。

**D08-P03｜区间与尾部指标需在最终执行链闭合，不能仅靠方法段或辅助函数存在。** 主体的双向配对bootstrap与`aggregate_results.py`实际调用一致：5,000次，保留geometry-only、seed-only和two-way区间及各种子效应，均未重采样单独时间点。该脚本的quantile输出目前是所有case×time上的正误差q95/q99；论文另称“per-case error quantiles”，应从已输出逐例MAE表明确计算并标明其分布，不能把两者混同。

工程代理CLI独立输出35个完整运行的逐种子、分层指标，**不会自动产生置信区间**。辅助`comparative_summary`可以生成配对重采样数组，默认1,000次；最终如果展示工程指标区间，需实际调用并绑定同一IDs、顺序、随机种子及重复次数。两模型区间不能用各自端点相减代替配对效应区间。此为完成已声明统计的执行检查，不改变主要指标或增加训练协议。

## 已检查并认可的要点

- 420=7模型×5种子×12协议，图与匹配集合参数数目及0.379%差异正确；均值/求和图只改变聚合，无额外参数。
- 四轮参数跨节点/边共享但不跨轮绑定；孤立节点仍更新；完整图边数N(N−1)正确。零边特征图未被误称无图基线。
- 图/集合共用可取得的裂缝信息；时间分支与解码器架构相同。固定单配置而非最优调参的限制已写明。
- 全场景在预测前已知，时间描述量不是实测热通量；档案公式与逐例实际Abaqus幅值尚未全量对齐的限制已写明。
- 61点标量变换顺序、原始空腔缺失、初值为1、不可逆点历史与标量包络的区别均明确；恢复常值例没有被当作坏样本。
- 格点阈值事件保留删失，界限不被写成耐火极限；神经预测未被强行单调化；相同来源/火灾下的排名不会被误称裂缝因果影响。
- 次要工程指标准确称为训练开始后、相关预测成绩未阅前声明；没有追称全部在训练前注册。
- 公共数据及许可尚待最终处理的事实保留，未假称已经公开。

建议先修改D08-P01和D08-P02，随后继续既定计算与结果装配；D08-P03随完整协议统计一起关闭。本次开发评阅不把最终计算、公开发布或正式全文复审提前标记完成。
