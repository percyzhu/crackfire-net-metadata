# 交互集合控制的独立后续扩展

2026-09-07 09:23:17 UTC，根任务独立软件审查并授权后，已启动全部60次固定队列，PID43364。首次健康检查在首个IID seed42的第54epoch完成，状态正常，工作进程与单元日志均无错误输出；这不代表60次完成。原420完整比较、源码、数据、划分、权重和预测全部保留。本扩展由看到完整420结果后的EAAI内部应用AI预审触发，标记 `AFTER_ORIGINAL_TEST_RESULTS_REVIEW_EXPLORATORY`，不能称原研究预注册。

固定结构：原小集合的11→64节点编码，四个宽64、4头、FFN宽104的pre-LayerNorm残差self-attention块（ReLU、无dropout、无位置编码）；严格mask padding并对有效节点均值池化。保留原全局融合、时间分支和decoder。自注意力含自身key/value；原GNN无自环但有残差，两者是已明确的架构差异。边输入被忽略，不能把集合模型对改边不敏感写成学习到的拓扑迁移。

参数194,273，比原GNN194,177多96（0.0494%）。FFN104仅按参数量解析匹配，不使用模型成绩搜索。所有参数参与前向与反向。模型比较帮助判断所选表示之间的相对性能，仍不能单独证明机制因果。

60次新训练＝1固定结构×12个原协议×5个原种子。节点/全局/时间输入、997目标、划分、AdamW/学习率/衰减、batch32、最大300epoch/patience50、梯度裁剪、最早最低validation选择与原计划相同。新结果与原matched set和sum GNN的120个已验证运行配对比较，不重新挑选其权重；原七模型的结果继续保留。新增两个探索性配对差为matched-set减attention及sum-GNN减attention，正值有利attention，沿原几何/种子5000次区间方法；完整60次审计前不据部分结果调参或选取家族。

软件检查使用四个原IID训练几何N=1/3/8/15，检查forward/backward、节点置换、批次组合/顺序、padding与去边、真实跨元素梯度、0/16节点范围拒绝及validation最低同值选择。检查没有optimizer step，不报告初始模型对科学目标的准确率。原始短检查保留在`software_smoke_cpu_initial.json`；当前冻结源码的CPU复核在`software_smoke_cpu_final.json`，最大不变性差8.94e-8，padding/去边差0，全部参数具有有限梯度，共同全局/时间/decoder分支初始权重与matched-set逐位相同。根任务的独立GPU前反向/不变性审查也通过，记录在`review/eaai_editor/research_v08/set_attention_independent_pretraining_review.json`。

正式执行前的顺序已完成：原7模型隔离计时及独立审计通过；最终CPU软件检查、新计划及独立GPU/代码审查通过；根任务写入绑定来源的`execution_authorization_root.json`。最初无授权的真实门槛检查阻止训练，授权后再次实际检查通过。新attention的推理成本不能从原7模型计时推算。

固定计划`plan_set_attention_60_v1.json` SHA256为`8d21e13af116dc97556b37060daaf86afed5f7ba645a5bdae18e3d4ffb3b05a9`。授权SHA256为`9c56cea3a2310b0956ea7e6429ee1ec6a23f940d3af1b58842129043e6448908`。`launch_record_v1.json`记录启动时间、PID、实际入口及源绑定。

只读完整性门槛（没有60次完整就返回等待，0成绩载荷/0预测读取）：

```powershell
python SCI_Manuscript/experiments/research_v09_set_attention/aggregate_extension.py --check-ready
```

计划与原12源码独立绑定；训练已启动，冻结训练源码、架构、数据和计划不得修改。队列采用单GPU子进程和不显示窗口的子进程启动标志；原子状态写入遇到共享锁时最多重试10秒，实际错误停止，不自动重训失败单元或覆盖已有尝试。

`aggregate_extension.py`独立于冻结训练源码。先核验所有60次身份/状态/文件存在与COMPLETE队列；全部齐备才读取成绩、加载checkpoint并复算。每个新运行须通过计划/数据/特征/目标/划分/授权/源绑定、194273参数、最早验证最低checkpoint、预算/余弦调度/早停及有序预测/真实标签核查。CPU回放保留原严格3e-6门槛；超阈须原GPU逐位相同并记录CPU差异，不放宽容差。原120比较文件由新计划预先绑定，并要求原420最终独立审计已通过；不覆盖或重跑原420汇总。

完整审计后，每协议输出三模型的主指标、逐例/时刻/来源分层CSV、两个5000次几何/种子/交叉配对主CI及1000次工程次要CI。次要指标沿原终值、时间平均J、0.8/0.6事件/删失/NA政策；五种子聚合仍只有N个独立几何。条件越界时间比较使用各模型自己的共同事件条件，不能宣称同一队列上的时间改善。所有12协议完成后才可解读相对性能，不做试验中调参。该检查器的完整分支与新CI仍需实际完成数据和另一智能体独立复核；软件通过不代表研究或投稿审查通过。
