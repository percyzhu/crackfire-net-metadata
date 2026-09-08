# 注意力扩展的独立计时声明（尚未测量）

此目录封存一个新的三模型共同计时批次：matched set、sum GNN、set attention。原七模型14355条真实记录及其审阅文件保持原样，新旧时段不混池、不替换，也不从两个时段中挑选有利指标。新批次内的两个旧模型是同期比较锚点，不能用旧七模型时间估算注意力速度。

固定同一IID split的全部160个测试几何、首个种子42，CPU4线程和CUDA，batch1/32，三轮。三模型按原固定顺序逐轮旋转，各位置恰出现一次；设备顺序为CPU/CUDA、CUDA/CPU、CPU/CUDA。所有三模型与接口都完成一次全160例预热和额外10个批次；每次CUDA计时前后同步。代码保持float32，图索引int64，禁用CUDA matmul TF32并保留原cuDNN TF32设置，使用原确定性配置。

保留两种接口：预先整理并放在设备上的共同输入到设备输出的forward；从内存中的几何/火灾参数重建五个共同特征张量、collate/传输、forward直到CPU NumPy输出。后一接口包含各模型可能并不使用的节点/边特征准备，不能解释为各架构分别优化后的部署成本。另独立记录共同CPU特征准备，不能用总时间减forward估计它。每个case是一整条61时刻响应曲线，不是单时刻预测。

来源接口已经逐段核对。原benchmark的`regenerate`、`verify_features`、`sync`、进程检测和回放状态判定函数可只读复用；原`prepare/run/validate_selection`硬编码七模型和原输出目录，因此没有调用这些驱动，也没有修改或临时替换原模块常量。新attention通过相同`model(*batch)`接口调用，其原始padding/节点打包实现完整计入成本，不作速度导向源码修改。

固定记录数为5940个模型批次时间，加495个独立CPU特征准备时间，共6435；26个总体分组、78个逐轮分组。所有原始秒数保留；汇总均值、p50、p95批次毫秒数、总样本数/总耗时吞吐量以及总毫秒/样本数摊销延迟。batch1另按N分层，不能把batch32摊销延迟当串行单例延迟。新60/旧120的已记录训练成本留在独立完整聚合中，本计时不重新测量或改称隔离训练成本。

`timing_plan_v1.json`于新attention成绩尚未解读时声明，绑定独立计时源码、原已审计benchmark源码、冻结学习源码、数据与IID划分、三个首种子checkpoint/元数据SHA和原时段证据。声明不是正式计时，尚无延迟结果。

执行必须同时具备：全部60注册运行及COMPLETE队列；绑定真实预测回放的完整聚合；真实独立数值结果审查通过；所有新60和旧120实际文件与完成审计一致；当前无检测到的项目Python训练/审计/绘图负载；另有根任务的计时专用授权。旧训练授权不能用于计时。完整文件只在开始前扫描一次，各轮/模型仅复查轻量队列/源/最终证据哈希和并发任务，避免在时间采集中重复哈希整批权重。进程识别不能证明匿名计算和所有其他系统负载都不存在，环境记录会保留这项限制。

准备和真实负面门槛检查已分别提供接口：

```powershell
python SCI_Manuscript/experiments/research_v09_set_attention/inference_benchmark_extension_v1/benchmark_attention_extension.py check-ready
```

完整计算/独立复算完成后，由根写入计时专用授权JSON，必须含`status=AUTHORIZED_SEPARATE_ATTENTION_TIMING_AFTER_COMPLETE60_REPLAY_AND_QUIET`以及`timing_plan_sha256`、`extension_plan_sha256`、`aggregate_report_sha256`、`aggregate_run_integrity_sha256`、`independent_review_sha256`。此说明不创建授权文件；当前不能正式执行。

根将实际确认后运行`run --audit-dir <真实完整聚合目录> --review <真实独立结果审查JSON> --authorization <根计时专用授权JSON>`。本目录已有任何`measured`尝试时都拒绝覆盖；失败保留日志/部分原始记录，不能自动重测或报成功。测完6435记录之后还需要独立覆盖与算术复核、再生成比较图表，不把完整代码检查或声明当成实测效率。
