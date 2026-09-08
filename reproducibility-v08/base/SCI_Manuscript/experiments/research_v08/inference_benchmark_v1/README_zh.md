# 推理效率测试：已声明，等待完整计算结束

2026年9月7日已冻结计时选择和程序。此声明在既有准确率结果可见之后、正式计时之前制定，不能称为全部实验开始前的预注册。声明为 timing_plan.json，SHA256：e44ab414c6f7c56a8a400292d6c460bdfb90ba5d2b3636f76ff353575c5eb5a7；程序为上级目录 benchmark_inference.py，SHA256：aad7de1e3e332a119adf5ea97f2813bf47dd8d3fe4d136a4ba60cb096f96bcf9。

选择固定IID协议首个种子42的全部7种模型和全部160个测试案例。每个查询输出完整61时刻曲线。CPU四线程与CUDA分别测试单例和32例批次，三轮测量，轮换模型与设备顺序，所有形状先经过完整样本预热。保存每一批耗时、各轮及汇总的均值/中位数/p95、单例按裂缝数量分层，以及总案例数除以总时间的吞吐量。批处理摊销耗时与单例延迟分开。

分开记录CPU公共特征准备、设备驻留输入的模型前向，以及从内存几何/火灾参数到主机预测输出的完整公共接口。后者包含特征构造、整理、设备传输、前向和返回主机；所有模型构造同一套公共输入，不能解释为各模型单独优化后的部署耗时。磁盘、JSON解析、模型加载、Python启动及网络传输不含在查询延迟内。CUDA测量前后同步；原始记录保留，不通过删去慢点改善结果。

另单列全部420次成功训练的历史主机墙时、实际轮数和参数量。它们包括训练、验证和检查点I/O，且可能受当时并发分析影响；冻结训练程序在最后恢复检查点后没有额外CUDA同步。这些是实际记录的训练成本，不能当作隔离环境下的架构速度。成功运行总和不含失败尝试/重放、进程启动、最终测试及拓扑视图推理，也不是项目总历时。历史FE耗时另列，不与本机推理拼成同硬件加速倍数。

运行入口：

```powershell
& 'C:/Users/admin/AppData/Local/Programs/Python/Python312/python.exe' SCI_Manuscript/experiments/research_v08/benchmark_inference.py run
```

必须先完成420次训练及最终 aggregate_results.py --replay，且其他项目计算/绘图/审核进程已经结束。程序会核验全部420个实际运行文件与最终独立审核，再开始计时；各轮、模型和完成时复查进程、队列及审核来源。若未就绪，只更新 readiness_latest.json，不建立 measured/ 或输出耗时。不要重复执行 prepare，不绕过门槛。

当前验证：160例×5类输入张量逐位重建一致；独立21项CPU模型接口/数值检查通过；完整260条已审重放状态分支核对通过。正式run在188/420时实际返回DEFERRED_NO_MEASUREMENTS，识别仍运行的训练队列及子进程，未建立测量目录。最终420正向运行和结果审核仍待完成。程序门槛只能检测已知项目Python进程，不能证明所有匿名进程或操作系统负载绝对不存在。

独立代码审查见上级目录 inference_benchmark_review_20260907_v2.json，接口检查见 inference_benchmark_api_review_20260907_v2.json。真正完成后还须独立复算原始CSV的分组、单位、样本覆盖及输出完整性，再写入论文；不能把本README或程序准备完成称为效率测试已完成。

独立实测结果检查器已准备并经第二位代理审阅，冻结计时程序和声明均未改变。实测全部完成后运行：

```powershell
& 'C:/Users/admin/AppData/Local/Programs/Python/Python312/python.exe' SCI_Manuscript/review/eaai_editor/research_v08/audit_inference_timing.py
```

它验证最终420次回放及真实产物身份，重新计算14,355条计时记录的完整覆盖、58个总体组、174个每轮组、435个单例裂缝数量组和420条训练成本。缺少真实完整计时会返回WAITING且不读计时记录，不生成实测审查通过报告。源码审阅记录为 review/eaai_editor/research_v08/inference_timing_checker_review_20260907.json；当前检查器SHA256为 cfb768af7addf6cf89095dd5f9a6e7ff2a5331238caf4cb3ee329a54310278f3。请把该独立核对安排在正式计时进程结束之后，避免并发工作使计时门槛失效。
