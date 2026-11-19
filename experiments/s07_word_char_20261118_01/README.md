# S07-01 Word/Char 消融对照

实验于 2026-11-18 启动，2026-11-19 汇总验收，保留原运行编号。仅改变启用的特征分支，未修改默认模型、C、n-gram、大小写或 class_weight；组合复用已验证的 S06-B，没有重复训练。

| 方案 | OOF BA | 相对组合百分点 |
| --- | --- | --- |
| Word+Char | 0.307437236 | 0 |
| Word-only | 0.262612478 | -4.4825 |
| Char-only | 0.314479244 | +0.7042 |

Char-only 暂为研究候选。它在 4/5 折高于组合，但 prompt 配对重采样的差值区间包含 0，不自动替换 V1.0 默认基线或推断公榜收益。Word-only 的较低成绩完整保留，它仍提供 227 条 Char 没识别对的样本，且本机成本最低。

- report.md：完整三组比较、全部类别正确数与召回、混淆、互补性、条件性不确定性及成本；同一报告也保存于 docs/S07-01_消融实验报告.md。
- overall.csv / fold-comparison.csv：整体和相同五折的成对指标。
- class-comparison.csv：18 类正确数、支持数及三组 Precision/Recall/F1。
- confusion-changes.csv：全部非对角错误方向计数。
- feature-dimensions.csv / resources.csv：逐折实际词表/系数宽度、耗时与资源。历史组合维度是由同折实测两支宽度相加推导，不是旧日志直接观测。
- complementarity.json：四格统计与真值辅助 Oracle。Oracle 只在两个固定预测中事后择优，不是可部署或保证达到的融合成绩。
- prompt-bootstrap.json：固定 OOF、整组 prompt 配对重采样 5000 次，不是逐行重采样或独立测试显著性检验。
- input-artifact-sha256.json / validation-summary.json：输入产物来源与实际验收结果。

两组正式完整 OOF、18 维分数、源码、警告、审计和逐折产物位于 artifacts/s07_word_only_v1_20261118_01 与 artifacts/s07_char_only_v1_20261118_01，完整清单在各自 artifact_manifest.json；冒烟使用独立编号，不登记为正式成绩。新报告/分析脚本不做训练，沿用公共 run_experiment 入口。

498 个历史数据/产物与默认配置保持不变。S07-01 的本轮 Word/Char 部分完成，原复合任务的分类器对照尚未实施；其他 S07 优化及新 Kaggle 提交均未开展。
