# S05 全量训练与本地提交候选

运行 `s05_full_baseline_20261118_01` 对应已验收基线 `s04_baseline_v1_20261118_01`，不是新的独立泛化实验，没有 Kaggle 成绩。2699 条全量训练、3 条真实预测、Parquet 写后验证与完整 Pipeline 重载一致性检查通过，V0.4 本地目标达到。

- `config.json`：与 S04 完全一致的配置及类别/分数映射。
- `run-summary.json`：实际环境、耗时、文件大小及指纹，使用仓库相对路径。
- `validation-summary.json`：13 组 S05、14 组 S02、7 组 S03、4 组 S04 及 S01 冒烟等验收事实。
- `capacity_report.json`：0/1/3/17/257/4000 条预测与写后检查；4000 条由训练文本循环、合成 ID 和未见后缀构成，仅检查容量，不评估准确率。

完整模型、提交候选、调试分数、警告、拟合审计、源代码快照与命令日志保存在本地忽略目录 `artifacts/s05_full_baseline_20261118_01/`。原始训练时源码在 `source/`；训练后空输入修复与容量测试代码在 `validation_source/`，元数据分别记录指纹。完整模型不提交 Git，当前 `experiments.csv` 未增加记录。

复现需使用记录的 Python/依赖环境、可信的项目源代码与官方数据，执行 README 和工作日志 L007 的普通 Python 命令。运行目录与输出必须使用新路径。序列化 joblib 只加载自己生成且核对过指纹的可信文件，跨 Python/sklearn/numpy 版本兼容性不保证。S06 尚未实施；Jupyter 通道、已跟踪 CSV、Rules 核实与 S03 延后任务继续保留。
