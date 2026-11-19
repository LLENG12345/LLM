# S06-B 基线重构等价性复核

V1.0 功能验收通过。真实运行编号为 s06b_equivalence_cv_20261118_01 和 s06b_equivalence_full_20261118_01，性质为代码重构等价性复核，不是算法优化或新 Kaggle 提交。

- equivalence.json：2699 条 OOF 标签、ID、折号和全部分数与 S04 相同；分类报告/召回/混淆及指标一致；全量预测和分数与 S05 相同，数值检查容差为 0。
- config.json / metrics.json：新五折实际配置和真实指标/耗时。运行来源与新共享配置已记录。
- notebook-validation.json：公共模块生成 Notebook 的实际隔离运行及 S01—S05 回归结果，保留当时执行字节指纹和临时路径。
- final-notebook-identity.json：最终交付所有单元格源码与隔离实际执行版本完全相同；构建时切换到共享轻量记录只影响来源信息，不改变单元格代码。
- submission-sources.json：当前新生成 Notebook 和公共模块/生成器指纹；它尚未上传，不关联旧版 0.311 为自己的成绩。
- summary.json：首次已确认提交与新等价性运行的关系、公共框架验收及保留限制。

首次真实提交为 notebookceeabf3006，scriptVersionId=371057481，公榜 0.311。用户确认上传后未修改，历史本地 SHA adcd730607e42e00721bbafa6b0002696fa7de08bb126c488927a90e61a631db；详细归档仍在原 S06-A kaggle-feedback.json。远端源码字节、环境、团队和提交时间/ID未独立获取，不填虚构字段。

完整模型、OOF、原始日志与源码快照保存在本地忽略目录 artifacts。新模型序列化文件的二进制 SHA 可以不同，验收依据完整参数、严格数值和真实预测等价性，不以序列化字节相同为算法要求。数据、固定分折和旧实验产物 208 个指纹保持不变；新实验不写新公榜值。

执行和重现说明见 docs/S06-B_共用框架.md 及工作日志 L010/L011。S01 已跟踪 CSV、S00 Rules/账户信息、标准 Jupyter 通道以及 S03 延后研究继续保留。
