# S06-A 本地提交 Notebook 验收

当前仅完成状态一：本地自包含 Notebook 制作与验收。Kaggle 关闭互联网的保存版本运行、正式有效提交及公榜分数尚未取得，V1.0 未达到；这不是新的独立模型实验，experiments.csv 未增加记录。

以上为本地建设完成时的历史状态。后续用户反馈已提交并取得 0.311，Notebook 标识 notebookceeabf3006，单独记录在 kaggle-feedback.json。原 validation-summary.json 保留当时事实，不追写为本机已验证远端运行；保存版本、提交链接和源码关联待补。本次真实评分另登记到 experiments.csv，未新增本地 OOF 成绩，S06-B 与 V1.0 仍未完成。

- `validation-summary.json`：真实规模、时间、版本、回归状态、完整性及隔离验证摘要。
- `notebook-sources.json`：生成器及权威源码 SHA-256、S04 配置指纹和最终 Notebook 字节指纹。
- `code-identity.json`：最终 Notebook 的所有单元格源码与实际隔离执行版本逐字相同，7 个代码单元格分别有指纹。执行后仅修正 UTF-8/LF 字节序列化及生成器来源信息，未更改单元格代码。

完整证据在本地 `artifacts/s06a/validation-r6/`，包括官方 3 条、替换 17 条测试输入和训练指纹错误三种独立子进程运行、普通 Python 已执行 Notebook、提交文件、实际参数和决策分数、全部回归日志及 159 个历史文件指纹。提交候选保留在 `artifacts/s06a/validation-r6/official/submission.parquet`；原运行日志记录的是当时临时目录，临时目录完成后已自动清理。

隔离使用 Python -I、临时目录和全新命名空间，输入只有 train/test，没有本地模型、分折、项目源码或 Git。Python 审计护栏阻断 DNS/连接等网络操作、项目文件访问和全部子进程执行；安装在解释器环境中的依赖可读取。Windows 的本机主机名读取属于系统本地查询；3 个 cmd/ver 系统版本探测仍被阻断并由依赖自行回退。没有放开子进程、联网或项目读取权限。该机制是针对实际代码的执行检查，不是 OS 沙箱、容器或 Kaggle 环境测试。

失败诊断保留在 artifacts/s06a/validation、validation-r2 至 validation-r5 及相应日志，均未登记为成功；最终本地验收为 validation-r6，最终字节指纹和代码一致性见后续封存文件。

重新生成和复查：在项目根目录执行 `python scripts/build_submission_notebook.py`、`python scripts/build_submission_notebook.py --check`；使用现有项目 .venv，完整本地检查命令见工作日志 L008。检查输出请使用新目录。正式人工流程和待核实的 Rules/账户/团队/配额见 docs/S06-A_Kaggle操作指南.md。
