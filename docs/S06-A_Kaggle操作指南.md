# S06-A：导入、离线运行与人工提交

本轮交付 `notebooks/99_submission.ipynb`，只含完整基线拟合、运行时预测及 Parquet 写后校验。它由权威代码生成，不需要上传 scripts、src、模型、分折或本地仓库。标准 Jupyter、Kaggle 保存版本运行和正式评分是不同状态；本地普通 Python 验收不能替代它们。

S06-B 更新：旧版 371057481 已由用户确认取得 0.311 且上传后未改代码。当前同名文件现由公共模块重新生成，本轮未再次提交，不能将其字节指纹当作旧评分版本。旧来源指纹/提交归档和新等价性记录分别保存；实际远端环境等未知项仍保留。

## 官方信息复核（2026-11-18）

直接打开本比赛 Overview、Data、Rules、Code 及文档动态页面均无可读正文；下列可确认项来自官方 URL 的检索正文，存在索引延迟，不等同于已登录的实时比赛界面。没有读取 Cookie、密码或 API Token，没有访问或操作账户。

| 信息 | 复核结果 | 官方来源 |
| --- | --- | --- |
| 正式提交 | Notebook 代码提交 | [比赛概览](https://www.kaggle.com/competitions/name-that-llm/overview/description) |
| 执行约束 | CPU/GPU 均不超过 9 小时，Internet 关闭 | [比赛概览](https://www.kaggle.com/competitions/name-that-llm/overview/description) |
| 外部资源 | 概览允许公开免费的外部数据和预训练模型；本方案不使用 | [比赛概览](https://www.kaggle.com/competitions/name-that-llm/overview/description) |
| 输入与输出 | 概览列出 `/kaggle/input/competitions/name-that-llm/test.csv`，工作目录输出 `submission.parquet` | [比赛概览](https://www.kaggle.com/competitions/name-that-llm/overview/description) |
| 提交内容 | `id, model`，每个 ID 一次，非空且预测为官方合法标签 | [比赛概览](https://www.kaggle.com/competitions/name-that-llm/overview/description) |
| 字段类型 | ID、文本和模型标签为字符串；测试仅需 id/text | [比赛数据](https://www.kaggle.com/competitions/name-that-llm/data) |
| 隐藏重跑 | 正式评分替换测试数据，不能绑定本地样例数量或 ID | [比赛概览](https://www.kaggle.com/competitions/name-that-llm/overview/description)、[调试说明](https://www.kaggle.com/code-competition-debugging) |
| 保存运行与输出位置 | Save & Run All 在新会话完整运行；输出保存在 /kaggle/working | [Notebook 官方文档](https://www.kaggle.com/docs/notebooks) |
| 发起提交 | 从成功保存版本的 Output 发起代码竞赛提交 | [比赛官方文档](https://www.kaggle.com/docs/competitions) |

概览正文称交互测试为 1 行，Data 正文称 3 行，本地实际为 3 行；保留这个来源差异。实现完全依据运行时 test.csv，不把任一示例数量当作契约。通用 Notebook 文档中的会话时长不能覆盖本比赛的 9 小时提交限制。

### 正式提交前必须人工核实

[本比赛 Rules](https://www.kaggle.com/competitions/name-that-llm/rules) 仍未取得正文，以下事项未确认，不引用其他比赛或通用文档的示例数字代替：

- 最大团队人数、合并条件、合并及入赛截止时间。
- 每日团队提交上限、当前剩余次数、最终可选提交数。
- 参赛及获奖资格、年龄/地区/账户认证要求是否适用于各成员。
- 资源许可、团队外分享、获奖许可证和复现义务等完整细则。
- 当前登录账号、规则是否已接受、是否加入预期团队、成员邀请是否生效。
- 当前团队是否可提交、CPU 资源/权限、数据访问权限和剩余配额。

这些事项不影响本地交付；未经确认和用户授权，不发起正式提交。请以你登录后看到的本比赛页面和按钮状态为准，不推测“通常每天几次”。

## 生成与本地复查

项目根目录 PowerShell：

```powershell
.\.venv\Scripts\python.exe scripts/build_submission_notebook.py
.\.venv\Scripts\python.exe scripts/build_submission_notebook.py --check
.\.venv\Scripts\python.exe scripts/check_s06a.py --regressions --checkpoint artifacts/s06b/before.json --output-dir artifacts/s06b/notebook-validation-new
```

检查输出目录必须尚不存在。Notebook 本身优先使用环境变量 `NAME_THAT_LLM_DATA_DIR`，其次检查概览给出的官方路径；若不存在，则查找 /kaggle/input 下同时有 train/test 的唯一候选。候选不唯一或显式路径无效时明确失败。自动发现不是认定另一条路径是官方标准，后续仍校验训练数据 SHA-256。

本地复制 Notebook 到独立临时目录测试时，可将上述数据变量指向受控输入目录，`NAME_THAT_LLM_OUTPUT_DIR` 指向全新输出目录。测试无需样例提交文件。训练数据使用 S04 的 SHA-256 验证版本；隐藏测试只做输入契约校验和运行时指纹记录，不与本地示例 SHA 比较。

## Kaggle 人工操作步骤

1. 打开[比赛页面](https://www.kaggle.com/competitions/name-that-llm)，确认当前账户及预期团队，阅读 Rules，核实上面的未定项目及剩余次数。保存 Team 页面链接供组内核对。
2. 从比赛 Code 页面创建 Notebook，或在现有 Notebook 编辑器使用导入/上传 Notebook 功能，选择本轮的 `99_submission.ipynb`。入口名称可能随界面变化，需要你在当前菜单确认。
3. 在 Input / Add Input 中关联 **Name That LLM 官方比赛数据**。若提示接受规则或验证账户，请先核实并完成；不要上传本地私有模型或历史 artifacts。
4. 在 Session options / Settings 选择 Python、CPU（Accelerator=None），将 Internet 关闭。不要临时安装或下载包。实际设置必须由界面确认，Notebook 文件不能代替账户环境配置。
5. 检查 Input 中实际显示的 train.csv/test.csv 路径。通常无需覆盖；若挂载不同或候选不唯一，明确设置 `NAME_THAT_LLM_DATA_DIR` 为实际官方目录。Kaggle 保持输出变量未设置，输出应为 `/kaggle/working/submission.parquet`。
6. 可以先交互运行全部单元格检查环境。若完整参数比对失败，保留版本信息和差异键，停止并反馈；不要自动修改模型参数或升级依赖。训练 SHA 不同也需先确认官方版本变化或错误数据挂载。
7. 点击 Save Version，选择 **Save & Run All**，启动干净会话的完整运行；Quick Save 仅保存编辑状态，不作为成功运行依据。[官方说明](https://www.kaggle.com/docs/notebooks)
8. 等待该版本完成，核对日志 `FULL_MODEL_PARAMETERS_MATCH_S04`、`FULL_TRAIN_FINISHED` 和 `SUBMISSION_VALIDATED`，没有错误。检查版本 Output 中实际存在非空的 submission.parquet。运行时可能只有示例行，不能用其行数推断隐藏测试规模。
9. 保存这个已完成版本的 URL/编号及实际环境、运行耗时和警告。交互会话中曾经成功不代表保存版本成功。
10. 在规则、团队和配额均核实后，由你从**成功版本**的 Output / Submit 入口发起本比赛提交；不要把本地 Parquet 当作普通预测上传。若当前界面没有提交入口，先核对数据源、Internet、保存版本状态和账户条件，再反馈具体提示。[官方流程](https://www.kaggle.com/docs/competitions)
11. 在比赛 Submissions 查看本次记录。发起请求、有效提交、评分完成分别核对；只有状态有效并产生真实公榜分数后，才填写实验/日志的提交字段。
12. 确认记录归属预期团队，并将提交页面/ID、Notebook 版本、状态、公榜分数或准确错误阶段反馈给项目。此后再验收 S06-A 的第二、第三状态。

输出拒绝覆盖已有同名文件。保存版本使用干净会话；重复交互运行若遇到已有输出，只在确认后清理自己生成的候选文件或重启新会话，保留正式历史产物。

## 回传信息和当前边界

请回传：团队与规则确认结果、Notebook 保存版本链接、实际环境版本、运行状态/耗时、Output 文件、提交 ID/状态及真实成绩。若失败，保留完整错误和发生单元格。

本轮不消耗 Kaggle 提交次数，不执行 Git 提交或取消跟踪。S06-A 本地制作、Kaggle 离线保存版本运行、有效评分依次验收；S06-B 共用框架另行授权，V1.0 还需满足有效提交和必要共用框架条件。
