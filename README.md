# Name That LLM 数据挖掘课程项目

华南理工大学《数据挖掘》课程报告的基础仓库。当前提供一个**可独立运行的基线 Notebook**，用于验证建模、生成 Kaggle 提交文件和记录后续改进。

## 比赛开发文档

- [工作计划](docs/工作计划.md)：按代码构建依赖列出任务、接口、产物和验收条件，不安排时间。
- [工作日志](docs/工作日志.md)：记录实际修改、验证、实验决策和阶段结果。

后续开发先阅读当前计划和最近日志，每完成一项可独立说明的功能修改或阶段结果就追加日志，验收通过后再更新计划状态。模型实验数值摘要继续保存在 [实验记录](experiments/experiments.csv)。

## 先跑通一次有效提交

S06-A 已生成独立提交 Notebook；用户已反馈首次提交完成、评分 **0.311**，Notebook 标识 `notebookceeabf3006`。保存版本链接及源码关联待补，详见 [提交反馈](experiments/s06a_notebook_20261118_01/kaggle-feedback.json)。具体规则核实状态及操作步骤见 [S06-A 操作指南](docs/S06-A_Kaggle操作指南.md)。

1. 在 Kaggle 比赛页确认已加入**同一支队伍**，并保存 Team Page 链接。[比赛页面](https://www.kaggle.com/competitions/name-that-llm)
2. 将 [`notebooks/99_submission.ipynb`](notebooks/99_submission.ipynb) 导入 Kaggle Notebook；关联本比赛官方数据，选择 CPU、Internet Off。
3. 保存并完整运行一个新版本。提交 Notebook 仅全量训练、运行时预测及写后检查，输出 `/kaggle/working/submission.parquet`；五折/EDA 保留在研究入口。
4. 提交这个 Notebook 的已保存版本。确认 Kaggle 显示提交**有效**且有公榜分数，然后把版本和分数填入 [`experiments/experiments.csv`](experiments/experiments.csv)。
5. 后续每次只改一项主要设置，记录本地验证和公榜结果；保留至少五次有效提交的历史截图。

Kaggle 是 Code Competition：正式提交需由 Notebook 生成 `submission.parquet`；互动运行时看到的测试数据只是示例行，正式评分会重新运行 Notebook。比赛要求关闭互联网，CPU/GPU Notebook 运行时间不超过 9 小时。[官方提交说明](https://www.kaggle.com/competitions/name-that-llm/overview/abstract)

## 本地运行

当前 S01 验证环境为 Python 3.12。官方文件保留在 `data/name-that-llm/`，无需移动或复制。以下 PowerShell 命令从仓库根目录执行；直接调用虚拟环境解释器，无需修改 PowerShell 执行策略：

```powershell
python -m venv .venv  # 仅在没有合适的项目环境时执行
.\.venv\Scripts\python.exe -m pip install -r requirements-s01.txt
.\.venv\Scripts\python.exe scripts/check_s01.py
.\.venv\Scripts\python.exe -m notebook notebooks/01_baseline.ipynb
```

`requirements.txt` 保留基础依赖范围，`requirements-s01.txt` 固定本轮验证的直接依赖版本。已有环境先检查版本，仅补缺失或不满足要求的依赖，不使用 `--upgrade`。完整环境快照见 `docs/environments/s01-pip-freeze.txt`。

也可先运行 `.\.venv\Scripts\Activate.ps1` 再使用 `python`；若策略禁止激活，继续使用上面的完整解释器路径。从 notebooks 目录启动使用：

```powershell
Set-Location notebooks
..\.venv\Scripts\python.exe -m notebook 01_baseline.ipynb
```

Notebook 输入优先级：环境变量 `NAME_THAT_LLM_DATA_DIR` → Kaggle 官方目录 → 项目根目录下 `data/name-that-llm/` → 旧 `data/`。每个候选要求同时存在 `train.csv` 和 `test.csv`。显式目录无效会报错，不切换数据版本；相对显式路径按内核当前目录解释。可在启动 Jupyter 前设置：

```powershell
$env:NAME_THAT_LLM_DATA_DIR = 'D:/你的数据目录'
# 随后启动 Jupyter；使用默认路径前移除此覆盖：
Remove-Item Env:NAME_THAT_LLM_DATA_DIR -ErrorAction SilentlyContinue
```

本地完整运行仍在内核当前目录输出结果，Kaggle 输出到 `/kaggle/working/`。S01 冒烟脚本只做导入、路径和 I/O 检查，不执行训练。新原始数据与生成产物由忽略规则保护；当前提交中已跟踪的两份 CSV 仍保留版本管理，本轮没有取消跟踪。

## S05 本地全量训练与提交候选

使用已验证的 S04 模型配置和现有 `.venv`，在仓库根目录执行（运行目录与提交文件必须尚不存在）：

```powershell
.\.venv\Scripts\python.exe scripts/run_s05.py --run-id s05_full_baseline_20261118_02 --data-dir data/name-that-llm --output-dir artifacts/s05_full_baseline_20261118_02 --submission-path artifacts/s05_full_baseline_20261118_02/submission.parquet
.\.venv\Scripts\python.exe scripts/check_s05.py artifacts/s05_full_baseline_20261118_02
.\.venv\Scripts\python.exe scripts/test_s05.py artifacts/s05_full_baseline_20261118_02
```

以上 `_02` 是下一次复跑的示例编号；本轮实际执行并验收的是 `_01`，完整命令与结果见工作日志 L007。入口支持显式数据/输出目录和 `--batch-size`（默认 256），省略数据目录时沿用 S01 优先级。运行时只用 test.csv 的 id/text，不要求隐藏测试 ID 对应旧 sample_submission；S02 的样例对应检查仅用于本地输入研究。

运行目录保存完整 `model.joblib`、`submission.parquet`、单独的 decision scores、配置、元数据、警告及源码快照。自动执行 Parquet 回读和模型重载一致性检查，已有输出拒绝覆盖。只加载本项目自己生成且核对过指纹的可信 joblib；重新加载应使用记录的 Python/sklearn/numpy 等兼容版本。

S05 采用普通 Python 执行，S06-A 已建设独立提交入口；Jupyter 启动通道限制继续保留。本地文件合法不代表 Kaggle 提交成功。

## S06-A 自包含提交 Notebook

```powershell
.\.venv\Scripts\python.exe scripts/build_submission_notebook.py
.\.venv\Scripts\python.exe scripts/build_submission_notebook.py --check
.\.venv\Scripts\python.exe scripts/check_s06a.py --regressions --output-dir artifacts/s06a/validation-new
```

检查目录必须尚不存在。生成器从原 Notebook 提取标签/输入/模型，从 S04 提取完整参数归一化，从 S05 提取拟合/预测/严格提交函数；源码指纹见 `notebooks/99_submission.sources.json`。生成结果直接包含全部代码，运行时不读取本地 scripts、src、Git、分折、模型或历史 artifacts，不下载、不安装依赖。

本地验收使用干净临时目录和独立 Python 进程，并阻断项目资源访问、网络操作和子进程执行。此验证不是 Jupyter 内核或 Kaggle 容器验收。用户已反馈完成首次评分 0.311；远端版本归档待补，S06-B 模块化与 V1.0 尚未完成。

## 仓库结构

```text
notebooks/01_baseline.ipynb    自包含的验证、训练与提交 Notebook
notebooks/99_submission.ipynb  生成的独立 CPU 提交 Notebook
scripts/build_submission_notebook.py  从权威源码生成并检查提交 Notebook
data/                          本地数据位置（数据文件不提交）
experiments/experiments.csv     每次实验与 Kaggle 提交记录
evidence/                       截图位置（图片不提交）
docs/REPORT_CHECKLIST.md        报告、成绩汇总和时间节点
```

## 建模和课程报告提醒

- 比赛是 18 分类，评分采用 **Balanced Accuracy**，即各类别召回率的平均值。训练集与测试集的 prompt 不重叠，因此基线使用 `prompt_id` 分组验证。[官方数据](https://www.kaggle.com/competitions/name-that-llm/data)、[官方评分](https://www.kaggle.com/competitions/name-that-llm/overview/abstract)
- 课程评分按赛题理解及结果 30 分、组员及分工 30 分、算法说明 20 分、核心代码 20 分。具体提交材料见 [报告清单](docs/REPORT_CHECKLIST.md)。
- 这个基线只用于建立可靠起点；需要根据验证、各类别召回率与实际公榜结果继续实验，不能把它的分数视为最终成绩。

