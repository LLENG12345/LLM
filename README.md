# Name That LLM 数据挖掘课程项目

华南理工大学《数据挖掘》课程报告的基础仓库。当前提供一个**可独立运行的基线 Notebook**，用于验证建模、生成 Kaggle 提交文件和记录后续改进。

## 比赛开发文档

- [工作计划](docs/工作计划.md)：按代码构建依赖列出任务、接口、产物和验收条件，不安排时间。
- [工作日志](docs/工作日志.md)：记录实际修改、验证、实验决策和阶段结果。

后续开发先阅读当前计划和最近日志，每完成一项可独立说明的功能修改或阶段结果就追加日志，验收通过后再更新计划状态。模型实验数值摘要继续保存在 [实验记录](experiments/experiments.csv)。

## B4 字符形态增量

[字符形态报告](docs/S07-B_B4_字符形态实验报告.md)：B4=Char+Morphology，完整BA **0.352262**，对主对照B2 **+3.7783个百分点**，五折全增、固定稀疏9胜1负。仍低于B3/B3-C；保留形态独立增量与互补性证据，默认B0不变。

本轮按用户指令简化后续路线：B4不含Word/Style，B5固定Char+Style+Morphology对B3-C，不再训练含Word旧B5、不另设B5-C；B6/B7不变。“Word未证明有用”不等于已证明有害。若后续提交不佳，只改配置branches即可恢复Word，无需重新开发。

当前检查：`.\.venv\Scripts\python.exe scripts/check_b4.py --formal`。新增1次冒烟+5次正式；B0—B3-C不重训、Style源码未改、无新Notebook或提交。B5尚未实施。

## B3-C 字符与风格消融

[组合消融报告](docs/S07-B_B3-C_组合消融实验报告.md)：Char+Style完整BA **0.377457**，比B3只高0.00124个百分点，正确数相同。十组稀疏4胜6负、prompt区间跨零，因此保留B3和B3-C，默认仍B0。新配置 `configs/s07b_b3c_char_style.json` 仅关闭Word，未修改模型/Style源码，新增1次冒烟+5次正式。

B3-C阶段完整检查见历史报告；当前活动源因B4授权新增形态分支，采用check_b4的历史保护及8项B3-C相关回归。B3相关回归入口仍有效。B4当前主对照已按新指令改为B2，B5固定Char+Style+Morphology。没有新Notebook或Kaggle提交。

## B3 风格增量结果

[S07-B/B3报告](docs/S07-B_B3_风格增强实验报告.md)：Word+Char+162维风格，完整OOF BA **0.377445**，对B0 +7.0007个百分点；固定十组稀疏均值0.380641。仅新增1次冒烟+5次正式折内拟合，B0—B2复用。保留B3为研究候选，默认仍B0，尚无B3公榜成绩。

配置为 `configs/s07b_b3_rich_style.json`，继续用 `scripts/run_experiment.py`。完整OOF/源快照在artifacts独立运行目录，轻量对照在 `experiments/s07b_b3_rich_style_v1_20261119_01/`。描述分析不用于删特征；B3阶段未实施Char+Style/B4；后续B3-C补充结果见上节。当前源工厂新增分支，旧99仍保留已验收B0字节；未重建Notebook，生成器来源检查不表示旧文件已同步。未来B3自包含支持要求见报告。

## 先跑通一次有效提交

当前计划已升级为 [v2.0](docs/工作计划.md)，[v1.1 原文](docs/history/工作计划_v1.1_20261119.md) 保留。S07-A 使用已有三组 OOF 完成零新增训练评价：完整 OOF 为主、冻结十组 k=2 为补充，B3风格增量已完成，下一轮等待授权；不重复训练 B0—B2。

- [评价协议](docs/实验评价协议.md)：冻结种子、固定18类口径、缺类处理、选型与泄漏边界。
- [实测对照](experiments/s07a_protocol_20261119_01/report.md)：完整/稀疏指标、类别覆盖、配对排序与限制。
- [来源核实](docs/S07-A_研究资料核实.md)：队友03固定源码、04缺失、社区联合分配与本项目抽样的区别、Rules及外部资源未知项。

当前工作区复核命令：`.\.venv\Scripts\python.exe scripts/check_b3.py --formal --regressions`（不拟合分类器）。重算使用 `scripts/run_s07a.py --output artifacts/s07a_review_new`，目录必须尚不存在；入口禁止模型/TF-IDF拟合。资源在 `experiments/evaluation_masks/sparse_k2_v1/`，不会改 folds_v1/v2 或 experiments.csv。仅有一个确认公榜分数，不能由本次结果验证社区相关性主张。

之前另生成的 `notebooks/99_submission_char_only.ipynb` 已通过本地运行，尚未提交；其来源清单所指独立生成脚本当前缺失，重建性待修复（L014）。默认提交 Notebook 仍为下述原组合。

首次提交 [notebookceeabf3006 / 371057481](https://www.kaggle.com/code/abcde1232121/notebookceeabf3006?scriptVersionId=371057481) 评分 **0.311**，用户确认上传后未改代码。S06-B 公共框架及等价性复核已完成，**V1.0 达到**；当前 99_submission 是重构后重新生成的版本，本轮未再次提交。归档与未知远端信息见 [提交反馈](experiments/s06a_notebook_20261118_01/kaggle-feedback.json)，操作步骤见 [指南](docs/S06-A_Kaggle操作指南.md)。

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
.\.venv\Scripts\python.exe scripts/install_local.py
.\.venv\Scripts\python.exe scripts/check_s01.py
.\.venv\Scripts\python.exe -m notebook notebooks/01_baseline.ipynb
```

`requirements.txt` 保留基础依赖范围，`requirements-s01.txt` 固定本轮验证的直接依赖版本。已有环境先检查版本，仅补缺失或不满足要求的依赖，不使用 `--upgrade`。完整环境快照见 `docs/environments/s01-pip-freeze.txt`。

S06-B 已复用原 .venv。`install_local.py` 只在该虚拟环境写入指向当前 src 的 .pth 路径，不下载依赖；注册后启动新 Python/Jupyter 进程。当前环境没有 setuptools，因此本轮未用 pip editable 构建；pyproject.toml 已提供最小包元数据。joblib/psutil 为原 S04/S05 已使用的库，本轮仅补直接依赖声明及既有版本固定。

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

研究 Notebook 的 CV 和预测现写入 artifacts 下唯一运行目录，Kaggle 输出到 `/kaggle/working/`。S01 检查包含导入、路径、I/O 和小型合成语料保护，不拟合官方数据。已跟踪的两份 CSV 仍保留版本管理，本轮未取消跟踪。

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
.\.venv\Scripts\python.exe scripts/check_s06a.py --regressions --checkpoint artifacts/s06b/before.json --output-dir artifacts/s06b/notebook-validation-new
```

检查目录必须尚不存在。生成器从 src/ntl 的唯一标签、数据、特征/模型配置及提交逻辑提取必要定义，运行时完全嵌入，源码指纹见 `notebooks/99_submission.sources.json`。生成结果不读取本地 scripts、src、Git、分折、模型或历史 artifacts，不下载、不安装依赖。S06-B 后使用对应历史保护 checkpoint，不能用 S06-A 的旧源码字节检查去禁止已授权的模块迁移。

本地验收使用干净临时目录和独立 Python 进程，移除公共包路径并阻断项目资源访问、网络操作和子进程执行。此验证不是 Jupyter 内核或 Kaggle 容器验收。首次 Kaggle 旧版本已归档，新生成文件本轮未上传。

## S06-B 公共实验入口

在完成本地注册后，使用相同命令运行默认基线；每次使用新的 run-id：

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --mode smoke --run-id baseline_smoke_new
.\.venv\Scripts\python.exe scripts/run_experiment.py --mode formal --run-id baseline_cv_new --smoke-run baseline_smoke_new
.\.venv\Scripts\python.exe scripts/test_s06b.py
```

支持 `--data-dir`、`--config <JSON覆盖文件>`、`--purpose`。默认配置唯一维护于公共模型/特征模块，覆盖设置被复制，不修改默认值。后续消融可用 `{"branches":["word"]}` 或 `{"branches":["char"]}`；本轮仅做合成小样例配置测试，没有运行正式分支消融。兼容 run_s04/run_s05/check_s04/check_s05 入口，详细职责、实际命令和等价性结果见 [公共框架说明](docs/S06-B_共用框架.md)。

## S07-01 Word/Char 消融

本轮已完成两个新增真实五折；默认基线和正式提交 Notebook 保持不变。组合 OOF BA=0.307437，Word-only=0.262612，Char-only=0.314479。Char 高于组合约 0.7042 个百分点，但当前 prompt 配对区间包含 0，暂作为研究候选，不自动替换或提交。完整结果、18 类错误分析、互补性及成本见 [消融报告](docs/S07-01_消融实验报告.md)。

配置仅包含分支选择，位于 configs/s07_word_only.json 和 configs/s07_char_only.json。复跑使用新 run-id，先 smoke 再 formal，沿用现有 run_experiment.py；分析入口 scripts/analyze_s07.py，检查入口 scripts/check_s07.py。报告和实验产物默认拒绝覆盖；本轮未做 S07-02 或新 Kaggle 提交。

## 仓库结构

```text
notebooks/00_eda.ipynb         导入公共数据/分析模块的研究入口
notebooks/01_baseline.ipynb    导入公共模块的 CV/全量训练展示入口
notebooks/99_submission.ipynb  生成的独立 CPU 提交 Notebook
src/ntl/                       数据/标签/分折/模型/实验/提交公共包
scripts/run_experiment.py       固定 v1、配置驱动的通用实验入口
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

