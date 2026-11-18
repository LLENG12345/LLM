# Name That LLM 数据挖掘课程项目

华南理工大学《数据挖掘》课程报告的基础仓库。当前提供一个**可独立运行的基线 Notebook**，用于验证建模、生成 Kaggle 提交文件和记录后续改进。

## 比赛开发文档

- [工作计划](docs/工作计划.md)：按代码构建依赖列出任务、接口、产物和验收条件，不安排时间。
- [工作日志](docs/工作日志.md)：记录实际修改、验证、实验决策和阶段结果。

后续开发先阅读当前计划和最近日志，每完成一项可独立说明的功能修改或阶段结果就追加日志，验收通过后再更新计划状态。模型实验数值摘要继续保存在 [实验记录](experiments/experiments.csv)。

## 先跑通一次有效提交

1. 在 Kaggle 比赛页确认已加入**同一支队伍**，并保存 Team Page 链接。[比赛页面](https://www.kaggle.com/competitions/name-that-llm)
2. 打开 [`notebooks/01_baseline.ipynb`](notebooks/01_baseline.ipynb)，将其上传或复制到 Kaggle Notebook；把比赛数据添加为输入。
3. 在 Kaggle Notebook 中运行全部单元格。Notebook 会按 `prompt_id` 分组做五折验证，然后在完整训练集训练，输出 `/kaggle/working/submission.parquet`。
4. 提交这个 Notebook 的已保存版本。确认 Kaggle 显示提交**有效**且有公榜分数，然后把版本和分数填入 [`experiments/experiments.csv`](experiments/experiments.csv)。
5. 后续每次只改一项主要设置，记录本地验证和公榜结果；保留至少五次有效提交的历史截图。

Kaggle 是 Code Competition：正式提交需由 Notebook 生成 `submission.parquet`；互动运行时看到的测试数据只是示例行，正式评分会重新运行 Notebook。比赛要求关闭互联网，CPU/GPU Notebook 运行时间不超过 9 小时。[官方提交说明](https://www.kaggle.com/competitions/name-that-llm/overview/abstract)

## 本地运行

需要 Python 3.10 或更新版本。将比赛下载的 `train.csv` 和 `test.csv` 放进 `data/`，然后：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook notebooks/01_baseline.ipynb
```

本地运行也会在当前工作目录生成 `submission.parquet`。Notebook 会优先寻找 Kaggle 比赛输入目录，再寻找仓库的 `data/`。训练数据、提交文件及证据截图默认不纳入 Git。

## 仓库结构

```text
notebooks/01_baseline.ipynb    自包含的验证、训练与提交 Notebook
data/                          本地数据位置（数据文件不提交）
experiments/experiments.csv     每次实验与 Kaggle 提交记录
evidence/                       截图位置（图片不提交）
docs/REPORT_CHECKLIST.md        报告、成绩汇总和时间节点
```

## 建模和课程报告提醒

- 比赛是 18 分类，评分采用 **Balanced Accuracy**，即各类别召回率的平均值。训练集与测试集的 prompt 不重叠，因此基线使用 `prompt_id` 分组验证。[官方数据](https://www.kaggle.com/competitions/name-that-llm/data)、[官方评分](https://www.kaggle.com/competitions/name-that-llm/overview/abstract)
- 课程评分按赛题理解及结果 30 分、组员及分工 30 分、算法说明 20 分、核心代码 20 分。具体提交材料见 [报告清单](docs/REPORT_CHECKLIST.md)。
- 这个基线只用于建立可靠起点；需要根据验证、各类别召回率与实际公榜结果继续实验，不能把它的分数视为最终成绩。

