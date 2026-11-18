# 数据目录

从比赛的 [Data 页面](https://www.kaggle.com/competitions/name-that-llm/data) 获取官方文件，默认布局为 `data/name-that-llm/train.csv`、`test.csv` 和 `sample_submission.parquet`。已有文件保持原位，不通过复制或移动解决路径问题。

基线按显式环境变量 `NAME_THAT_LLM_DATA_DIR`、Kaggle 官方目录、项目根目录的 `data/name-that-llm/`、旧 `data/` 顺序查找。必须同时具备 train.csv 和 test.csv；显式目录无效直接报错，其余候选会报告缺失文件。支持从仓库根目录与 notebooks 目录运行。

`.gitignore` 递归匹配原始 CSV、Parquet 等文件，但 Git 对已跟踪文件不应用忽略规则。当前 train.csv、test.csv 已在提交中，按要求保留跟踪，未执行移除索引操作。数据说明、`experiments/folds/` 和实验摘要继续可共享。

## S02 输入契约

Notebook 的第二个代码单元格维护唯一的 `OFFICIAL_MODEL_LABELS`。`class_order` 为这 18 个字符串按 Python 字典序排序后的不可变 tuple，位置从 0 开始；后续模型分数必须按模型自身 `classes_` 显式核对或对齐到此顺序，不能猜测列顺序。本轮不改模型或实现融合。

`read_input_csv(path, kind, require_all_labels=True)` 统一按 pandas StringDtype 读取，不自动识别 NA/NULL 为缺失，不跳过空白记录。必需字段中的真正缺失值、空字符串、纯空白、非字符串和重复 ID 会报错；错误包含来源、字段及最多五个记录位置。record 是从 1 开始的数据记录序号，不是含多行引用文本的 CSV 物理行号。

训练必需列为 id、prompt_id、text、model；测试只需 id、text。完整训练入口要求标签集合与官方集合完全一致，同时报告缺失和未知标签。小样例可用 `require_all_labels=False` 跳过全类别覆盖检查，但仍拒绝每行非法标签。没有通用的 2,699 或 3 行限制。

校验中的 strip 只用于判空，不覆盖有效内容；读取层不做清洗、排序、去重、截断或 Unicode 规范化。CSV 中无内容的字段按空字符串拒绝，DataFrame 中 None/NaN/pd.NA 按真正缺失值拒绝；字面 NA、NULL、NaN 等文本保留。

`load_competition_data` 返回训练、测试、可选的样例提交和摘要。摘要在内存中产生；执行以下命令可将官方数据检查摘要保存到被忽略的 `artifacts/s02/input-summary.json`：

```powershell
.\.venv\Scripts\python.exe scripts/check_s02.py
```

摘要含原始字节 SHA-256、路径、大小、行数、列、类型、类别分布和 class_order。样例提交若存在则检查 id/model 列、字符串 ID、非空与唯一、ID 集合与测试一致；不要求样例行顺序相同，也不将 model 占位值当作正式预测进行评分或标签合法性验收。当前样例 3 行，id/model 均为字符串，model 全为 gpt-5.5。

## S05 正式预测与本地样例检查的区别

`scripts/run_s05.py` 复用相同路径解析、字符串读取、测试输入校验及唯一 class_order，分别加载 train/test。运行时提交以实际 test.csv 的 ID、数量及顺序为准，不调用要求旧样例 ID 对应的 `load_competition_data`；因此 Kaggle 替换测试数据后不会被本地 sample_submission 阻断。样例占位值不进入模型预测。

输出严格为字符串 `id, model`，不含索引、分数或 prompt；写后独立回读并核对全部字段、ID 和标签。空输入约定产生两列字符串的 0 行文件，非空数据仍严格拒绝缺失和空白字段。0/1/3/17/257/4000 条独立输入已通过，4000 条仅用于容量检查。

原始文件、固定分折和 S04 资源受保护，输出不能位于这些目录，已有文件不覆盖。实际生成文件位于 `artifacts/s05_full_baseline_20261118_01/submission.parquet`，SHA-256 为 `12209416456505034751a82b190ae037e1b026164a149e666b2990ee1ec005fd`。本轮未修改、移动、删除或取消跟踪原始数据，未作 Kaggle 提交。

## 已跟踪数据的只读复核与后续建议

两份 CSV 首次加入提交均为 `77bb475`；S02 检查时原始文件没有工作区或暂存改动。若以后经确认取消跟踪，可在项目根目录使用以下命令；本轮未执行：

```powershell
git rm --cached -- data/name-that-llm/train.csv data/name-that-llm/test.csv
```

该命令保留执行者本地文件，在索引中暂存删除记录；提交并推送前不改变远端。之后提交/推送会使新版本仓库不再包含这两份文件，但不会移除历史提交中的副本。其他成员拉取该删除提交时，干净的已跟踪副本可能从工作目录移除；有本地修改时可能阻止拉取或产生冲突，因此应先协调并在仓库外保留副本，或准备重新获取官方数据。无需重写 Git 历史；本轮只记录建议，不进行上述索引、提交或远端操作。

