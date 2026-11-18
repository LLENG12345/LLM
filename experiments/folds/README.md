# S03 固定分折资源

四个 CSV/JSON 文件为全组共用轻量资源，不含原始文本。CSV 固定列为 id、prompt_id、fold；ID 为字符串，fold 为 1 至 5。CSV 按字符串 ID 排序，仅用于存储，加载后按训练数据当前 ID 顺序对齐。

- folds_v1：GroupKFold(n_splits=5, shuffle=False, random_state=None)，复现原始基线确定性划分，用于日常比较。
- folds_v2：GroupKFold(n_splits=5, shuffle=True, random_state=20261118)，使用当前 scikit-learn 1.9.1，预先固定供后续候选稳定性复核，没有根据模型成绩挑选；并非新的独立测试集。

元数据包含版本、训练原文件与分折文件 SHA-256、算法参数、生成环境、样本和 prompt 数、各折训练/验证规模与类别分布、class_order、生成方式及用途。

在仓库根目录验证：

```powershell
.\.venv\Scripts\python.exe scripts/check_s03.py
```

`--generate` 只供首次创建；任一目标文件已经存在就明确报错，不覆盖已有资源。数据或分折确需变化时使用新版本设计，不删除现有资源来绕过检查。本轮已创建 v1/v2，正常使用不再执行生成命令。

基线 CV 在模型拟合前按 ID 加载 v1 并检查原文件指纹、映射字节指纹、完整性、prompt 一致、合法折号、组隔离和各折类别覆盖。内存重排行只改变对齐结果顺序，不改变磁盘数据指纹；实际 CSV 改动应触发版本错误。

`.gitattributes` 对这组资源固定 LF，避免 Windows/Linux 检出换行差异破坏分折文件字节指纹。若实际训练文件的字节指纹不同，应先确认数据来源与版本，而不是更新元数据掩盖差异。

Notebook 当前为本地研究入口，CV 需要这组文件；后续 S06 的 Kaggle 离线版本必须同时包含对应资源，不能静默重新生成另一套分折。
