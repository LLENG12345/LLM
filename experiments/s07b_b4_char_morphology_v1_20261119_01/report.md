# S07-B/B4 字符形态增量实验

2026-11-19；正式 `s07b_b4_char_morphology_v1_20261119_01`，冒烟 `s07b_b4_char_morphology_v1_smoke_20261119_01`。B4=Char+Morphology，主对照B2；B3-C仅为次要候选对照。

## 计划简化与历史保留

按本轮明确指令替代此前B4=B0+Morphology：B4从B2只增Morphology；B5从B3-C只增Morphology，固定Char+Style+Morphology，不实施含Word旧B5、不另设B5-C；B6/B7编号和定义不变，本轮仅训练B4。

依据：B1最低；无Style时Word增量−.704201pp，有Style时约−.001243pp且B3/B3-C区间跨0。只能说**Word在现有实验中未证明有用**，不能说已经证明Word有害。简化是当前研究路线而非永久禁用。B0—B3-C所有配置、OOF及18维分数保留，B3供S10，正式默认B0不变。

## 固定源码与准确规则

参考[03固定源码](https://github.com/051019zhao/LLM/blob/e83b9e3eacbda00621034384b72f6cdea6555060/notebooks/03_optimized_submission.ipynb)的shape、make_tfidf(shape_only=True)、prepare中的.65权重；Notebook SHA为5cce55a25644ca7652f4810d8d7bf9df41890f3a69fd78e88df00977cd6dadcd。本项目独立按方法实现，不分发参考函数源码。

参考先用Unicode正则\d将十进制Nd变0，再将[^\W\d_]类按isupper映A/否则a。独立实现按Unicode isdecimal/isalnum逐字符判断：ASCII/Unicode有大小写字母按实际isupper处理；中文、日文等无大小写字母→a；²等非十进制数值→a、罗马大写Ⅷ→A，不能误用更宽的isdigit把它们全变0。下划线、组合符号、空白、标点/引号/dash/Markdown、Emoji保留，不合并连续重复字符，不做Unicode规范化。

13组边界（含长文本）、2699条官方字符串、当前解释器全部1,112,064个Unicode标量值与参考完全一致，差异0，2699条长度变化0；合成数据的形态TF-IDF训练/转换矩阵也逐值一致。未使用2699条文本提前fit形态TF-IDF。非字符串/二维输入显式拒绝、空批次支持、fit仅记审计身份为参考之外的接口扩展。

典型自造输入：

```text
AbC 123 → AaA 000
²Ⅷ 中文 日本語 😀 → aA aa aaa 😀
# Title → # Aaaaa
```

完整含换行/制表/Markdown/URL/邮箱/缩写/Emoji的输入输出见[参考fixture](../s07b_b4_definition_v1/reference_fixture.json)。

## lowercase陷阱与空白处理

参考实际lowercase=False，实际词表Aa与aa有不同索引，A/a都保留；没有擅自改参考参数。转换层保持所有空白，但sklearn内置字符analyzer会把连续≥2空白折叠成一个空格；所以不能把原文结构保留与最终字符ngrams的行为混为一谈。句长/词长/空白和符号仍可能受题材影响，不能声称完全去主题。

## 实际配置及隔离

Char完全继承B2：char(2,5)、min_df2/max_df1、80k、float32、lowercase=True、sublinear_tf、l2、IDF/smooth_IDF，权重1。Morphology：独立转换→char(2,6)、min_df2/max_df1、max20k、float32、lowercase=False、sublinear_tf、l2、IDF/smooth_IDF，权重.65。LinearSVC继承B2全部实际参数：C1/balanced/max_iter3000/seed20261118/tol1e-4/l2/squared_hinge/ovr/dual=auto等。

Word不构建或拟合、Style不启用。两分支接收相同原文，Char原文不被形态结果覆盖。每折新Pipeline、新词表/IDF，converter只作逐文本确定性处理；实际训练文本摘要匹配折内训练ID，验证后词表/IDF摘要不变。所有统计/分折及标签不进预测文本；v2/测试标签不参与。默认及B2/B3/B3-C关闭形态时get_params完全不变，Style实现未修改。

## 命令与真实拟合次数

```powershell
.\.venv\Scripts\python.exe artifacts/s07b_b4_20261119_01/reference_check.py
.\.venv\Scripts\python.exe scripts/check_b4.py
.\.venv\Scripts\python.exe scripts/run_experiment.py --mode smoke --run-id s07b_b4_char_morphology_v1_smoke_20261119_01 --config configs/s07b_b4_char_morphology.json --purpose char_morphology_increment --data-dir data/name-that-llm
.\.venv\Scripts\python.exe scripts/run_experiment.py --mode formal --run-id s07b_b4_char_morphology_v1_20261119_01 --smoke-run s07b_b4_char_morphology_v1_smoke_20261119_01 --config configs/s07b_b4_char_morphology.json --purpose char_morphology_increment --data-dir data/name-that-llm
.\.venv\Scripts\python.exe scripts/analyze_b4.py
.\.venv\Scripts\python.exe scripts/check_b4.py --formal
.\.venv\Scripts\python.exe scripts/check_b3.py --formal --regressions
.\.venv\Scripts\python.exe artifacts/s07b_b4_20261119_01/check_b3c_related.py
git diff --check
```

官方Pipeline实际新增6次fit：1次冒烟、5次正式，失败/重试0；B0—B3-C不重训。冒烟BA=0.340740741、fit=13.193885s、predict+decision=3.427247s，18类/18维、所有分数有限，通过后再正式。参考/单测只fit必要合成特征转换器，无额外官方向量器/分类器拟合；测试/分析阻断LinearSVC.fit。

## 正式完整OOF

| model | balanced_accuracy | accuracy | macro_f1 |
| --- | --- | --- | --- |
| B0 | 0.307437 | 0.307521 | 0.295638 |
| B1 | 0.262612 | 0.262690 | 0.248365 |
| B2 | 0.314479 | 0.314561 | 0.304080 |
| B3 | 0.377445 | 0.377547 | 0.364327 |
| B3-C | 0.377457 | 0.377547 | 0.364117 |
| B4 | 0.352262 | 0.352353 | 0.342116 |

B4汇总2699条BA=0.352261993537、Accuracy=0.352352723231、MacroF1=0.342115867645；五折BA均值=0.352273307791、总体std(ddof0)=0.015324904195，告警0。

| fold | char | b4 | b4_minus_b2 | b3c | b4_minus_b3c |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.305556 | 0.340741 | 0.035185 | 0.372222 | -0.031481 |
| 2 | 0.287037 | 0.329630 | 0.042593 | 0.372222 | -0.042593 |
| 3 | 0.318519 | 0.355556 | 0.037037 | 0.387037 | -0.031481 |
| 4 | 0.325926 | 0.372222 | 0.046296 | 0.392593 | -0.020370 |
| 5 | 0.335504 | 0.363218 | 0.027714 | 0.363218 | -0.000000 |

主对照B2五折全提升；次要B3-C为4折较低、末折数值平局（1e-12容差，原始差保留）。B4−B0不是本轮增量指标。

## 主要增量与次要候选比较

| comparison | delta | delta_pp |
| --- | --- | --- |
| balanced_accuracy_b4_minus_char | 0.037783 | 3.778275 |
| accuracy_b4_minus_char | 0.037792 | 3.779177 |
| macro_f1_b4_minus_char | 0.038036 | 3.803584 |
| balanced_accuracy_b4_minus_b3c | -0.025195 | -2.519513 |
| accuracy_b4_minus_b3c | -0.025195 | -2.519452 |
| macro_f1_b4_minus_b3c | -0.022001 | -2.200079 |

B4−B2完整BA增量+.037782749（+3.778275pp）；Accuracy+3.779177pp、MacroF1+3.803584pp，支持当前配置下形态具有独立正增量。B4−B3-C为−.025195128（−2.519513pp），这是不同增强类型的候选比较，不能把差值全归因于Morphology。B3仍保留作未来融合输入。

## 固定稀疏评价

| model | balanced_accuracy_mean | balanced_accuracy_std_ddof0 | balanced_accuracy_min | balanced_accuracy_max |
| --- | --- | --- | --- | --- |
| combo | 0.307500 | 0.028259 | 0.264930 | 0.353169 |
| word | 0.256880 | 0.027313 | 0.196096 | 0.295105 |
| char | 0.323946 | 0.030941 | 0.277591 | 0.362450 |
| b3 | 0.380641 | 0.026832 | 0.343032 | 0.426031 |
| b3c | 0.376696 | 0.025974 | 0.343843 | 0.435332 |
| b4 | 0.355695 | 0.026967 | 0.324532 | 0.417541 |

| reference | mean | std_ddof0 | min | max | wins | ties | losses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| char | 0.031749 | 0.027851 | -0.030133 | 0.066825 | 9 | 0 | 1 |
| b3c | -0.021002 | 0.023794 | -0.055705 | 0.016400 | 2 | 0 | 8 |

原10seed、每组300ID/18类、mask SHA c1c28dcfd25f8e22479a73a851bb7ed1b73be7e13d112a0d6498dff2bd6fe19f保持。旧50条模型×seed成绩与原记录误差≤1e-15。B4对B2为9胜1负、对B3-C为2胜8负；只算汇总18类BA，不平均缺类单折，不把重叠掩码视为独立实验。

## 错误、弱类和混淆

相对B2：共同正确689、共同错误1588、修正262、新错160、净增102（849→951正确），预测一致61.652464%。同厂商错911→901，跨厂商939→847。

| label | char | b4 | b4_minus_b2_pp | b3c | b4_minus_b3c_pp |
| --- | --- | --- | --- | --- | --- |
| claude-fable-5 | 0.093960 | 0.107383 | 1.342282 | 0.134228 | -2.684564 |
| claude-opus-4-6 | 0.406667 | 0.446667 | 4.000000 | 0.533333 | -8.666667 |
| claude-opus-4-8 | 0.246667 | 0.260000 | 1.333333 | 0.246667 | 1.333333 |
| claude-opus-5 | 0.620000 | 0.553333 | -6.666667 | 0.660000 | -10.666667 |
| claude-sonnet-5 | 0.233333 | 0.380000 | 14.666667 | 0.333333 | 4.666667 |
| gemini-3.1-pro-preview | 0.726667 | 0.820000 | 9.333333 | 0.800000 | 2.000000 |
| gemini-3.6-flash | 0.386667 | 0.373333 | -1.333333 | 0.380000 | -0.666667 |
| gemini-3.7-flash | 0.393333 | 0.386667 | -0.666667 | 0.473333 | -8.666667 |
| gemini-3.8-flash | 0.326667 | 0.393333 | 6.666667 | 0.400000 | -0.666667 |
| glm-5.3 | 0.140000 | 0.146667 | 0.666667 | 0.186667 | -4.000000 |
| glm-5.3-flash | 0.173333 | 0.233333 | 6.000000 | 0.166667 | 6.666667 |
| gpt-5.5 | 0.266667 | 0.273333 | 0.666667 | 0.373333 | -10.000000 |
| gpt-5.6-luna | 0.266667 | 0.273333 | 0.666667 | 0.353333 | -8.000000 |
| gpt-5.6-sol | 0.146667 | 0.160000 | 1.333333 | 0.193333 | -3.333333 |
| gpt-5.6-terra | 0.253333 | 0.240000 | -1.333333 | 0.246667 | -0.666667 |
| gpt-6-astra | 0.466667 | 0.580000 | 11.333333 | 0.626667 | -4.666667 |
| kimi-k3 | 0.213333 | 0.273333 | 6.000000 | 0.240000 | 3.333333 |
| qwen3.8-max | 0.300000 | 0.440000 | 14.000000 | 0.446667 | -0.666667 |

14类相对B2上升、4类下降。Fable14→16/149、Sol22→24、GLM21→22、GLM Flash26→35、Opus4.8 37→39、Kimi32→41；Gemini3.6 58→56、3.7 59→58下降，3.8 49→59上升；Opus5 93→83下降明显。

Gemini3.6→3.7错37→31，但3.6→3.1错29→36；3.7→3.8错43→46、反向42→42；Terra→Luna30→19，Sol→Luna29→25；GLM→GLM Flash27→32恶化。保留全部306个主对照错误方向及次要对照，不只列改善。

## Bootstrap及互补性

预先固定150prompt配对Bootstrap5000、seed20261118。B4−B2条件95%区间[0.023711098683,0.051494308964] BA；B4−B3-C区间[-0.042983108108,-0.006681654466]。仅已有固定OOF的条件区间，不完整反映重新训练/模型选择/CV重叠依赖，不保证公榜收益。

B4与B3-C均正确712、仅B4正确239、仅B3-C正确307、均错1441。真值辅助理想选择器BA=0.465995526、Accuracy=0.466098555，只是从两组固定预测择优的理论诊断，不是实际融合成绩、可部署规则或B5保证。互补性值得在固定B5中检验，本轮没有融合。

## 实际形态词表、文档频率与描述

| fold | char_actual | morphology_actual | total_actual | uppercase_ngram_count | lowercase_ngram_count | df_min | df_max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 80000 | 2804 | 82804 | 1060 | 2268 | 2 | 2159 |
| 2 | 80000 | 2866 | 82866 | 1047 | 2315 | 2 | 2159 |
| 3 | 80000 | 2698 | 82698 | 1023 | 2197 | 2 | 2159 |
| 4 | 80000 | 2261 | 82261 | 831 | 1851 | 2 | 2159 |
| 5 | 80000 | 2768 | 82768 | 1038 | 2244 | 2 | 2160 |

各折实际词表2261—2866，不把max_features=20000上限当实测。词表/IDF完整保存在各训练fold；高/低文档频率代表项见representative_ngrams.csv，文档频率由已训练平滑IDF反推后取整，明确不是原始term频次。fold1中`Aa`、` a`等DF=2159/IDF=1，稀有保留项DF可为2；A/a同时出现在实际词表，大小写模式未合并。

2699条转换长度逐条相同，大小写/数字模式、标点、引号、破折号/Markdown（转换层）保留；典型原文对照只用合成样例，不另共享官方原文。各类A/a/0比例及文本长度在shape_by_class.csv，原始/形态描述在shape_descriptive.csv。词长、句法、数字/URL/代码结构仍与题材相关；不能称完全消除主题，不能因某类未见标点而硬排除标签。全部描述不用于删特征、改权重或选超参。

## 成本与产物

| model | total_fit_seconds | total_predict_seconds | total_decision_seconds | total_inference_seconds | peak_process_mib |
| --- | --- | --- | --- | --- | --- |
| combo | 38.291958 | 3.887045 | 4.138824 | 8.025870 | 391.097656 |
| word | 4.216902 | 0.383802 | 0.367608 | 0.751411 | 224.613281 |
| char | 34.979316 | 3.663090 | 3.633593 | 7.296683 | 359.765625 |
| b3 | 41.730264 | 5.573639 | 5.017571 | 10.591209 | 407.257812 |
| b3c | 54.057905 | 6.773024 | 7.010904 | 13.783928 | 372.460938 |
| b4 | 61.252889 | 9.200678 | 8.993323 | 18.194001 | 374.343750 |

B4各折实际总维度82261—82866。本次相比历史B2训练/推理更慢，峰值较高；采集时段不同，不作受控硬件加速结论，峰值为进程生命周期。时间口径为模型fit/predict/decision，不包含全部审计I/O；完整运行时间见元数据起止。

正式artifacts目录保留2699×18 OOF、配置、源码/数据/分折SHA、逐折词表/IDF和audit；本目录共享指标/类别报告/两种混淆/主次对照/错误ID/稀疏/Bootstrap/代表词表/候选决策。参考fixture保存边界输出、原始2699条转换摘要及Unicode一致性结果。分析/验收源按实际时间另存，不冒充训练前文件。

## 验收范围与后续

预检11通过、2正式项暂未运行；训练后新增13、原B3相关34及B3-C相关8共55项通过，测试/分析不fit分类器。旧B3-C活动源字节冻结断言因授权工厂变更不运行，原测试文件不改，由当前阶段1277项保护、三份开发前源码逐字存档及全部旧模型实际参数/OOF校验替代；B3入口既有两项排除沿用原说明。不重跑S01—S06整套模型拟合。最终25项清单、Git/链接检查和历史指纹在final-acceptance.json。

B0—B3-C与B3的18维分数保留，默认B0不变。B4/B5当前不含Word的理由仅是未证明有用。若后续真实提交不佳，可将B4配置branches一项改为[word,char,morphology]重新启用，当前只做未拟合构建验证，不额外训练或搜索。未来B4自包含Notebook需嵌入mapping/批处理/Transformer/morphology_defaults及依赖，移除包内相对导入，冻结配置/权重并独立离线验证；本轮未生成Notebook或全量模型。

保留B4作为有独立正增量、与Style候选互补的研究方案；B3/B3-C仍有更高本地单模型分数。下一轮B5定义固定为Char+Style+Morphology，从B3-C只增本轮同配置Morphology；主比较B5−B3-C、次比较B5−B4，先冒烟再五折，不含Word、不搜索权重、不另设B5-C。B5/B6/B7本轮未训练。Rules/外部资源准入、已跟踪CSV、Jupyter、深入EDA/近重复、候选Notebook生成等遗留继续保留；无Kaggle/Git提交操作。
