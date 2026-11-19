# S07-B/B3-C 字符与风格组合消融

日期：2026-11-19。run_id：`s07b_b3c_char_style_v1_20261119_01`，冒烟：`s07b_b3c_char_style_v1_smoke_20261119_01`。正式运行只有一个新配置；B0—B3读取历史OOF，不重训。

## 结论与候选选择

B3-C完整OOF BA=0.377457121551，Accuracy=0.377547239718，MacroF1=0.364116658045。相比B3仅高0.000012428536 BA（+0.001242854个百分点），Accuracy完全相同、都正确1019条，MacroF1略低。十组稀疏4胜6负，prompt配对区间跨0，不能仅凭最高单次分数宣布优胜。**同时保留B3与B3-C；B3继续作为已建立的增强参考，默认仍B0。**

完整数据Fable支持149、其他各150；B3-C对Fable多对5条、其他合计少对5条，BA轻微差别来自宏召回权重，而非净正确数增长。这不推导隐藏测试分布。

## 单变量配置及实际隔离

新配置configs/s07b_b3c_char_style.json是B3的独立副本，唯一差异branches=[char,style]，实际没有构建/拟合Word向量器，不是把已拟合Word权重置零。现有src/ntl、风格公式/名称/顺序、B3原配置、默认B0及训练入口原字节未改。实际get_params除Word及组合对象表示外，与B3相同。

Char：char(2,5)、min_df2、max_features80000、float32、sublinear_tf/lowercase、l2、IDF/smooth_IDF、max_df1、权重1。Style：固定rich162-v1，训练折StandardScaler→clip[-5,5]→权重.075。LinearSVC：C1、balanced、max_iter3000、random_state20261118、tol1e-4、l2/squared_hinge/ovr、dual=auto及其余B3参数完全继承。

每折重新创建Pipeline；模型输入仅text，拟合目标model；id/prompt仅用于对应和隔离。每折scaler审计及split审计与B3文件逐字一致，训练/验证prompt不重叠。训练/验证样本2159/540（末折2160/539），保留150个prompt、18类及class_order。

## 实际命令与拟合次数

```powershell
.\.venv\Scripts\python.exe scripts/check_b3c.py
.\.venv\Scripts\python.exe scripts/run_experiment.py --mode smoke --run-id s07b_b3c_char_style_v1_smoke_20261119_01 --config configs/s07b_b3c_char_style.json --purpose char_style_ablation --data-dir data/name-that-llm
.\.venv\Scripts\python.exe scripts/run_experiment.py --mode formal --run-id s07b_b3c_char_style_v1_20261119_01 --smoke-run s07b_b3c_char_style_v1_smoke_20261119_01 --config configs/s07b_b3c_char_style.json --purpose char_style_ablation --data-dir data/name-that-llm
.\.venv\Scripts\python.exe scripts/analyze_b3c.py
.\.venv\Scripts\python.exe scripts/check_b3c.py --formal
.\.venv\Scripts\python.exe scripts/check_b3.py --formal --regressions
git diff --check
```

官方Pipeline新增fit=6（1冒烟+5正式），失败/重试0。冒烟BA=0.372222222，fit=10.587733s、predict+decision=2.987168s，80162维、18类/18维分数通过后才开始正式。模型告警0；Git行尾提示在独立stderr记录，不混称模型告警。测试只fit少量合成转换器，不fit分类器；分析阻断LinearSVC.fit。

## 全部模型完整OOF

| model | balanced_accuracy | accuracy | macro_f1 |
| --- | --- | --- | --- |
| B0 Word+Char | 0.307437 | 0.307521 | 0.295638 |
| B1 Word | 0.262612 | 0.262690 | 0.248365 |
| B2 Char | 0.314479 | 0.314561 | 0.304080 |
| B3 Word+Char+Style | 0.377445 | 0.377547 | 0.364327 |
| B3-C Char+Style | 0.377457 | 0.377547 | 0.364117 |

B3-C五折BA均值=0.377458492976，总体std(ddof0)=0.010755529278。完整2699条OOF与折均值分别报告。

| fold | b3 | b3c | b3c_minus_b3 | b3c_minus_b3_pp | outcome_b3c |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.368519 | 0.372222 | 0.003704 | 0.370370 | win |
| 2 | 0.368519 | 0.372222 | 0.003704 | 0.370370 | win |
| 3 | 0.387037 | 0.387037 | 0.000000 | 0.000000 | tie |
| 4 | 0.409259 | 0.392593 | -0.016667 | -1.666667 | loss |
| 5 | 0.353831 | 0.363218 | 0.009387 | 0.938697 | win |

3胜/1平/1负；第三折原始差5.55e-17，只按1e-12的数值容差记平，原始差保留。第4折下降抵消其他折增长，不以折间std较低单独宣布泛化更好。

## 十组相同稀疏掩码

| model | balanced_accuracy_mean | balanced_accuracy_std_ddof0 | balanced_accuracy_min | balanced_accuracy_max |
| --- | --- | --- | --- | --- |
| combo | 0.307500 | 0.028259 | 0.264930 | 0.353169 |
| word | 0.256880 | 0.027313 | 0.196096 | 0.295105 |
| char | 0.323946 | 0.030941 | 0.277591 | 0.362450 |
| b3 | 0.380641 | 0.026832 | 0.343032 | 0.426031 |
| b3c | 0.376696 | 0.025974 | 0.343843 | 0.435332 |

B3-C−B3配对均差=-0.003944402348（-0.394440pp），std=0.013314422709，范围[-0.030598149343,0.014526696347]，4胜/0平/6负。原10seed和掩码SHA c1c28dcfd25f8e22479a73a851bb7ed1b73be7e13d112a0d6498dff2bd6fe19f未变；每组300条、18类齐全，只算汇总BA。旧40条模型×seed指标与B3报告误差≤1e-15。重叠掩码不是10个独立实验。

## Word/Style四项增量与交互

| contrast | delta | delta_pp |
| --- | --- | --- |
| A_word_without_style | -0.007042 | -0.704201 |
| B_word_with_style | -0.000012 | -0.001243 |
| C_style_with_word | 0.070007 | 7.000746 |
| D_style_without_word | 0.062978 | 6.297788 |
| interaction_B_minus_A | 0.007030 | 0.702958 |

A=B0−B2；B=B3−B3-C；C=B3−B0；D=B3-C−B2；交互=B−A=C−D。每条原始差与百分点均保留，不用四舍五入后的指标相减。十组相同掩码的全部55条对照在factor_contrasts.csv，其汇总如下：

| contrast | mean_delta | std_ddof0 | min_delta | max_delta | mean_pp |
| --- | --- | --- | --- | --- | --- |
| A_word_without_style | -0.016446 | 0.007902 | -0.028149 | -0.002270 | -1.644629 |
| B_word_with_style | 0.003944 | 0.013314 | -0.014527 | 0.030598 | 0.394440 |
| C_style_with_word | 0.073141 | 0.012395 | 0.046489 | 0.091110 | 7.314106 |
| D_style_without_word | 0.052750 | 0.019327 | 0.021142 | 0.079892 | 5.275037 |
| interaction_B_minus_A | 0.020391 | 0.017400 | -0.007030 | 0.052321 | 2.039069 |

当前设置中，没有Style时Word呈负增量；有Style后完整OOF中的Word增量近零。Style在含Word/不含Word时均有较大正增量，但大小不同。正交互只描述固定数据/向量拼接/正则配置，不能证明普遍因果规律，也不能断言增强后Word永久无用。稀疏平均与完整小差方向不同，保留选型不确定性。

## 配对错误与18类召回

共同正确926、共同错误1587、B3-C修正93、新增错误93，净正确增量0。预测一致2276/2699=84.327529%。同厂商错误835→841，跨厂商845→839，不支持兄弟模型混淆整体改善。

| label | b3 | b3c | b3c_minus_b3_pp | b3_correct | b3c_correct |
| --- | --- | --- | --- | --- | --- |
| claude-fable-5 | 0.100671 | 0.134228 | 3.355705 | 15 | 20 |
| claude-opus-4-6 | 0.486667 | 0.533333 | 4.666667 | 73 | 80 |
| claude-opus-4-8 | 0.240000 | 0.246667 | 0.666667 | 36 | 37 |
| claude-opus-5 | 0.666667 | 0.660000 | -0.666667 | 100 | 99 |
| claude-sonnet-5 | 0.380000 | 0.333333 | -4.666667 | 57 | 50 |
| gemini-3.1-pro-preview | 0.806667 | 0.800000 | -0.666667 | 121 | 120 |
| gemini-3.6-flash | 0.420000 | 0.380000 | -4.000000 | 63 | 57 |
| gemini-3.7-flash | 0.440000 | 0.473333 | 3.333333 | 66 | 71 |
| gemini-3.8-flash | 0.413333 | 0.400000 | -1.333333 | 62 | 60 |
| glm-5.3 | 0.206667 | 0.186667 | -2.000000 | 31 | 28 |
| glm-5.3-flash | 0.200000 | 0.166667 | -3.333333 | 30 | 25 |
| gpt-5.5 | 0.326667 | 0.373333 | 4.666667 | 49 | 56 |
| gpt-5.6-luna | 0.340000 | 0.353333 | 1.333333 | 51 | 53 |
| gpt-5.6-sol | 0.180000 | 0.193333 | 1.333333 | 27 | 29 |
| gpt-5.6-terra | 0.260000 | 0.246667 | -1.333333 | 39 | 37 |
| gpt-6-astra | 0.620000 | 0.626667 | 0.666667 | 93 | 94 |
| kimi-k3 | 0.246667 | 0.240000 | -0.666667 | 37 | 36 |
| qwen3.8-max | 0.460000 | 0.446667 | -1.333333 | 69 | 67 |

重点弱类：Fable15→20/149、Sol27→29、Opus4.8 36→37改善；GLM31→28、GLM Flash30→25、Kimi37→36下降。Gemini3.6 63→57、3.7 66→71、3.8 62→60。Gemini3.7→3.8错误42→37，但反向43→44；Sol→Luna32→33，GLM Flash→GLM22→23。全部306个非对角方向和无文本ID级变化均保存，不能只列改善项。

## 预先固定的Bootstrap

在新模型读分前冻结seed=20261118、5000次、150个prompt为配对重采样单位。B3-C−B3的条件95%百分位区间[-0.010347888671,0.009681900861] BA，即[-1.034789,0.968190]pp，跨0。基于已有固定OOF，不重训，不完整反映重叠训练、模型选择或公榜不确定性。

## 特征宽度与成本

| fold | char | style | b3c_actual | b3_actual | removed_word_actual |
| --- | --- | --- | --- | --- | --- |
| 1 | 80000 | 162 | 80162 | 120162 | 40000 |
| 2 | 80000 | 162 | 80162 | 120096 | 39934 |
| 3 | 80000 | 162 | 80162 | 120162 | 40000 |
| 4 | 80000 | 162 | 80162 | 120162 | 40000 |
| 5 | 80000 | 162 | 80162 | 120162 | 40000 |

| model | total_fit_seconds | total_predict_seconds | total_decision_seconds | total_inference_seconds | peak_process_mib |
| --- | --- | --- | --- | --- | --- |
| combo | 38.291958 | 3.887045 | 4.138824 | 8.025870 | 391.097656 |
| word | 4.216902 | 0.383802 | 0.367608 | 0.751411 | 224.613281 |
| char | 34.979316 | 3.663090 | 3.633593 | 7.296683 | 359.765625 |
| b3 | 41.730264 | 5.573639 | 5.017571 | 10.591209 | 407.257812 |
| b3c | 54.057905 | 6.773024 | 7.010904 | 13.783928 | 372.460938 |

每折Char80000+Style162=80162；实际移除Word40000（第二折39934）。本次B3-C观测内存更低、训练和推理更慢；不能从减少维数推断一定更快。采集时段不同，峰值是进程生命周期而非每折独立峰值。

## 验收、产物及边界

预检7通过、2项正式检查暂未执行；正式后新增9项全过，原B3及相关34项回归通过，共43项，无新增分类器测试训练。阶段前1085项保护中1084项当前字节不变（全部src、旧scripts/配置、数据、两套分折、掩码、Notebook、历史结果），实验CSV只追加1条真实B3-C行并保留原字节前缀；最终15项验收见final-acceptance.json。

artifacts独占正式目录保存2699×18 OOF、配置、元数据、源码/数据SHA、逐折split/metrics/warnings/style审计；本目录共享报告、P/R/F1、混淆矩阵、特征宽度、五模型指标、50条稀疏成绩、55条因子对照、配对错误、Bootstrap及冻结分析方案。分析/验收源码按真实时间归档，不冒称后续新增文件已在训练前存在。

保留B3和B3-C两个候选，现有B3仍为增强参考，默认B0不变。B4继续是B0+Morphology的独立受控增量，不因本轮结果擅自改为Char+Style+Morphology；是否增设该组合等待审阅。没有B4、B5—B7、全量训练、v2选模、参数搜索、外部下载、Notebook生成或Kaggle提交。Rules/资源准入、已跟踪CSV、Jupyter、深入EDA/近重复及候选Notebook生成待办继续保留。

相关回归沿用B3入口：未运行一项会新增合成分类器fit的旧测试及一项S07-A开发前活动源文件字节冻结断言，其原有替代保护见B3验收。当前B3-C阶段新增1085项全历史保护，不修改任何旧测试或源文件。未重跑S01—S06整套拟合/容量脚本，不把它们记作本轮已执行。
