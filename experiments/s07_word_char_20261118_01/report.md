# S07-01 Word/Char TF-IDF 消融实验报告

实验于 2026-11-18 启动，2026-11-19 汇总验收，跨日保留启动时分配的运行编号。固定官方 2699 条、150 prompt、18 类、folds_v1 和 seed=20261118；唯一变量为启用词、字符或两支。分类器和各支完整参数不变。组合结果复用 S06-B，没有再次训练。

## 1 假设与控制变量

H1/H2：词/字符表示可能提供模型识别信号；H3：两支可能互补且组合可能优于单支。这些是研究假设，不能由分数直接证明词汇/标点/词缀的因果作用。每折新建 Pipeline，仅当前训练 text 拟合，验证只预测；id/prompt/测试不参与特征。

## 2 总体指标

| experiment | run_id | balanced_accuracy | accuracy | macro_f1 | mean_fold_balanced_accuracy | std_fold_balanced_accuracy_ddof0 | ba_minus_combo | ba_minus_combo_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combo | s06b_equivalence_cv_20261118_01 | 0.307437 | 0.307521 | 0.295638 | 0.307446 | 0.011192 | 0.000000 | 0.000000 |
| word | s07_word_only_v1_20261118_01 | 0.262612 | 0.262690 | 0.248365 | 0.262605 | 0.017282 | -0.044825 | -4.482476 |
| char | s07_char_only_v1_20261118_01 | 0.314479 | 0.314561 | 0.304080 | 0.314508 | 0.016865 | 0.007042 | 0.704201 |

百分点=原始 BA 差值×100，不是相对增长率。主指标为整体 OOF BA，折均值另列。

## 3 五折配对对照

| fold | n_train | n_valid | combo_ba | word_ba | char_ba | word_minus_combo | char_minus_combo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2159 | 540 | 0.296296 | 0.231481 | 0.305556 | -0.064815 | 0.009259 |
| 2 | 2159 | 540 | 0.292593 | 0.262963 | 0.287037 | -0.029630 | -0.005556 |
| 3 | 2159 | 540 | 0.312963 | 0.261111 | 0.318519 | -0.051852 | 0.005556 |
| 4 | 2159 | 540 | 0.322222 | 0.277778 | 0.325926 | -0.044444 | 0.003704 |
| 5 | 2160 | 539 | 0.313155 | 0.279693 | 0.335504 | -0.033461 | 0.022350 |

{"word": {"folds_above_combo": 0, "folds_below_combo": 5, "folds_tied_combo": 0, "paired_mean_fold_delta": -0.04484035759897827}, "char": {"folds_above_combo": 4, "folds_below_combo": 1, "folds_tied_combo": 0, "paired_mean_fold_delta": 0.007062579821200543}}

## 4 全部类别召回与正确数量

| model | support | combo_correct | word_correct | char_correct | combo_recall | word_recall | char_recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| claude-fable-5 | 149 | 12 | 8 | 14 | 0.080537 | 0.053691 | 0.093960 |
| claude-opus-4-6 | 150 | 51 | 34 | 61 | 0.340000 | 0.226667 | 0.406667 |
| claude-opus-4-8 | 150 | 32 | 23 | 37 | 0.213333 | 0.153333 | 0.246667 |
| claude-opus-5 | 150 | 91 | 82 | 93 | 0.606667 | 0.546667 | 0.620000 |
| claude-sonnet-5 | 150 | 40 | 33 | 35 | 0.266667 | 0.220000 | 0.233333 |
| gemini-3.1-pro-preview | 150 | 111 | 105 | 109 | 0.740000 | 0.700000 | 0.726667 |
| gemini-3.6-flash | 150 | 65 | 69 | 58 | 0.433333 | 0.460000 | 0.386667 |
| gemini-3.7-flash | 150 | 51 | 40 | 59 | 0.340000 | 0.266667 | 0.393333 |
| gemini-3.8-flash | 150 | 42 | 41 | 49 | 0.280000 | 0.273333 | 0.326667 |
| glm-5.3 | 150 | 21 | 22 | 21 | 0.140000 | 0.146667 | 0.140000 |
| glm-5.3-flash | 150 | 22 | 15 | 26 | 0.146667 | 0.100000 | 0.173333 |
| gpt-5.5 | 150 | 38 | 27 | 40 | 0.253333 | 0.180000 | 0.266667 |
| gpt-5.6-luna | 150 | 37 | 33 | 40 | 0.246667 | 0.220000 | 0.266667 |
| gpt-5.6-sol | 150 | 21 | 16 | 22 | 0.140000 | 0.106667 | 0.146667 |
| gpt-5.6-terra | 150 | 38 | 37 | 38 | 0.253333 | 0.246667 | 0.253333 |
| gpt-6-astra | 150 | 74 | 59 | 70 | 0.493333 | 0.393333 | 0.466667 |
| kimi-k3 | 150 | 45 | 37 | 32 | 0.300000 | 0.246667 | 0.213333 |
| qwen3.8-max | 150 | 39 | 28 | 45 | 0.260000 | 0.186667 | 0.300000 |

完整 Precision/Recall/F1 保存在仓库 experiments/s07_word_char_20261118_01/class-comparison.csv，正确数和 support 直接从 OOF 重算。

### 特征偏好与组合影响

Word 相对 Char 召回差最大的类别：

| model | word_correct | char_correct | word_minus_char_recall |
| --- | --- | --- | --- |
| gemini-3.6-flash | 69 | 58 | 0.073333 |
| kimi-k3 | 37 | 32 | 0.033333 |
| glm-5.3 | 22 | 21 | 0.006667 |
| gpt-5.6-terra | 37 | 38 | -0.006667 |
| claude-sonnet-5 | 33 | 35 | -0.013333 |

Char 相对 Word 优势最大的类别：

| model | word_correct | char_correct | word_minus_char_recall |
| --- | --- | --- | --- |
| claude-opus-4-6 | 34 | 61 | -0.180000 |
| gemini-3.7-flash | 40 | 59 | -0.126667 |
| qwen3.8-max | 28 | 45 | -0.113333 |
| claude-opus-4-8 | 23 | 37 | -0.093333 |
| gpt-5.5 | 27 | 40 | -0.086667 |

组合相对每类最佳单支的正确数增减（描述当前样本，不作显著性结论）：

| model | combo_minus_best_single_correct |
| --- | --- |
| claude-fable-5 | -2 |
| claude-opus-4-6 | -10 |
| claude-opus-4-8 | -5 |
| claude-opus-5 | -2 |
| claude-sonnet-5 | 5 |
| gemini-3.1-pro-preview | 2 |
| gemini-3.6-flash | -4 |
| gemini-3.7-flash | -8 |
| gemini-3.8-flash | -7 |
| glm-5.3 | -1 |
| glm-5.3-flash | -4 |
| gpt-5.5 | -2 |
| gpt-5.6-luna | -3 |
| gpt-5.6-sol | -1 |
| gpt-5.6-terra | 0 |
| gpt-6-astra | 4 |
| kimi-k3 | 8 |
| qwen3.8-max | -6 |

## 5 主要混淆变化

| true_model | predicted_model | combo | word | char |
| --- | --- | --- | --- | --- |
| gemini-3.8-flash | gemini-3.7-flash | 47 | 46 | 42 |
| gemini-3.7-flash | gemini-3.8-flash | 41 | 43 | 43 |
| gpt-5.6-sol | gpt-5.6-luna | 34 | 33 | 29 |
| gpt-5.6-luna | gpt-5.6-sol | 18 | 17 | 20 |

原组合错误最多的前 12 个方向：

| true_model | predicted_model | combo | word | char |
| --- | --- | --- | --- | --- |
| gemini-3.8-flash | gemini-3.7-flash | 47 | 46 | 42 |
| gemini-3.7-flash | gemini-3.8-flash | 41 | 43 | 43 |
| gpt-5.6-sol | gpt-5.6-luna | 34 | 33 | 29 |
| gemini-3.6-flash | gemini-3.7-flash | 33 | 32 | 37 |
| gemini-3.7-flash | gemini-3.6-flash | 32 | 32 | 24 |
| glm-5.3 | claude-opus-5 | 31 | 30 | 28 |
| gpt-5.6-luna | gpt-6-astra | 30 | 27 | 25 |
| gpt-5.6-sol | gpt-5.6-terra | 28 | 21 | 26 |
| gemini-3.8-flash | gemini-3.6-flash | 27 | 25 | 27 |
| gpt-5.6-terra | gpt-5.6-luna | 26 | 23 | 30 |
| gpt-5.6-luna | gpt-5.6-terra | 26 | 24 | 22 |
| kimi-k3 | gpt-5.5 | 25 | 21 | 27 |

Char-only 自身错误最多的前 8 个方向：

| true_model | predicted_model | combo | word | char |
| --- | --- | --- | --- | --- |
| gemini-3.7-flash | gemini-3.8-flash | 41 | 43 | 43 |
| gemini-3.8-flash | gemini-3.7-flash | 47 | 46 | 42 |
| gemini-3.6-flash | gemini-3.7-flash | 33 | 32 | 37 |
| gpt-5.6-terra | gpt-5.6-luna | 26 | 23 | 30 |
| gpt-5.6-sol | gpt-5.6-luna | 34 | 33 | 29 |
| gemini-3.6-flash | gemini-3.1-pro-preview | 24 | 18 | 29 |
| glm-5.3 | claude-opus-5 | 31 | 30 | 28 |
| kimi-k3 | gpt-5.5 | 25 | 21 | 27 |

## 6 预测互补性及组合利用

| group | n | combo_correct |
| --- | --- | --- |
| both_correct | 482 | 479 |
| word_only_correct | 227 | 119 |
| char_only_correct | 367 | 204 |
| both_wrong | 1623 | 28 |

Word/Char 预测一致率 0.444980（1201/2699）；至少一个正确 1076 条，占 0.398666。

两支恰好一个正确的 594 条中，组合识别对 323 条，利用率 0.543771。组合额外救回两支都错的 28 条，同时漏掉单支至少一个正确的 274 条。

事后 Oracle Accuracy=0.398666、BA=0.398571。Oracle 使用真值在两个固定预测中选择，仅供互补性参考，不是可部署模型、可保证的融合成绩，也不是组合特征分类器的绝对上界。本轮没有分数相加、校准或实际融合。

## 7 Prompt 配对 Bootstrap

| comparison | observed_delta | percentile_95_lower | percentile_95_upper |
| --- | --- | --- | --- |
| word_minus_combo | -0.044825 | -0.057793 | -0.031481 |
| char_minus_combo | 0.007042 | -0.004812 | 0.019262 |
| char_minus_word | 0.051867 | 0.034134 | 0.068925 |

整组 prompt 配对重采样 5000 次，固定种子 20261118，95% 百分位区间仅描述当前固定 OOF 差异的不确定性；不是逐行独立抽样。未重训模型，未覆盖 CV 训练集重叠依赖，不能当独立测试显著性结论。

## 8 成本、维度与资源

| experiment | total_fit_seconds | total_predict_seconds | total_decision_seconds | total_inference_seconds | feature_dimensions_by_fold | warning_count |
| --- | --- | --- | --- | --- | --- | --- |
| combo | 38.291958 | 3.887045 | 4.138824 | 8.025870 | 120000,119934,120000,120000,120000 | 0 |
| word | 4.216902 | 0.383802 | 0.367608 | 0.751411 | 40000,39934,40000,40000,40000 | 0 |
| char | 34.979316 | 3.663090 | 3.633593 | 7.296683 | 80000,80000,80000,80000,80000 | 0 |

| fold | word_actual | char_actual | combo_derived |
| --- | --- | --- | --- |
| 1 | 40000 | 80000 | 120000 |
| 2 | 39934 | 80000 | 119934 |
| 3 | 40000 | 80000 | 120000 |
| 4 | 40000 | 80000 | 120000 |
| 5 | 40000 | 80000 | 120000 |

Word/Char 是实际拟合词表长度并与分类器系数宽度核对；未把 max_features 当实测值。旧组合未记录维度，其同折维度由两支实测长度相加推导：数据、训练 ID、TF-IDF 参数相同，原 FeatureUnion 独立拼接；这不是历史直接观测。

Same perf_counter boundaries: fold fit, predict, decision separately. Historical baseline and new runs occur at different times; no hardware-controlled speedup claim. RSS are snapshots; peak_wset is process-lifetime, not independent per-fold peaks.

逐折实际资源（RSS 和峰值为字节）：

| experiment | fold | actual_or_derived_features | fit_seconds | inference_seconds | rss_before_bytes | rss_after_bytes | process_peak_wset_bytes | warning_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combo | 1 | 120000 | 7.582455 | 1.598078 | 190304256 | 238915584 | 410095616 | 0 |
| word | 1 | 40000 | 0.903149 | 0.150859 | 189976576 | 214519808 | 235524096 | 0 |
| char | 1 | 80000 | 7.107491 | 1.528302 | 190504960 | 216846336 | 376238080 | 0 |
| combo | 2 | 119934 | 7.267155 | 1.625085 | 212406272 | 239980544 | 410095616 | 0 |
| word | 2 | 39934 | 0.831594 | 0.147494 | 204419072 | 215187456 | 235524096 | 0 |
| char | 2 | 80000 | 6.928032 | 1.506080 | 200982528 | 218902528 | 376238080 | 0 |
| combo | 3 | 120000 | 7.369846 | 1.584605 | 213811200 | 241975296 | 410095616 | 0 |
| word | 3 | 40000 | 0.827349 | 0.149407 | 208867328 | 219668480 | 235524096 | 0 |
| char | 3 | 80000 | 7.060954 | 1.416961 | 205762560 | 221040640 | 377008128 | 0 |
| combo | 4 | 120000 | 6.560185 | 1.272334 | 216227840 | 241983488 | 410095616 | 0 |
| word | 4 | 40000 | 0.842942 | 0.155174 | 210333696 | 220991488 | 235524096 | 0 |
| char | 4 | 80000 | 6.966243 | 1.418106 | 205946880 | 223956992 | 377241600 | 0 |
| combo | 5 | 120000 | 9.512318 | 1.945767 | 211763200 | 242475008 | 410095616 | 0 |
| word | 5 | 40000 | 0.811868 | 0.148477 | 208576512 | 218578944 | 235524096 | 0 |
| char | 5 | 80000 | 6.916596 | 1.427233 | 206831616 | 221581312 | 377241600 | 0 |

## 9 结论与下一轮

当前最高 OOF BA 方案为 char。排名只针对本数据、分折和当前参数；正式 V1.0 默认组合保持不变，不自动替换或提交。

两支存在的独占正确样本支持“互补预测集合”，但不保证默认拼接能利用全部互补信息。H3 中“组合优于两支”须依据真实总体/配对结果判断，不能由 Oracle 证明。

本轮实测候选、假设判断与下一轮单因素建议见第 10 节，本轮不实施 S07-02。

所有成功/失败日志保留。新公榜分数为空；0.311 只属于此前已提交的版本。原数据、v1/v2、历史实验和默认配置继续保护。S07-01 的分类器替换部分未实施，原完整任务不勾为全部完成。


## 10 本轮数据支持的判断

Word 的召回高于 Char 的类别共 3 个；Char 高于 Word 共 15 个。前者具体计数如下，不能因 Word 总分较低而忽略其局部贡献：

| model | support | word_correct | char_correct |
| --- | --- | --- | --- |
| gemini-3.6-flash | 150 | 69 | 58 |
| glm-5.3 | 150 | 22 | 21 |
| kimi-k3 | 150 | 37 | 32 |

组合高于每类最佳单支的类别共 4 个，说明词分支在默认拼接下仍可能对部分标签有用：

| model | combo_correct | word_correct | char_correct | combo_minus_best_single_correct |
| --- | --- | --- | --- | --- |
| claude-sonnet-5 | 40 | 33 | 35 | 5 |
| gemini-3.1-pro-preview | 111 | 105 | 109 | 2 |
| gpt-6-astra | 74 | 59 | 70 | 4 |
| kimi-k3 | 45 | 37 | 32 | 8 |

三方案下均较困难的类别：

| model | support | combo_correct | word_correct | char_correct | best_recall |
| --- | --- | --- | --- | --- | --- |
| claude-fable-5 | 149 | 12 | 8 | 14 | 0.093960 |
| glm-5.3 | 150 | 21 | 22 | 21 | 0.146667 |
| gpt-5.6-sol | 150 | 21 | 16 | 22 | 0.146667 |

Char−组合的当前 OOF BA 差值为 0.007042，prompt 配对 95% 区间 [-0.004812, 0.019262]。四折提高但第五折贡献了最大的提升，区间包含 0；不能据此直接替换正式方案或保证公榜收益。

H1/H2 得到“这两类表示分别包含可用分类信号”的经验支持，字符方案在本设置下更强；未隔离词汇/标点/词缀的因果作用。H3 的“独占正确样本互补”得到支持，而“默认组合优于最佳单支”在本轮不成立。

保留 V1.0 组合作正式长期基线，Char-only 作为下一轮主要研究候选，Word-only 作为成本低的对照；均不自动换默认或提交。

S07-02 首选一个单因素实验：以本轮 Char-only 为对照，仅将 features.char.lowercase 改为 false，固定其余参数、seed 和 v1，先冒烟再正式五折。根据结果再决定是否单独做 char_wb 或词分支降权对照；不同时调整 n-gram、词表规模和 C。上述均未在本轮实施。


## 11 复现与记录

实际训练命令如下；原运行目录已经存在，复跑必须换新编号，不覆盖历史：

```powershell
.\.venv\Scripts\python.exe scripts/run_experiment.py --mode smoke --run-id s07_word_only_v1_smoke_20261118_01 --config configs/s07_word_only.json --purpose s07_word_char_ablation
.\.venv\Scripts\python.exe scripts/run_experiment.py --mode formal --run-id s07_word_only_v1_20261118_01 --config configs/s07_word_only.json --purpose s07_word_char_ablation --smoke-run s07_word_only_v1_smoke_20261118_01
.\.venv\Scripts\python.exe scripts/run_experiment.py --mode smoke --run-id s07_char_only_v1_smoke_20261118_01 --config configs/s07_char_only.json --purpose s07_word_char_ablation
.\.venv\Scripts\python.exe scripts/run_experiment.py --mode formal --run-id s07_char_only_v1_20261118_01 --config configs/s07_char_only.json --purpose s07_word_char_ablation --smoke-run s07_char_only_v1_smoke_20261118_01
.\.venv\Scripts\python.exe scripts/check_s07.py
.\.venv\Scripts\python.exe scripts/analyze_s07.py
```

原始文件与固定分折 SHA-256：

| file | sha256 |
| --- | --- |
| sample_submission.parquet | ccf86796b80fd7f83d8463664603fd1208e17559abcf416f198776b819646188 |
| test.csv | 4545aa3a3459f4c0724f97ad5bd06c4caa3506bdfea5fab905e8203184fc0c57 |
| train.csv | 40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707 |
| folds_v1.csv | 9e5e0f3db0a359bf314f904862a22d70db92b27c1d91eeb414d8b554e3556af7 |
| folds_v1_metadata.json | 665aa28c5d03fe2f301be041da39712f08c54779cbce6e730a343de3405a6811 |

完整 OOF、18 维分数、逐折审计/警告/资源、源码与配置在 artifacts/<run_id>/，轻量结果和比较表在 experiments。新报告和输出目录默认拒绝覆盖，已有报告需使用新路径重算。

必要回归使用既有已保存模型和 OOF，不再次训练组合基线或全量候选；完整性、测试命令及日志见 L012/L013。
