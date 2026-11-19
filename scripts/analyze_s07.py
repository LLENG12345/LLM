"""Read validated OOF artifacts for the controlled word/char ablation; no training."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from ntl.io import digest, write_json
from ntl.labels import class_order
from ntl.models import make_model, model_config, baseline_config
from ntl.validation import validate_run

ROOT = Path(__file__).resolve().parents[1]
RUNS = {'combo': 's06b_equivalence_cv_20261118_01',
        'word': 's07_word_only_v1_20261118_01', 'char': 's07_char_only_v1_20261118_01'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def align_oof(frames):
    """Align by original baseline string IDs; reject any ambiguity or provenance mismatch."""
    base = next(iter(frames.values()))
    if base['id'].isna().any() or not base['id'].is_unique:
        raise ValueError('Reference OOF has missing/duplicate IDs')
    ids = base['id'].tolist()
    aligned = {}
    for name, frame in frames.items():
        if (frame['id'].isna().any() or not frame['id'].is_unique or set(frame['id']) != set(ids)
                or not all(isinstance(value,str) for value in frame['id'])):
            raise ValueError(f'{name}: OOF ID coverage/uniqueness mismatch')
        current = frame.set_index('id').loc[ids].reset_index()
        for column in ('prompt_id', 'fold', 'true_model'):
            if current[column].tolist() != base[column].tolist():
                raise ValueError(f'{name}: OOF {column} mismatch')
        aligned[name] = current
    return aligned


def complementarity(truth, word, char, combo):
    truth, word, char, combo = map(np.asarray, (truth, word, char, combo))
    if not (len(truth) == len(word) == len(char) == len(combo)) or len(truth) == 0:
        raise ValueError('Complementarity needs equal nonempty arrays')
    w, c, b = word == truth, char == truth, combo == truth
    groups = {'both_correct': w & c, 'word_only_correct': w & ~c,
              'char_only_correct': ~w & c, 'both_wrong': ~w & ~c}
    details = {name: {'n': int(mask.sum()), 'combo_correct': int((mask & b).sum())}
               for name, mask in groups.items()}
    exclusive = w ^ c
    # Uses truth only for a diagnostic ceiling over these TWO fixed predictions.
    oracle = np.where(w, word, np.where(c, char, word))
    return {'n_samples': len(truth), 'groups': details,
        'prediction_agreement_count': int((word == char).sum()),
        'prediction_agreement_rate': float((word == char).mean()),
        'at_least_one_correct_count': int((w | c).sum()),
        'at_least_one_correct_rate': float((w | c).mean()),
        'exclusive_complementary_count': int(exclusive.sum()),
        'combo_correct_exclusive_count': int((exclusive & b).sum()),
        'combo_correct_exclusive_rate': float(b[exclusive].mean()) if exclusive.any() else None,
        'combo_missed_single_oracle_correct_count': int(((w | c) & ~b).sum()),
        'combo_rescued_both_wrong_count': int((~w & ~c & b).sum()),
        'oracle_accuracy': float((oracle == truth).mean()),
        'oracle_balanced_accuracy': float(balanced_accuracy_score(truth, oracle)),
        'oracle_is_deployable': False,
        'oracle_note': 'Truth-assisted choice between the two fixed single-branch predictions only; not a deployable score, guaranteed ensemble result, or bound on the combined-feature classifier.'}


def prompt_bootstrap(truth, prompts, predictions, repeats=5000, seed=20261118):
    """Paired resampling of whole prompts, conditional on fixed OOF predictions."""
    truth, prompts = np.asarray(truth), np.asarray(prompts)
    groups, group_id = np.unique(prompts, return_inverse=True)
    classes, class_id = np.unique(truth, return_inverse=True)
    support = np.zeros((len(groups), len(classes)), dtype=np.int64)
    np.add.at(support, (group_id, class_id), 1)
    weights = np.random.default_rng(seed).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=repeats)
    denominator = weights @ support
    if not (denominator > 0).all():
        raise ValueError('A bootstrap sample lost a class; revise/report the estimator, do not silently discard')
    distributions, points = {}, {}
    for name, pred in predictions.items():
        correct = np.zeros_like(support)
        np.add.at(correct, (group_id, class_id), np.asarray(pred) == truth)
        distributions[name] = ((weights @ correct) / denominator).mean(axis=1)
        points[name] = float(balanced_accuracy_score(truth, pred))
    intervals = {}
    for left, right in [('word','combo'), ('char','combo'), ('char','word')]:
        delta = distributions[left] - distributions[right]
        lower, upper = np.quantile(delta, [0.025, 0.975])
        intervals[left + '_minus_' + right] = {'observed_delta': points[left] - points[right],
            'percentile_95_lower': float(lower), 'percentile_95_upper': float(upper)}
    return {'unit': 'prompt', 'n_prompts': len(groups), 'repeats': repeats, 'seed': seed,
        'intervals': intervals,
        'limitation': 'Conditional uncertainty reference for the current fixed OOF differences; preserves within-prompt dependence, does not retrain models or model dependence from overlapping CV training sets. Not an independent-test significance claim.'}


def check_controls(config, branch):
    base = model_config(make_model())
    resolved = config['experiment_configuration']
    expected = baseline_config(); expected['branches'] = [branch]
    assert resolved == expected, f'{branch}: unexpected experimental change'
    current = config['model_parameters']
    other = 'char' if branch == 'word' else 'word'
    removed = {k for k in base if k == f'features__{other}' or k.startswith(f'features__{other}__')}
    assert set(current) == set(base) - removed
    allowed = {'steps', 'features', 'features__transformer_list'}
    assert all(current[k] == v for k, v in base.items() if k not in removed | allowed)
    assert current == model_config(make_model({'branches': [branch]}))
    assert config['class_order'] == list(class_order) and config['random_seed'] == 20261118


def load_runs():
    results = {}
    for name, identifier in RUNS.items():
        directory = ROOT / 'artifacts' / identifier
        validate_run(directory)
        results[name] = {'directory': directory, 'config': read(directory/'config.json'),
            'meta': read(directory/'run_metadata.json'), 'metrics': read(directory/'metrics.json'),
            'oof': pd.read_parquet(directory/'oof_predictions.parquet'),
            'classes': pd.read_csv(directory/'classification_report.csv', index_col=0),
            'folds': [read(directory/f'fold_{i}/metrics.json') for i in range(1,6)]}
    baseline = results['combo']
    assert baseline['config']['model_parameters'] == model_config(make_model())
    for name in ('word','char'):
        candidate = results[name]; check_controls(candidate['config'], name)
        assert candidate['meta']['input_sha256'] == baseline['meta']['input_sha256']
        assert candidate['config']['class_order'] == baseline['config']['class_order']
        assert candidate['config']['score_columns'] == baseline['config']['score_columns']
        assert candidate['config']['fold_version'] == 'folds_v1'
        for i, metric in enumerate(candidate['folds'], 1):
            assert metric['feature_branches'] == [name]
            assert list(metric['feature_dimensions']) == [name]
            assert metric['n_features'] == metric['feature_dimensions'][name] > 0
            assert metric['n_features'] <= (40000 if name == 'word' else 80000)
            assert read(candidate['directory']/f'fold_{i}/split_audit.json') == read(baseline['directory']/f'fold_{i}/split_audit.json')
    aligned = align_oof({name:r['oof'] for name,r in results.items()})
    for name, frame in aligned.items(): results[name]['oof'] = frame
    return results


def markdown(frame, columns=None):
    if columns is not None: frame = frame[columns]
    def render(value):
        if isinstance(value, (float, np.floating)): return f'{value:.6f}'
        return str(value).replace('|','/')
    header = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
    return '\n'.join([header, '| ' + ' | '.join(['---']*len(frame.columns)) + ' |',
        *['| ' + ' | '.join(render(v) for v in row) + ' |' for row in frame.itertuples(index=False,name=None)]])


def analyze(output, report_path):
    output, report_path = Path(output), Path(report_path)
    if output.exists() or report_path.exists(): raise FileExistsError('Analysis/report output already exists')
    runs = load_runs(); base = runs['combo']; base_ba = base['metrics']['balanced_accuracy']
    dimension_rows, resource_rows, fold_rows, overall_rows = [], [], [], []
    for i in range(5):
        wd, cd = runs['word']['folds'][i]['n_features'], runs['char']['folds'][i]['n_features']
        dimension_rows.append({'fold':i+1,'word_actual':wd,'char_actual':cd,'combo_derived':wd+cd})
        row = {'fold':i+1,'n_train':base['folds'][i]['n_train'],'n_valid':base['folds'][i]['n_valid']}
        for name in RUNS:
            metric = runs[name]['folds'][i]
            row[name+'_ba'] = metric['balanced_accuracy']
            row[name+'_accuracy'] = metric['accuracy']; row[name+'_macro_f1'] = metric['macro_f1']
            if name != 'combo':
                row[name+'_minus_combo'] = metric['balanced_accuracy']-base['folds'][i]['balanced_accuracy']
                row[name+'_minus_combo_pp'] = 100*row[name+'_minus_combo']
            resource_rows.append({'experiment':name,'fold':i+1,'actual_or_derived_features':wd+cd if name=='combo' else (wd if name=='word' else cd),
                **{k:metric[k] for k in ('fit_seconds','predict_seconds','decision_seconds','inference_seconds',
                     'rss_before_bytes','rss_after_bytes','process_peak_wset_bytes','warning_count','n_iter')}})
        fold_rows.append(row)
    for name, run in runs.items():
        metric = run['metrics']; dimension = [r['combo_derived' if name=='combo' else name+'_actual'] for r in dimension_rows]
        overall_rows.append({'experiment':name,'run_id':RUNS[name],
            **{k:metric[k] for k in ('balanced_accuracy','accuracy','macro_f1','mean_fold_balanced_accuracy',
                 'std_fold_balanced_accuracy_ddof0','total_fit_seconds','total_predict_seconds','total_decision_seconds','total_inference_seconds','warning_count')},
            'ba_minus_combo':metric['balanced_accuracy']-base_ba,'ba_minus_combo_pp':100*(metric['balanced_accuracy']-base_ba),
            'feature_dimensions_by_fold':','.join(map(str,dimension)),
            'dimension_origin':'sum of actual same-fold standalone branch vocabularies; inferred for historical FeatureUnion' if name=='combo' else 'observed fitted vocabularies and classifier coefficient width'})
    class_rows = []
    for label in class_order:
        row = {'model':label,'support':int(base['classes'].loc[label,'support'])}
        for name, run in runs.items():
            oof = run['oof']; mask = oof.true_model.eq(label)
            row[name+'_correct'] = int((mask & oof.predicted_model.eq(label)).sum())
            for metric in ('precision','recall','f1-score'): row[name+'_'+metric] = float(run['classes'].loc[label,metric])
        row['word_minus_char_recall'] = row['word_recall']-row['char_recall']
        row['combo_minus_best_single_correct'] = row['combo_correct']-max(row['word_correct'],row['char_correct'])
        class_rows.append(row)
    classes = pd.DataFrame(class_rows)
    confusion_rows = []
    for true in class_order:
        for predicted in class_order:
            if true == predicted: continue
            row = {'true_model':true,'predicted_model':predicted}
            for name,run in runs.items():
                oof = run['oof']; row[name] = int((oof.true_model.eq(true)&oof.predicted_model.eq(predicted)).sum())
            row['word_minus_combo'] = row['word']-row['combo']; row['char_minus_combo'] = row['char']-row['combo']
            confusion_rows.append(row)
    confusions = pd.DataFrame(confusion_rows).sort_values(['combo','char','word'],ascending=False)
    truth = base['oof'].true_model.to_numpy()
    predictions = {name:r['oof'].predicted_model.to_numpy() for name,r in runs.items()}
    complementary = complementarity(truth,predictions['word'],predictions['char'],predictions['combo'])
    bootstrap = prompt_bootstrap(truth,base['oof'].prompt_id.to_numpy(),predictions)
    stability = {}
    for name in ('word','char'):
        delta = np.array([r[name+'_minus_combo'] for r in fold_rows])
        stability[name] = {'folds_above_combo':int((delta>0).sum()),'folds_below_combo':int((delta<0).sum()),
                           'folds_tied_combo':int((delta==0).sum()),'paired_mean_fold_delta':float(delta.mean())}
    summary = {'run_ids':RUNS,'baseline_retrained':False,'n_train':len(truth),'n_prompts':int(base['oof'].prompt_id.nunique()),
        'input_sha256':{Path(name).name:value for name,value in base['meta']['input_sha256'].items()},
        'class_order':list(class_order),'primary_metric':'OOF balanced accuracy',
        'highest_oof_ba':max(RUNS,key=lambda n:runs[n]['metrics']['balanced_accuracy']),
        'overall':overall_rows,'stability':stability,'complementarity':complementary,'prompt_bootstrap':bootstrap,
        'default_baseline_replaced':False,'kaggle_submission':'not executed',
        'dimension_note':overall_rows[0]['dimension_origin'],
        'cost_note':'Same perf_counter boundaries: fold fit, predict, decision separately. Historical baseline and new runs occur at different times; no hardware-controlled speedup claim. RSS are snapshots; peak_wset is process-lifetime, not independent per-fold peaks.',
        'hypothesis_note':'Single-branch predictive signal and complementary correct sets can be measured. Scores alone do not isolate causal contributions of vocabulary, punctuation or suffixes; conclusions apply to these settings, including changed total vector norm when a branch is removed.'}
    output.mkdir(parents=True)
    tables={'overall.csv':pd.DataFrame(overall_rows),'fold-comparison.csv':pd.DataFrame(fold_rows),
            'class-comparison.csv':classes,'confusion-changes.csv':confusions,'resources.csv':pd.DataFrame(resource_rows),
            'feature-dimensions.csv':pd.DataFrame(dimension_rows)}
    for name, table in tables.items(): table.to_csv(output/name,index=False,lineterminator='\n')
    write_json(output/'analysis-summary.json',summary)
    write_json(output/'complementarity.json',complementary); write_json(output/'prompt-bootstrap.json',bootstrap)
    source_files={str(p.relative_to(ROOT)).replace('\\','/'):digest(p)
        for run in runs.values() for p in [run['directory']/'config.json',run['directory']/'metrics.json',run['directory']/'oof_predictions.parquet']}
    write_json(output/'input-artifact-sha256.json',source_files)
    compact=base['oof'][['id','prompt_id','fold','true_model']].copy()
    for name in RUNS: compact[name+'_prediction']=predictions[name]
    compact.to_parquet(ROOT/'artifacts/s07_01/aligned-oof.parquet',index=False)
    report = build_report(summary,tables)
    (output/'report.md').write_text(report,encoding='utf-8',newline='\n')
    report_path.parent.mkdir(parents=True,exist_ok=True); report_path.write_text(report,encoding='utf-8',newline='\n')
    return summary


def build_report(summary,tables):
    overall=tables['overall.csv']; folds=tables['fold-comparison.csv']; classes=tables['class-comparison.csv']; cm=tables['confusion-changes.csv']
    groups=pd.DataFrame([{'group':k,**v} for k,v in summary['complementarity']['groups'].items()])
    intervals=pd.DataFrame([{'comparison':k,**v} for k,v in summary['prompt_bootstrap']['intervals'].items()])
    important = {('gemini-3.7-flash','gemini-3.8-flash'),('gemini-3.8-flash','gemini-3.7-flash'),
                 ('gpt-5.6-sol','gpt-5.6-luna'),('gpt-5.6-luna','gpt-5.6-sol')}
    focused=cm[[tuple(x) in important for x in cm[['true_model','predicted_model']].itertuples(index=False,name=None)]]
    best=summary['highest_oof_ba']; comp=summary['complementarity']
    text=['# S07-01 Word/Char TF-IDF 消融实验报告','',
      '实验于 2026-11-18 启动，2026-11-19 汇总验收，跨日保留启动时分配的运行编号。固定官方 2699 条、150 prompt、18 类、folds_v1 和 seed=20261118；唯一变量为启用词、字符或两支。分类器和各支完整参数不变。组合结果复用 S06-B，没有再次训练。',
      '', '## 1 假设与控制变量','',
      'H1/H2：词/字符表示可能提供模型识别信号；H3：两支可能互补且组合可能优于单支。这些是研究假设，不能由分数直接证明词汇/标点/词缀的因果作用。每折新建 Pipeline，仅当前训练 text 拟合，验证只预测；id/prompt/测试不参与特征。',
      '', '## 2 总体指标','',markdown(overall,['experiment','run_id','balanced_accuracy','accuracy','macro_f1','mean_fold_balanced_accuracy','std_fold_balanced_accuracy_ddof0','ba_minus_combo','ba_minus_combo_pp']),
      '', '百分点=原始 BA 差值×100，不是相对增长率。主指标为整体 OOF BA，折均值另列。',
      '', '## 3 五折配对对照','',markdown(folds,['fold','n_train','n_valid','combo_ba','word_ba','char_ba','word_minus_combo','char_minus_combo']),
      '',json.dumps(summary['stability'],ensure_ascii=False),
      '', '## 4 全部类别召回与正确数量','',markdown(classes,['model','support','combo_correct','word_correct','char_correct','combo_recall','word_recall','char_recall']),
      '', '完整 Precision/Recall/F1 保存在仓库 experiments/s07_word_char_20261118_01/class-comparison.csv，正确数和 support 直接从 OOF 重算。',
      '', '### 特征偏好与组合影响','',
      'Word 相对 Char 召回差最大的类别：','',markdown(classes.sort_values('word_minus_char_recall',ascending=False).head(5),['model','word_correct','char_correct','word_minus_char_recall']),
      '', 'Char 相对 Word 优势最大的类别：','',markdown(classes.sort_values('word_minus_char_recall').head(5),['model','word_correct','char_correct','word_minus_char_recall']),
      '', '组合相对每类最佳单支的正确数增减（描述当前样本，不作显著性结论）：','',markdown(classes,['model','combo_minus_best_single_correct']),
      '', '## 5 主要混淆变化','',markdown(focused,['true_model','predicted_model','combo','word','char']),
      '', '原组合错误最多的前 12 个方向：','',markdown(cm.head(12),['true_model','predicted_model','combo','word','char']),
      '', 'Char-only 自身错误最多的前 8 个方向：','',markdown(cm.sort_values('char',ascending=False).head(8),['true_model','predicted_model','combo','word','char']),
      '', '## 6 预测互补性及组合利用','',markdown(groups),
      '',f'Word/Char 预测一致率 {comp["prediction_agreement_rate"]:.6f}（{comp["prediction_agreement_count"]}/{comp["n_samples"]}）；至少一个正确 {comp["at_least_one_correct_count"]} 条，占 {comp["at_least_one_correct_rate"]:.6f}。',
      '',f'两支恰好一个正确的 {comp["exclusive_complementary_count"]} 条中，组合识别对 {comp["combo_correct_exclusive_count"]} 条，利用率 {comp["combo_correct_exclusive_rate"]:.6f}。组合额外救回两支都错的 {comp["combo_rescued_both_wrong_count"]} 条，同时漏掉单支至少一个正确的 {comp["combo_missed_single_oracle_correct_count"]} 条。',
      '',f'事后 Oracle Accuracy={comp["oracle_accuracy"]:.6f}、BA={comp["oracle_balanced_accuracy"]:.6f}。Oracle 使用真值在两个固定预测中选择，仅供互补性参考，不是可部署模型、可保证的融合成绩，也不是组合特征分类器的绝对上界。本轮没有分数相加、校准或实际融合。',
      '', '## 7 Prompt 配对 Bootstrap','',markdown(intervals),
      '',f'整组 prompt 配对重采样 {summary["prompt_bootstrap"]["repeats"]} 次，固定种子 20261118，95% 百分位区间仅描述当前固定 OOF 差异的不确定性；不是逐行独立抽样。未重训模型，未覆盖 CV 训练集重叠依赖，不能当独立测试显著性结论。',
      '', '## 8 成本、维度与资源','',markdown(overall,['experiment','total_fit_seconds','total_predict_seconds','total_decision_seconds','total_inference_seconds','feature_dimensions_by_fold','warning_count']),
      '',markdown(tables['feature-dimensions.csv']),
      '', 'Word/Char 是实际拟合词表长度并与分类器系数宽度核对；未把 max_features 当实测值。旧组合未记录维度，其同折维度由两支实测长度相加推导：数据、训练 ID、TF-IDF 参数相同，原 FeatureUnion 独立拼接；这不是历史直接观测。',
      '',summary['cost_note'],
      '', '逐折实际资源（RSS 和峰值为字节）：','',markdown(tables['resources.csv'],['experiment','fold','actual_or_derived_features','fit_seconds','inference_seconds','rss_before_bytes','rss_after_bytes','process_peak_wset_bytes','warning_count']),
      '', '## 9 结论与下一轮','',
      f'当前最高 OOF BA 方案为 {best}。排名只针对本数据、分折和当前参数；正式 V1.0 默认组合保持不变，不自动替换或提交。',
      '', '两支存在的独占正确样本支持“互补预测集合”，但不保证默认拼接能利用全部互补信息。H3 中“组合优于两支”须依据真实总体/配对结果判断，不能由 Oracle 证明。',
      '', '本轮实测候选、假设判断与下一轮单因素建议见第 10 节，本轮不实施 S07-02。',
      '', '所有成功/失败日志保留。新公榜分数为空；0.311 只属于此前已提交的版本。原数据、v1/v2、历史实验和默认配置继续保护。S07-01 的分类器替换部分未实施，原完整任务不勾为全部完成。','']
    word_better=classes[classes.word_recall>classes.char_recall]
    combo_help=classes[classes.combo_minus_best_single_correct>0]
    weak=classes.assign(best_recall=classes[['combo_recall','word_recall','char_recall']].max(axis=1)).sort_values('best_recall').head(3)
    char_interval=summary['prompt_bootstrap']['intervals']['char_minus_combo']
    specific=['', '## 10 本轮数据支持的判断','',
      f'Word 的召回高于 Char 的类别共 {len(word_better)} 个；Char 高于 Word 共 {int((classes.char_recall>classes.word_recall).sum())} 个。前者具体计数如下，不能因 Word 总分较低而忽略其局部贡献：','',
      markdown(word_better,['model','support','word_correct','char_correct']),
      '',f'组合高于每类最佳单支的类别共 {len(combo_help)} 个，说明词分支在默认拼接下仍可能对部分标签有用：','',
      markdown(combo_help,['model','combo_correct','word_correct','char_correct','combo_minus_best_single_correct']),
      '', '三方案下均较困难的类别：','',markdown(weak,['model','support','combo_correct','word_correct','char_correct','best_recall']),
      '',f'Char−组合的当前 OOF BA 差值为 {char_interval["observed_delta"]:.6f}，prompt 配对 95% 区间 [{char_interval["percentile_95_lower"]:.6f}, {char_interval["percentile_95_upper"]:.6f}]。四折提高但第五折贡献了最大的提升，区间包含 0；不能据此直接替换正式方案或保证公榜收益。',
      '', 'H1/H2 得到“这两类表示分别包含可用分类信号”的经验支持，字符方案在本设置下更强；未隔离词汇/标点/词缀的因果作用。H3 的“独占正确样本互补”得到支持，而“默认组合优于最佳单支”在本轮不成立。',
      '', '保留 V1.0 组合作正式长期基线，Char-only 作为下一轮主要研究候选，Word-only 作为成本低的对照；均不自动换默认或提交。',
      '', 'S07-02 首选一个单因素实验：以本轮 Char-only 为对照，仅将 features.char.lowercase 改为 false，固定其余参数、seed 和 v1，先冒烟再正式五折。根据结果再决定是否单独做 char_wb 或词分支降权对照；不同时调整 n-gram、词表规模和 C。上述均未在本轮实施。','']
    reproduction=['', '## 11 复现与记录','',
      '实际训练命令如下；原运行目录已经存在，复跑必须换新编号，不覆盖历史：','', '```powershell',
      *[f'.\\.venv\\Scripts\\python.exe scripts/run_experiment.py --mode {mode} --run-id s07_{branch}_only_v1_'+('smoke_' if mode=='smoke' else '')+f'20261118_01 --config configs/s07_{branch}_only.json --purpose s07_word_char_ablation'+('' if mode=='smoke' else f' --smoke-run s07_{branch}_only_v1_smoke_20261118_01')
        for branch in ('word','char') for mode in ('smoke','formal')],
      '.\\.venv\\Scripts\\python.exe scripts/check_s07.py',
      '.\\.venv\\Scripts\\python.exe scripts/analyze_s07.py','```','',
      '原始文件与固定分折 SHA-256：','',markdown(pd.DataFrame([{'file':k,'sha256':v} for k,v in summary['input_sha256'].items()])),
      '', '完整 OOF、18 维分数、逐折审计/警告/资源、源码与配置在 artifacts/<run_id>/，轻量结果和比较表在 experiments。新报告和输出目录默认拒绝覆盖，已有报告需使用新路径重算。',
      '', '必要回归使用既有已保存模型和 OOF，不再次训练组合基线或全量候选；完整性、测试命令及日志见 L012/L013。','']
    return '\n'.join(text+specific+reproduction)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=ROOT/'experiments/s07_word_char_20261118_01')
    parser.add_argument('--report-path',type=Path,default=ROOT/'docs/S07-01_消融实验报告.md')
    args=parser.parse_args(); summary=analyze(args.output_dir,args.report_path)
    print('S07_ANALYSIS_COMPLETE',summary['highest_oof_ba'],summary['complementarity'])
