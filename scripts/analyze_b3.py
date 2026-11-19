"""B0-B3 saved-OOF comparison and descriptive style analysis. Never trains."""
from pathlib import Path
import json
import shutil

import numpy as np
import pandas as pd

from ntl.data import read_input_csv
from ntl.evaluation import load_masks, strict_metrics, SPARSE_SEEDS
from ntl.io import digest, write_json
from ntl.labels import class_order
from ntl.models import baseline_config, make_model, model_config
from ntl.style import rich_style_matrix, STYLE_FEATURE_NAMES, _TOKEN
from ntl.validation import validate_run
from analyze_s07 import load_runs, align_oof, markdown, prompt_bootstrap

ROOT=Path(__file__).resolve().parents[1]
RUN='s07b_b3_rich_style_v1_20261119_01'
OUT=ROOT/'experiments'/RUN


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compare_errors(truth, old, new):
    truth,old,new=map(np.asarray,(truth,old,new))
    if not len(truth) or len(truth)!=len(old) or len(old)!=len(new):
        raise ValueError('Error comparison needs equal nonempty arrays')
    a,b=old==truth,new==truth
    family=lambda values:np.asarray([v.split('-')[0] for v in values])
    return {'n':len(truth),'both_correct':int((a&b).sum()),'b0_only_correct_new_errors':int((a&~b).sum()),
        'b3_only_correct_corrected':int((~a&b).sum()),'both_wrong':int((~a&~b).sum()),
        'changed_predictions':int((old!=new).sum()),
        'same_provider_errors_b0':int((~a & (family(truth)==family(old))).sum()),
        'same_provider_errors_b3':int((~b & (family(truth)==family(new))).sum()),
        'cross_provider_errors_b0':int((~a & (family(truth)!=family(old))).sum()),
        'cross_provider_errors_b3':int((~b & (family(truth)!=family(new))).sum())}


def describe_style(train):
    raw=rich_style_matrix(train.text).astype(np.float64)
    chars=train.text.str.len().to_numpy(dtype=float)
    words=np.asarray([len(_TOKEN.findall(t.lower())) for t in train.text],dtype=float)
    def eta2(groups):
        total=((raw-raw.mean(axis=0))**2).sum(axis=0)
        between=np.zeros(raw.shape[1])
        for group in sorted(set(groups)):
            one=raw[np.asarray(groups)==group]
            between+=len(one)*(one.mean(axis=0)-raw.mean(axis=0))**2
        return np.divide(between,total,out=np.zeros_like(total),where=total>0)
    by_prompt,by_model=eta2(train.prompt_id),eta2(train.model)
    rows=[]
    for col,name in enumerate(STYLE_FEATURE_NAMES):
        values=raw[:,col]; q1,median,q3,p99=np.quantile(values,[.25,.5,.75,.99]); iqr=q3-q1
        corr=lambda other:float(np.corrcoef(values,other)[0,1]) if values.std()>0 and other.std()>0 else None
        rows.append({'index':col,'feature':name,'mean':float(values.mean()),'std_ddof0':float(values.std()),
            'min':float(values.min()),'q25':q1,'median':median,'q75':q3,'p99':p99,'max':float(values.max()),
            'n_unique':len(np.unique(values)),'zero_fraction':float((values==0).mean()),
            'constant':bool(values.std()==0),'near_zero_ge99pct':bool((values==0).mean()>=.99),
            'outside_1_5_iqr':int(((values<q1-1.5*iqr)|(values>q3+1.5*iqr)).sum()),
            'pearson_char_length':corr(chars),'pearson_word_count':corr(words),
            'eta2_prompt_descriptive':float(by_prompt[col]),'eta2_model_descriptive':float(by_model[col])})
    class_rows=[]
    for label in class_order:
        one=raw[train.model.eq(label)]
        for col,name in enumerate(STYLE_FEATURE_NAMES):
            class_rows.append({'label':label,'feature':name,'support':len(one),'mean':float(one[:,col].mean()),
                'std_ddof0':float(one[:,col].std()),'zero_fraction':float((one[:,col]==0).mean())})
    return pd.DataFrame(rows),pd.DataFrame(class_rows)


def analyze():
    if OUT.exists():raise FileExistsError(f'Refusing to overwrite B3 analysis: {OUT}')
    runs=load_runs(); folder=ROOT/'artifacts'/RUN;validate_run(folder)
    runs['b3']={'directory':folder,'config':read(folder/'config.json'),'meta':read(folder/'run_metadata.json'),
        'metrics':read(folder/'metrics.json'),'oof':pd.read_parquet(folder/'oof_predictions.parquet'),
        'classes':pd.read_csv(folder/'classification_report.csv',index_col=0),
        'folds':[read(folder/f'fold_{i}/metrics.json') for i in range(1,6)]}
    b3=runs['b3'];base=runs['combo']
    assert b3['config']['model_parameters']==model_config(make_model(read(ROOT/'configs/s07b_b3_rich_style.json')))
    assert b3['meta']['input_sha256']==base['meta']['input_sha256']
    assert b3['config']['class_order']==base['config']['class_order']==list(class_order)
    for i in range(1,6):
        assert read(folder/f'fold_{i}/split_audit.json')==read(base['directory']/f'fold_{i}/split_audit.json')
    frames=align_oof({name:r['oof'] for name,r in runs.items()})
    train_path=Path(base['meta']['data_dir'])/'train.csv';train=read_input_csv(train_path,'train')
    ref=frames['combo'][['id','prompt_id','fold']]
    masks,mask_meta=load_masks(ref,train_path,ROOT/'experiments/folds/folds_v1.csv',ROOT/'experiments/evaluation_masks/sparse_k2_v1')
    full_rows=[];sparse_rows=[];recall_rows=[];fold_rows=[]
    for name,run in runs.items():
        metrics=run['metrics'];m=strict_metrics(frames[name].true_model,frames[name].predicted_model)
        for k in ['balanced_accuracy','accuracy','macro_f1']:np.testing.assert_allclose(m[k],metrics[k],rtol=0,atol=1e-15)
        full_rows.append({'model':name,'run_id':run['meta']['run_id'],**metrics,
            'ba_minus_b0_pp':100*(metrics['balanced_accuracy']-base['metrics']['balanced_accuracy']),
            'peak_process_mib':max(f.get('process_peak_wset_bytes') or 0 for f in run['folds'])/1024**2})
        lookup=frames[name].set_index('id')
        for seed in SPARSE_SEEDS:
            ids=masks.loc[masks.seed==seed,'id'];one=lookup.loc[ids]
            scored=strict_metrics(one.true_model,one.predicted_model)
            sparse_rows.append({'model':name,'seed':seed,'mask_sha256':mask_meta['mask_sha256'],
                **{k:scored[k] for k in ['balanced_accuracy','accuracy','macro_f1','n_samples','n_classes','ba_status']}})
        for label in class_order:
            row={'model':name,'label':label,'support':m['support'][label],'recall':m['recall'][label]}
            row.update({k:float(run['classes'].loc[label,k]) for k in ['precision','f1-score']})
            row['correct_count']=int(((frames[name].true_model==label)&(frames[name].predicted_model==label)).sum())
            recall_rows.append(row)
        for i,f in enumerate(run['folds'],1):
            fold_rows.append({'model':name,'fold':i,**{k:f[k] for k in ['n_train','n_valid','balanced_accuracy','accuracy','macro_f1','fit_seconds','predict_seconds','decision_seconds']}})
    full=pd.DataFrame(full_rows);sparse=pd.DataFrame(sparse_rows);recalls=pd.DataFrame(recall_rows)
    # Verify pre-existing models still receive exactly their original frozen-mask scores.
    old_sparse=pd.read_csv(ROOT/'experiments/s07a_protocol_20261119_01/sparse_metrics.csv')
    check=sparse[sparse.model!='b3'].merge(old_sparse,on=['model','seed'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(check)==30
    for k in ['balanced_accuracy','accuracy','macro_f1']:np.testing.assert_allclose(check[k+'_new'],check[k+'_old'],rtol=0,atol=1e-15)
    summaries=[]
    for name,one in sparse.groupby('model',sort=False):
        row={'model':name}
        for metric in ['balanced_accuracy','accuracy','macro_f1']:
            values=one[metric].to_numpy();assert np.isfinite(values).all()
            row.update({metric+'_mean':float(values.mean()),metric+'_std_ddof0':float(values.std(ddof=0)),
                        metric+'_min':float(values.min()),metric+'_max':float(values.max())})
        summaries.append(row)
    sparse_summary=pd.DataFrame(summaries)
    wide=sparse.pivot(index='seed',columns='model',values='balanced_accuracy')
    rank_rows=[{'seed':int(seed),'ranks':{k:float(v) for k,v in row.rank(ascending=False,method='min').items()},
                'descending_ba':row.sort_values(ascending=False).index.tolist()} for seed,row in wide.iterrows()]
    paired=[]
    for other in ['combo','word','char']:
        d=wide.b3-wide[other]
        paired.append({'comparison':'b3_minus_'+other,'mean':float(d.mean()),'std_ddof0':float(d.std(ddof=0)),
            'min':float(d.min()),'max':float(d.max()),'wins':int((d>0).sum()),'ties':int((d==0).sum()),'losses':int((d<0).sum())})
    errors=compare_errors(frames['combo'].true_model,frames['combo'].predicted_model,frames['b3'].predicted_model)
    trans=ref.copy();trans['true_model']=frames['combo'].true_model;trans['b0_prediction']=frames['combo'].predicted_model;trans['b3_prediction']=frames['b3'].predicted_model
    a=trans.b0_prediction==trans.true_model;b=trans.b3_prediction==trans.true_model
    trans['transition']=np.select([a&b,a&~b,~a&b],['both_correct','new_error','corrected'],default='both_wrong')
    changes=[]
    for truth in class_order:
        for pred in class_order:
            if truth==pred:continue
            old=int(((frames['combo'].true_model==truth)&(frames['combo'].predicted_model==pred)).sum())
            new=int(((frames['b3'].true_model==truth)&(frames['b3'].predicted_model==pred)).sum())
            changes.append({'true_model':truth,'predicted_model':pred,'b0_errors':old,'b3_errors':new,'change':new-old})
    classwide=recalls.pivot(index='label',columns='model',values='recall').reindex(class_order).reset_index()
    classwide['b3_minus_b0_pp']=100*(classwide.b3-classwide.combo)
    classwide['b3_minus_b2_pp']=100*(classwide.b3-classwide.char)
    # Reuse the existing prompt bootstrap without changing historical analysis source.
    raw_boot=prompt_bootstrap(frames['combo'].true_model,frames['combo'].prompt_id,
        {'combo':frames['combo'].predicted_model,'word':frames['b3'].predicted_model,'char':frames['char'].predicted_model})
    reverse=raw_boot['intervals']['char_minus_word']
    bootstrap={k:v for k,v in raw_boot.items() if k!='intervals'}
    bootstrap['intervals']={'b3_minus_b0':raw_boot['intervals']['word_minus_combo'],
        'b3_minus_b2':{'observed_delta':-reverse['observed_delta'],'percentile_95_lower':-reverse['percentile_95_upper'],
                      'percentile_95_upper':-reverse['percentile_95_lower']}}
    descriptive,class_style=describe_style(train)
    OUT.mkdir()
    tables={'overall':full,'fold_comparison':pd.DataFrame(fold_rows),'sparse_metrics':sparse,'sparse_summary':sparse_summary,
        'class_report':recalls,'class_recall_comparison':classwide,'error_transition_ids':trans,
        'confusion_changes':pd.DataFrame(changes),'style_descriptive':descriptive,'style_by_class':class_style,
        'sparse_paired':pd.DataFrame(paired)}
    for name,frame in tables.items():frame.to_csv(OUT/f'{name}.csv',index=False,lineterminator='\n')
    write_json(OUT/'rankings.json',rank_rows);write_json(OUT/'error_comparison.json',errors);write_json(OUT/'prompt_bootstrap.json',bootstrap)
    write_json(OUT/'comparison_provenance.json',{'run_ids':{name:r['meta']['run_id'] for name,r in runs.items()},
        'oof_sha256':{name:digest(r['directory']/'oof_predictions.parquet') for name,r in runs.items()},
        'mask_sha256':mask_meta['mask_sha256'],'mask_metadata_sha256':digest(ROOT/'experiments/evaluation_masks/sparse_k2_v1/metadata.json'),
        'new_model_fits_during_analysis':0,'reference_definition':'../s07b_b3_definition_v1/reference_fixture.json',
        'claim_boundary':'B3-B0 isolates added rich-style branch; B3-B2 also differs in word branch. Descriptive statistics not used to select/delete features.'})
    for filename in ['config.json','run_metadata.json','metrics.json','fold_metrics.csv','classification_report.csv',
                     'class_recall.csv','confusion_matrix.csv','confusion_matrix_normalized.csv','warnings.json']:
        shutil.copyfile(folder/filename,OUT/filename)
    for i in range(1,6):
        target=OUT/f'fold_{i}';target.mkdir();shutil.copyfile(folder/f'fold_{i}/style_audit.json',target/'style_audit.json')
    validation_source=OUT/'validation-source';validation_source.mkdir()
    for path in [Path(__file__),ROOT/'scripts/check_b3.py']:
        shutil.copyfile(path,validation_source/path.name)
    write_json(OUT/'validation-source.json',{'timing':'analysis/acceptance source archived after fitting, not presented as training-start source',
        'sha256':{p.name:digest(p) for p in validation_source.iterdir()}})
    write_json(OUT/'validation-summary.json',{'status':'passed','run_id':RUN,'oof_rows':2699,'score_dimensions':18,
        'frozen_seeds':list(SPARSE_SEEDS),'same_masks_as_s07a':True,'full_metrics_independently_recomputed':True,
        'historical_three_sparse_scores_identical':True,'default_model_unchanged':True,'new_formal_fold_fits':5,'smoke_fold_fits':1})
    print(full[['model','balanced_accuracy','accuracy','macro_f1','total_fit_seconds','total_inference_seconds']].to_string(index=False))
    print('Sparse:',sparse_summary[['model','balanced_accuracy_mean','balanced_accuracy_std_ddof0','balanced_accuracy_min','balanced_accuracy_max']].to_string(index=False))
    print('Errors:',errors);print('Bootstrap:',bootstrap)
    print('Style constants:',descriptive.loc[descriptive.constant,'feature'].tolist())
    print('Near-zero:',descriptive.loc[descriptive.near_zero_ge99pct,'feature'].tolist())
    print('Validated analysis:',OUT)


if __name__=='__main__':
    from unittest.mock import patch
    from sklearn.svm import LinearSVC
    with patch.object(LinearSVC,'fit',side_effect=AssertionError('Analysis cannot fit a classifier')):
        analyze()
