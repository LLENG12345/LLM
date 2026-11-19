"""B3-C OOF-only factorial/paired analysis; no training or mask generation."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from ntl.evaluation import load_masks,strict_metrics,SPARSE_SEEDS
from ntl.io import digest,write_json
from ntl.labels import class_order
from ntl.validation import validate_run
from analyze_s07 import load_runs,align_oof,prompt_bootstrap,markdown
from analyze_b3 import compare_errors

ROOT=Path(__file__).resolve().parents[1]
RUN='s07b_b3c_char_style_v1_20261119_01'
STAGE=ROOT/'artifacts/s07b_b3c_20261119_01'
OUT=ROOT/'experiments'/RUN
read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))


def factor_contrasts(scores):
    a=scores['combo']-scores['char'];b=scores['b3']-scores['b3c']
    return {'A_word_without_style':float(a),'B_word_with_style':float(b),
        'C_style_with_word':float(scores['b3']-scores['combo']),
        'D_style_without_word':float(scores['b3c']-scores['char']),
        'interaction_B_minus_A':float(b-a)}


def load_all_runs():
    runs=load_runs()
    for name,identifier in [('b3','s07b_b3_rich_style_v1_20261119_01'),('b3c',RUN)]:
        p=ROOT/'artifacts'/identifier;validate_run(p)
        runs[name]={'directory':p,'config':read(p/'config.json'),'meta':read(p/'run_metadata.json'),
            'metrics':read(p/'metrics.json'),'oof':pd.read_parquet(p/'oof_predictions.parquet'),
            'classes':pd.read_csv(p/'classification_report.csv',index_col=0),
            'folds':[read(p/f'fold_{i}/metrics.json') for i in range(1,6)]}
    frames=align_oof({name:r['oof'] for name,r in runs.items()})
    for name,frame in frames.items():runs[name]['oof']=frame
    for name,r in runs.items():
        assert r['meta']['input_sha256']==runs['combo']['meta']['input_sha256']
        assert r['config']['class_order']==list(class_order)
    return runs


def analyze():
    if OUT.exists():raise FileExistsError(f'Existing analysis not overwritten: {OUT}')
    plan=read(STAGE/'analysis-plan.json');runs=load_all_runs();base=runs['combo']['oof']
    masks,meta=load_masks(base[['id','prompt_id','fold']],Path(runs['combo']['meta']['data_dir'])/'train.csv',
        ROOT/'experiments/folds/folds_v1.csv',ROOT/'experiments/evaluation_masks/sparse_k2_v1')
    full=[];folds=[];recalls=[];sparse=[];sparse_recall=[];dimensions=[]
    for name,r in runs.items():
        oof=r['oof'];m=r['metrics'];scored=strict_metrics(oof.true_model,oof.predicted_model)
        for key in ['balanced_accuracy','accuracy','macro_f1']:np.testing.assert_allclose(scored[key],m[key],rtol=0,atol=1e-15)
        full.append({'model':name,'run_id':r['meta']['run_id'],**m,
                     'peak_process_mib':max(f.get('process_peak_wset_bytes') or 0 for f in r['folds'])/1024**2})
        for label in class_order:
            cls=r['classes'].loc[label]
            recalls.append({'model':name,'label':label,'precision':float(cls['precision']),'recall':float(cls['recall']),
                'f1':float(cls['f1-score']),'support':int(cls['support']),
                'correct':int(((oof.true_model==label)&(oof.predicted_model==label)).sum())})
        for f in r['folds']:
            folds.append({'model':name,**{k:f[k] for k in ['fold','balanced_accuracy','accuracy','macro_f1','fit_seconds','predict_seconds','decision_seconds','n_train','n_valid']}})
        lookup=oof.set_index('id')
        for seed in SPARSE_SEEDS:
            subset=lookup.loc[masks.loc[masks.seed==seed,'id']];s=strict_metrics(subset.true_model,subset.predicted_model)
            sparse.append({'model':name,'seed':seed,'mask_sha256':meta['mask_sha256'],**{k:s[k] for k in ['balanced_accuracy','accuracy','macro_f1','n_samples','n_classes']}})
            for label in class_order:sparse_recall.append({'model':name,'seed':seed,'label':label,'support':s['support'][label],'recall':s['recall'][label]})
    for i in range(1,6):
        f=runs['b3c']['folds'][i-1];old=runs['b3']['folds'][i-1]
        assert f['feature_dimensions']=={'char':old['feature_dimensions']['char'],'style':162}
        assert read(runs['b3c']['directory']/f'fold_{i}/split_audit.json')==read(runs['b3']['directory']/f'fold_{i}/split_audit.json')
        assert read(runs['b3c']['directory']/f'fold_{i}/style_audit.json')==read(runs['b3']['directory']/f'fold_{i}/style_audit.json')
        dimensions.append({'fold':i,'char':f['feature_dimensions']['char'],'style':162,'b3c_actual':f['n_features'],
            'b3_actual':old['n_features'],'removed_word_actual':old['feature_dimensions']['word']})
    full=pd.DataFrame(full);folds=pd.DataFrame(folds);recalls=pd.DataFrame(recalls);sparse=pd.DataFrame(sparse)
    prior=pd.read_csv(ROOT/'experiments/s07b_b3_rich_style_v1_20261119_01/sparse_metrics.csv')
    match=sparse[sparse.model!='b3c'].merge(prior,on=['model','seed'],suffixes=('_current','_prior'),validate='one_to_one');assert len(match)==40
    for key in ['balanced_accuracy','accuracy','macro_f1']:np.testing.assert_allclose(match[key+'_current'],match[key+'_prior'],rtol=0,atol=1e-15)
    wide=sparse.pivot(index='seed',columns='model',values='balanced_accuracy');fw=folds.pivot(index='fold',columns='model',values='balanced_accuracy')
    fw['b3c_minus_b3']=fw.b3c-fw.b3;fw['b3c_minus_b3_pp']=100*fw.b3c_minus_b3
    factor_rows=[]
    for subset,values in [('full',full.set_index('model').balanced_accuracy.to_dict()),*[(int(seed),row.to_dict()) for seed,row in wide.iterrows()]]:
        for key,value in factor_contrasts(values).items():factor_rows.append({'evaluation':subset,'contrast':key,'delta':value,'delta_pp':100*value})
    summaries=[]
    for name,one in sparse.groupby('model',sort=False):
        row={'model':name}
        for metric in ['balanced_accuracy','accuracy','macro_f1']:
            v=one[metric].to_numpy();assert np.isfinite(v).all()
            row.update({metric+'_mean':float(v.mean()),metric+'_std_ddof0':float(v.std(ddof=0)),metric+'_min':float(v.min()),metric+'_max':float(v.max())})
        summaries.append(row)
    d=wide.b3c-wide.b3
    paired={'mean':float(d.mean()),'std_ddof0':float(d.std(ddof=0)),'min':float(d.min()),'max':float(d.max()),
        'wins':int((d>0).sum()),'ties':int((d==0).sum()),'losses':int((d<0).sum())}
    old,new=runs['b3']['oof'],runs['b3c']['oof'];e=compare_errors(old.true_model,old.predicted_model,new.predicted_model)
    errors={'both_correct':e['both_correct'],'both_wrong':e['both_wrong'],'b3c_corrected':e['b3_only_correct_corrected'],
        'b3c_new_errors':e['b0_only_correct_new_errors'],'net_correct_gain':e['b3_only_correct_corrected']-e['b0_only_correct_new_errors'],
        'prediction_agreement':len(old)-e['changed_predictions'],'prediction_agreement_rate':1-e['changed_predictions']/len(old),
        'same_provider_errors_b3':e['same_provider_errors_b0'],'same_provider_errors_b3c':e['same_provider_errors_b3'],
        'cross_provider_errors_b3':e['cross_provider_errors_b0'],'cross_provider_errors_b3c':e['cross_provider_errors_b3']}
    transitions=base[['id','prompt_id','fold','true_model']].copy();transitions['b3_prediction']=old.predicted_model;transitions['b3c_prediction']=new.predicted_model
    a=old.predicted_model==old.true_model;b=new.predicted_model==new.true_model
    transitions['transition']=np.select([a&b,a&~b,~a&b],['both_correct','b3c_new_error','b3c_corrected'],default='both_wrong')
    conf=[]
    for truth in class_order:
        for pred in class_order:
            if truth==pred:continue
            before=int(((old.true_model==truth)&(old.predicted_model==pred)).sum());after=int(((new.true_model==truth)&(new.predicted_model==pred)).sum())
            conf.append({'true_model':truth,'predicted_model':pred,'b3_errors':before,'b3c_errors':after,'change':after-before})
    # Existing helper with explicit aliases; no resampling policy selected by score.
    boot=prompt_bootstrap(old.true_model,old.prompt_id,{'combo':base.predicted_model,'word':new.predicted_model,'char':old.predicted_model},
        repeats=plan['bootstrap_repeats'],seed=plan['bootstrap_seed'])
    reverse=boot['intervals']['char_minus_word']
    bootstrap={k:v for k,v in boot.items() if k!='intervals'}
    bootstrap['primary_comparison']='B3-C minus B3'
    bootstrap['interval']={'observed_delta':-reverse['observed_delta'],'percentile_95_lower':-reverse['percentile_95_upper'],
                           'percentile_95_upper':-reverse['percentile_95_lower']}
    OUT.mkdir()
    for name,frame in {'overall':full,'fold_metrics_comparison':folds,'fold_paired':fw.reset_index(),'class_report':recalls,
        'sparse_metrics':sparse,'sparse_class_recall':pd.DataFrame(sparse_recall),'sparse_summary':pd.DataFrame(summaries),
        'factor_contrasts':pd.DataFrame(factor_rows),'error_transition_ids':transitions,'confusion_changes':pd.DataFrame(conf),
        'feature_dimensions':pd.DataFrame(dimensions)}.items():frame.to_csv(OUT/f'{name}.csv',index=False,lineterminator='\n')
    for name,value in {'paired_errors':errors,'sparse_paired':paired,'prompt_bootstrap':bootstrap,
        'rankings':[{'seed':int(seed),'descending_ba':row.sort_values(ascending=False).index.tolist(),'ranks':row.rank(ascending=False,method='min').to_dict()} for seed,row in wide.iterrows()],
        'analysis_plan':plan,'comparison_provenance':{'runs':{n:r['meta']['run_id'] for n,r in runs.items()},
            'oof_sha256':{n:digest(r['directory']/'oof_predictions.parquet') for n,r in runs.items()},'mask_sha256':meta['mask_sha256'],
            'mask_metadata_sha256':digest(ROOT/'experiments/evaluation_masks/sparse_k2_v1/metadata.json'),
            'old_40_sparse_scores_unchanged':True,'new_model_fits_during_analysis':0,'primary_protocol':'full_group_cv',
            'interpretation':'Factorial contrasts describe this fixed configuration and OOF only; no universal causal claim'}}.items():write_json(OUT/f'{name}.json',value)
    for filename in ['config.json','run_metadata.json','metrics.json','fold_metrics.csv','classification_report.csv','class_recall.csv',
                     'confusion_matrix.csv','confusion_matrix_normalized.csv','warnings.json']:
        shutil.copyfile(runs['b3c']['directory']/filename,OUT/filename)
    for i in range(1,6):
        p=OUT/f'fold_{i}';p.mkdir();shutil.copyfile(runs['b3c']['directory']/f'fold_{i}/style_audit.json',p/'style_audit.json')
    print(full[['model','balanced_accuracy','accuracy','macro_f1','total_fit_seconds','total_inference_seconds','peak_process_mib']].to_string(index=False))
    print(pd.DataFrame(summaries)[['model','balanced_accuracy_mean','balanced_accuracy_std_ddof0','balanced_accuracy_min','balanced_accuracy_max']].to_string(index=False))
    print('Factors:',factor_contrasts(full.set_index('model').balanced_accuracy.to_dict()))
    print('Paired errors:',errors);print('Sparse paired:',paired);print('Bootstrap:',bootstrap)


if __name__=='__main__':
    from unittest.mock import patch
    from sklearn.svm import LinearSVC
    with patch.object(LinearSVC,'fit',side_effect=AssertionError('OOF analysis cannot train classifiers')):analyze()
