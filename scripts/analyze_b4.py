"""Read saved B4 OOF and fold vocabularies; B2 primary, B3-C secondary."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from ntl.io import digest,write_json
from ntl.validation import validate_run
from ntl.labels import class_order
from ntl.evaluation import strict_metrics,load_masks,SPARSE_SEEDS
from ntl.morphology import morphology_batch
from ntl.data import read_input_csv
from analyze_b3c import load_all_runs
from analyze_s07 import align_oof,prompt_bootstrap,markdown
from analyze_b3 import compare_errors

ROOT=Path(__file__).resolve().parents[1];RUN='s07b_b4_char_morphology_v1_20261119_01'
STAGE=ROOT/'artifacts/s07b_b4_20261119_01';OUT=ROOT/'experiments'/RUN
read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))


def pair_errors(truth,reference,predicted):
    raw=compare_errors(truth,reference,predicted)
    return {'both_correct':raw['both_correct'],'both_wrong':raw['both_wrong'],
        'b4_corrected':raw['b3_only_correct_corrected'],'b4_new_errors':raw['b0_only_correct_new_errors'],
        'net_correct_gain':raw['b3_only_correct_corrected']-raw['b0_only_correct_new_errors'],
        'agreement_count':len(truth)-raw['changed_predictions'],'agreement_rate':1-raw['changed_predictions']/len(truth),
        'same_provider_errors_reference':raw['same_provider_errors_b0'],'same_provider_errors_b4':raw['same_provider_errors_b3'],
        'cross_provider_errors_reference':raw['cross_provider_errors_b0'],'cross_provider_errors_b4':raw['cross_provider_errors_b3']}


def analyze():
    if OUT.exists():raise FileExistsError(f'Analysis already exists: {OUT}')
    runs=load_all_runs();folder=ROOT/'artifacts'/RUN;validate_run(folder)
    runs['b4']={'directory':folder,'config':read(folder/'config.json'),'meta':read(folder/'run_metadata.json'),
        'metrics':read(folder/'metrics.json'),'oof':pd.read_parquet(folder/'oof_predictions.parquet'),
        'classes':pd.read_csv(folder/'classification_report.csv',index_col=0),
        'folds':[read(folder/f'fold_{i}/metrics.json') for i in range(1,6)]}
    frames=align_oof({n:r['oof'] for n,r in runs.items()})
    for n,r in runs.items():
        r['oof']=frames[n]
        assert r['meta']['input_sha256']==runs['char']['meta']['input_sha256']
        assert r['config']['class_order']==list(class_order)
    masks,mask_meta=load_masks(frames['char'][['id','prompt_id','fold']],Path(runs['char']['meta']['data_dir'])/'train.csv',
        ROOT/'experiments/folds/folds_v1.csv',ROOT/'experiments/evaluation_masks/sparse_k2_v1')
    overall=[];folds=[];classes=[];sparse=[]
    for name,r in runs.items():
        frame=r['oof'];metrics=r['metrics'];m=strict_metrics(frame.true_model,frame.predicted_model)
        for k in ['balanced_accuracy','accuracy','macro_f1']:np.testing.assert_allclose(m[k],metrics[k],rtol=0,atol=1e-15)
        overall.append({'model':name,'run_id':r['meta']['run_id'],**metrics,
            'peak_process_mib':max(x.get('process_peak_wset_bytes') or 0 for x in r['folds'])/1024**2})
        for f in r['folds']:folds.append({'model':name,**{k:f[k] for k in ['fold','balanced_accuracy','accuracy','macro_f1','fit_seconds','predict_seconds','decision_seconds','n_train','n_valid']}})
        for label in class_order:
            cls=r['classes'].loc[label]
            classes.append({'model':name,'label':label,'precision':float(cls['precision']),'recall':float(cls['recall']),
                'f1':float(cls['f1-score']),'support':int(cls['support']),
                'correct':int(((frame.true_model==label)&(frame.predicted_model==label)).sum())})
        lookup=frame.set_index('id')
        for seed in SPARSE_SEEDS:
            selected=lookup.loc[masks.loc[masks.seed==seed,'id']];s=strict_metrics(selected.true_model,selected.predicted_model)
            sparse.append({'model':name,'seed':seed,'mask_sha256':mask_meta['mask_sha256'],
                **{k:s[k] for k in ['balanced_accuracy','accuracy','macro_f1','n_samples','n_classes']}})
    overall=pd.DataFrame(overall);folds=pd.DataFrame(folds);classes=pd.DataFrame(classes);sparse=pd.DataFrame(sparse)
    previous=pd.read_csv(ROOT/'experiments/s07b_b3c_char_style_v1_20261119_01/sparse_metrics.csv')
    match=sparse[sparse.model!='b4'].merge(previous,on=['model','seed'],validate='one_to_one',suffixes=('_current','_old'));assert len(match)==50
    for k in ['balanced_accuracy','accuracy','macro_f1']:np.testing.assert_allclose(match[k+'_current'],match[k+'_old'],rtol=0,atol=1e-15)
    summary=[]
    for name,one in sparse.groupby('model',sort=False):
        row={'model':name}
        for k in ['balanced_accuracy','accuracy','macro_f1']:
            v=one[k].to_numpy();assert np.isfinite(v).all()
            row.update({k+'_mean':float(v.mean()),k+'_std_ddof0':float(v.std(ddof=0)),k+'_min':float(v.min()),k+'_max':float(v.max())})
        summary.append(row)
    wide=sparse.pivot(index='seed',columns='model',values='balanced_accuracy');paired=[]
    for ref in ['char','b3c']:
        d=wide.b4-wide[ref];paired.append({'reference':ref,'mean':float(d.mean()),'std_ddof0':float(d.std(ddof=0)),
            'min':float(d.min()),'max':float(d.max()),'wins':int((d>1e-12).sum()),'ties':int((d.abs()<=1e-12).sum()),'losses':int((d< -1e-12).sum())})
    fw=folds.pivot(index='fold',columns='model',values='balanced_accuracy');fw['b4_minus_b2']=fw.b4-fw.char;fw['b4_minus_b3c']=fw.b4-fw.b3c
    point={}
    for reference in ['char','b3c']:
        for metric in ['balanced_accuracy','accuracy','macro_f1']:
            values=overall.set_index('model')[metric];point[f'{metric}_b4_minus_{reference}']={'delta':float(values.b4-values[reference]),'delta_pp':float(100*(values.b4-values[reference]))}
    truth=frames['b4'].true_model;new=frames['b4'].predicted_model
    primary=pair_errors(truth,frames['char'].predicted_model,new);secondary=pair_errors(truth,frames['b3c'].predicted_model,new)
    a=new.eq(truth);b=frames['b3c'].predicted_model.eq(truth)
    oracle=np.where(a,new,np.where(b,frames['b3c'].predicted_model,new))
    oracle_metrics=strict_metrics(truth,oracle)
    complement={'pair':'B4 and B3-C','both_correct':int((a&b).sum()),'b4_only_correct':int((a&~b).sum()),
        'b3c_only_correct':int((~a&b).sum()),'both_wrong':int((~a&~b).sum()),
        'oracle_accuracy':oracle_metrics['accuracy'],'oracle_balanced_accuracy':oracle_metrics['balanced_accuracy'],
        'warning':'Truth-assisted ideal selector over TWO fixed predictions; diagnostic only, not an achieved fusion or guaranteed score'}
    transitions=frames['b4'][['id','prompt_id','fold','true_model']].copy()
    transitions['b2_prediction']=frames['char'].predicted_model;transitions['b3c_prediction']=frames['b3c'].predicted_model;transitions['b4_prediction']=new
    old_correct=frames['char'].predicted_model.eq(truth)
    transitions['primary_transition']=np.select([old_correct&a,old_correct&~a,~old_correct&a],['both_correct','b4_new_error','b4_corrected'],default='both_wrong')
    changes=[]
    for ref in ['char','b3c']:
        for label in class_order:
            for prediction in class_order:
                if label==prediction:continue
                before=int((truth.eq(label)&frames[ref].predicted_model.eq(prediction)).sum());after=int((truth.eq(label)&new.eq(prediction)).sum())
                changes.append({'reference':ref,'true_model':label,'predicted_model':prediction,'reference_errors':before,'b4_errors':after,'change':after-before})
    plan=read(STAGE/'analysis-plan.json')
    boot=prompt_bootstrap(truth,frames['b4'].prompt_id,{'combo':frames['char'].predicted_model,'word':new,'char':frames['b3c'].predicted_model},repeats=plan['bootstrap_repeats'],seed=plan['bootstrap_seed'])
    reverse=boot['intervals']['char_minus_word'];bootstrap={k:v for k,v in boot.items() if k!='intervals'}
    bootstrap['intervals']={'b4_minus_b2':boot['intervals']['word_minus_combo'],'b4_minus_b3c':{
        'observed_delta':-reverse['observed_delta'],'percentile_95_lower':-reverse['percentile_95_upper'],'percentile_95_upper':-reverse['percentile_95_lower']}}
    dimensions=[];grams=[]
    for i in range(1,6):
        audit=read(folder/f'fold_{i}/morphology_audit.json');metric=runs['b4']['folds'][i-1]
        assert metric['feature_dimensions']=={'char':runs['char']['folds'][i-1]['n_features'],'morphology':audit['vocabulary_size']}
        assert read(folder/f'fold_{i}/split_audit.json')==read(runs['char']['directory']/f'fold_{i}/split_audit.json')
        vocab=pd.read_csv(folder/f'fold_{i}/morphology_vocabulary.csv',keep_default_na=False)
        dfcol='document_frequency_from_smoothed_idf'
        for kind,part in [('high_document_frequency',vocab.sort_values([dfcol,'ngram'],ascending=[False,True]).head(20)),
                          ('low_document_frequency',vocab.sort_values([dfcol,'ngram'],ascending=[True,True]).head(20))]:
            for row in part.to_dict('records'):grams.append({'fold':i,'kind':kind,**row})
        dimensions.append({'fold':i,'char_actual':metric['feature_dimensions']['char'],'morphology_actual':audit['vocabulary_size'],
            'total_actual':metric['n_features'],'uppercase_ngram_count':int(vocab.ngram.str.contains('A',regex=False).sum()),
            'lowercase_ngram_count':int(vocab.ngram.str.contains('a',regex=False).sum()),
            'df_min':int(vocab[dfcol].min()),'df_max':int(vocab[dfcol].max())})
    train=read_input_csv(Path(runs['char']['meta']['data_dir'])/'train.csv','train');shapes=morphology_batch(train.text)
    descriptions=[]
    for text,shape in zip(train.text,shapes):
        n=len(text);descriptions.append({'original_length':n,'shape_length':len(shape),'upper_A_fraction':shape.count('A')/max(n,1),
            'lower_a_fraction':shape.count('a')/max(n,1),'zero_fraction':shape.count('0')/max(n,1),
            'unchanged_character_fraction':sum(x==y for x,y in zip(text,shape))/max(n,1)})
    description=pd.DataFrame(descriptions);class_description=description.assign(model=train.model.to_numpy()).groupby('model').mean().reset_index()
    OUT.mkdir()
    for name,frame in {'overall':overall,'fold_comparison':folds,'fold_paired':fw.reset_index(),'class_report':classes,
        'sparse_metrics':sparse,'sparse_summary':pd.DataFrame(summary),'sparse_paired':pd.DataFrame(paired),
        'error_transition_ids':transitions,'confusion_changes':pd.DataFrame(changes),'feature_dimensions':pd.DataFrame(dimensions),
        'representative_ngrams':pd.DataFrame(grams),'shape_descriptive':description.describe().reset_index(),'shape_by_class':class_description}.items():
        frame.to_csv(OUT/f'{name}.csv',index=False,lineterminator='\n')
    for name,value in {'metric_deltas':point,'primary_errors':primary,'secondary_errors':secondary,'complementarity':complement,
        'prompt_bootstrap':bootstrap,'analysis_plan':plan,'comparison_provenance':{'primary_control':'B2','secondary_control':'B3-C',
            'runs':{n:r['meta']['run_id'] for n,r in runs.items()},'oof_sha256':{n:digest(r['directory']/'oof_predictions.parquet') for n,r in runs.items()},
            'mask_sha256':mask_meta['mask_sha256'],'old_50_sparse_scores_unchanged':True,'analysis_classifier_fits':0,
            'raw_and_shape_lengths_equal':bool(description.original_length.eq(description.shape_length).all()),
            'B5_plan':'Char+Style+Morphology, control B3-C; not trained'}}.items():write_json(OUT/f'{name}.json',value)
    for filename in ['config.json','run_metadata.json','metrics.json','fold_metrics.csv','classification_report.csv','class_recall.csv',
        'confusion_matrix.csv','confusion_matrix_normalized.csv','warnings.json']:shutil.copyfile(folder/filename,OUT/filename)
    for i in range(1,6):
        target=OUT/f'fold_{i}';target.mkdir();shutil.copyfile(folder/f'fold_{i}/morphology_audit.json',target/'morphology_audit.json')
    print(overall[['model','balanced_accuracy','accuracy','macro_f1','total_fit_seconds','total_inference_seconds','peak_process_mib']].to_string(index=False))
    print('Sparse:',pd.DataFrame(summary)[['model','balanced_accuracy_mean','balanced_accuracy_std_ddof0','balanced_accuracy_min','balanced_accuracy_max']].to_string(index=False))
    print('Deltas:',point);print('Primary errors:',primary);print('Complementarity:',complement);print('Bootstrap:',bootstrap)
    print('Dimensions:',dimensions);print('Sparse paired:',paired)


if __name__=='__main__':
    from unittest.mock import patch
    from sklearn.svm import LinearSVC
    with patch.object(LinearSVC,'fit',side_effect=AssertionError('OOF analysis cannot fit classifiers')):analyze()
