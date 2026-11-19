"""Morphology mapping, lowercase, factory isolation and saved B4 acceptance."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from ntl.io import digest
from ntl.models import make_model,model_config,baseline_config,resolve_config
from ntl.morphology import morphology_text,morphology_batch,MorphologyTransformer,morphology_input_digest
from ntl.validation import validate_run
from ntl.evaluation import load_masks,strict_metrics,SPARSE_SEEDS

ROOT=Path(__file__).resolve().parents[1];STAGE=ROOT/'artifacts/s07b_b4_20261119_01'
RUN='s07b_b4_char_morphology_v1_20261119_01';OUT=ROOT/'experiments'/RUN
CONFIG=json.loads((ROOT/'configs/s07b_b4_char_morphology.json').read_text(encoding='utf-8'))
FORMAL='--formal' in sys.argv
if FORMAL:sys.argv.remove('--formal')


class B4Tests(unittest.TestCase):
    def test_mapping_boundaries_reference_fixture(self):
        fixture=json.loads((ROOT/'experiments/s07b_b4_definition_v1/reference_fixture.json').read_text(encoding='utf-8'))
        for row in fixture['examples']:
            self.assertEqual(morphology_text(row['input']),row['expected'])
        self.assertEqual(fixture['official_comparison']['mismatches'],0)
        self.assertEqual(fixture['unicode_mismatches'],0)
        self.assertEqual(fixture['unicode_scalar_values_compared'],1112064)

    def test_unicode_digits_case_and_punctuation(self):
        self.assertEqual(morphology_text('Aa 1.2 １２ ١٢ ²Ⅷ 中文 日本語 😀 _ e\u0301'),'Aa 0.0 00 00 aA aa aaa 😀 _ a\u0301')
        self.assertEqual(morphology_text('ǅİ'),'aA')
        self.assertEqual(morphology_text('# Ab\n\t```X=12``` — ’'), '# Aa\n\t```A=00``` — ’')
        self.assertEqual(morphology_text('A'*10000),'A'*10000)

    def test_batch_determinism_original_order_and_invalid_inputs(self):
        texts=[' AbC  12\t\n','NA','NULL','中文😀'];old=deepcopy(texts)
        a=morphology_batch(texts);self.assertEqual(a,morphology_batch(texts));self.assertEqual(texts,old)
        self.assertEqual(a[::-1],morphology_batch(texts[::-1]));self.assertEqual(morphology_batch([]),[])
        self.assertEqual(MorphologyTransformer().fit_transform(iter(texts)),a)
        for bad in [None,'not a batch',[None],[12],pd.DataFrame({'text':['x']})]:
            with self.assertRaises(ValueError):morphology_batch(bad)

    def test_clone_and_stateless_label_independence(self):
        obj=MorphologyTransformer().fit(['Aa'],['arbitrary'])
        copied=clone(obj);self.assertFalse(hasattr(copied,'fit_text_sha256_'))
        self.assertEqual(copied.transform(['Aa 12']),['Aa 00'])
        self.assertEqual(obj.transform(['Aa 12']),copied.transform(['Aa 12']))

    def test_reference_tfidf_parameters_and_lowercase_trap(self):
        fixture=json.loads((ROOT/'experiments/s07b_b4_definition_v1/reference_fixture.json').read_text(encoding='utf-8'))
        from ntl.models import normalise
        branch=dict(make_model(CONFIG).named_steps['features'].transformer_list)['morphology']
        self.assertEqual(normalise(branch.named_steps['tfidf'].get_params()),fixture['tfidf_actual_parameters'])
        branch.fit(['Aa Aa','Aa Aa','aa aa','aa aa'])
        vocab=branch.named_steps['tfidf'].vocabulary_
        self.assertIn('Aa',vocab);self.assertIn('aa',vocab);self.assertNotEqual(vocab['Aa'],vocab['aa'])
        self.assertFalse(branch.named_steps['tfidf'].lowercase)
        analyzer=branch.named_steps['tfidf'].build_analyzer()
        self.assertEqual(analyzer('Aa \t\naa'),analyzer('Aa aa'))

    def test_actual_b2_parameters_only_add_morphology(self):
        before=json.loads((STAGE/'before.json').read_text(encoding='utf-8'));base=before['model_parameters']['b2']
        new=model_config(make_model(CONFIG))
        for key,value in base.items():
            if key.startswith(('features__char','classifier__')):self.assertEqual(new[key],value,key)
        self.assertEqual(new['features__transformer_weights'],{'morphology':.65})
        self.assertNotIn('features__word',new);self.assertNotIn('features__style',new)
        self.assertEqual(resolve_config(CONFIG)['classifier'],before['baseline_config']['classifier'])

    def test_word_style_never_constructed_and_raw_char_not_polluted(self):
        built=[];fit_inputs=[];original=TfidfVectorizer.fit_transform
        def construct(**kwargs):built.append(kwargs);return TfidfVectorizer(**kwargs)
        def tracked(obj,X,*args,**kwargs):fit_inputs.append((obj.lowercase,list(X)));return original(obj,X,*args,**kwargs)
        with patch('ntl.features.TfidfVectorizer',side_effect=construct),patch('ntl.style.RichStyleTransformer',side_effect=AssertionError('style forbidden')):
            model=make_model(CONFIG)
        self.assertEqual(len(built),2);self.assertTrue(all(x['analyzer']=='char' for x in built))
        raw=['AbC marker!','AbC marker!','xyZ 123?','xyZ 123?'];saved=deepcopy(raw)
        with patch.object(TfidfVectorizer,'fit_transform',tracked):
            features=model.named_steps['features'];matrix=features.fit_transform(raw)
        self.assertEqual(fit_inputs,[(True,raw),(False,morphology_batch(raw))]);self.assertEqual(raw,saved)
        self.assertEqual(matrix.dtype,np.float32)
        transforms=dict(features.transformer_list)
        expected=hstack([transforms['char'].transform(raw),.65*transforms['morphology'].transform(raw)],format='csr')
        self.assertEqual((matrix!=expected).nnz,0)

    def test_vocabulary_and_idf_fit_only_training(self):
        branch=dict(make_model(CONFIG).named_steps['features'].transformer_list)['morphology']
        train=['Aa !!!','Aa !!!','aa ???','aa ???'];branch.fit(train)
        vec=branch.named_steps['tfidf'];vocab=deepcopy(vec.vocabulary_);idf=vec.idf_.copy()
        branch.transform(['AA⁂⁂⁂ new unseen','AA⁂⁂⁂ new unseen'])
        self.assertEqual(vec.vocabulary_,vocab);np.testing.assert_array_equal(vec.idf_,idf)
        self.assertTrue(all('⁂' not in n for n in vocab))
        self.assertEqual(branch.named_steps['shape'].fit_text_sha256_,morphology_input_digest(train))

    def test_disabled_branch_all_legacy_parameters_exact(self):
        before=json.loads((STAGE/'before.json').read_text(encoding='utf-8'))
        self.assertEqual(baseline_config(),before['baseline_config'])
        for name,cfg in [('default',None),('b2',{'branches':['char']}),
            ('b3',json.loads((ROOT/'configs/s07b_b3_rich_style.json').read_text(encoding='utf-8'))),
            ('b3c',json.loads((ROOT/'configs/s07b_b3c_char_style.json').read_text(encoding='utf-8')))]:
            self.assertEqual(model_config(make_model(cfg)),before['model_parameters'][name],name)

    def test_word_restore_one_configuration_switch_no_training(self):
        restored=deepcopy(CONFIG);restored['branches']=['word','char','morphology']
        model=make_model(restored);self.assertEqual([n for n,_ in model.named_steps['features'].transformer_list],restored['branches'])
        self.assertEqual(model_config(model)['features__word__max_features'],40000)
        self.assertEqual(CONFIG['branches'],['char','morphology'])

    def test_protected_history_or_exact_source_archive(self):
        before=json.loads((STAGE/'before.json').read_text(encoding='utf-8'))
        for path,expected in before['protected_sha256'].items():
            if path in before['authorized_source_updates']:self.assertEqual(digest(STAGE/'before-source'/path),expected,path)
            elif path=='experiments/experiments.csv':
                self.assertEqual(digest(STAGE/'experiments-before.csv'),expected)
                self.assertTrue((ROOT/path).read_bytes().startswith((STAGE/'experiments-before.csv').read_bytes()))
            else:self.assertEqual(digest(ROOT/path),expected,path)

    @unittest.skipUnless(FORMAL,'Run --formal after the new official experiment')
    def test_formal_oof_morphology_audits_and_vocabulary(self):
        folder=ROOT/'artifacts'/RUN;validate_run(folder)
        frame=pd.read_parquet(folder/'oof_predictions.parquet')
        from ntl.data import read_input_csv
        train=read_input_csv(ROOT/'data/name-that-llm/train.csv','train')
        for i in range(1,6):
            audit=json.loads((folder/f'fold_{i}/morphology_audit.json').read_text(encoding='utf-8'))
            fits=train[~train.id.isin(frame.loc[frame.fold==i,'id'])]
            self.assertEqual(audit['raw_fit_text_sha256'],morphology_input_digest(fits.text))
            self.assertEqual(audit['n_fit_samples'],len(fits));self.assertTrue(audit['vocabulary_idf_unchanged_after_validation'])
            vocab=pd.read_csv(folder/f'fold_{i}/morphology_vocabulary.csv',keep_default_na=False)
            self.assertEqual(len(vocab),audit['vocabulary_size']);self.assertTrue(np.isfinite(vocab.idf).all())
            self.assertTrue(vocab.ngram.str.len().between(2,6).all())

    @unittest.skipUnless(FORMAL,'Run --formal after fixed-mask analysis')
    def test_saved_sparse_scores_primary_control_and_bootstrap(self):
        frame=pd.read_parquet(ROOT/'artifacts'/RUN/'oof_predictions.parquet')
        masks,meta=load_masks(frame[['id','prompt_id','fold']],ROOT/'data/name-that-llm/train.csv',ROOT/'experiments/folds/folds_v1.csv',ROOT/'experiments/evaluation_masks/sparse_k2_v1')
        scores=pd.read_csv(OUT/'sparse_metrics.csv');self.assertEqual(len(scores),60)
        for seed in SPARSE_SEEDS:
            one=frame.set_index('id').loc[masks.loc[masks.seed==seed,'id']];m=strict_metrics(one.true_model,one.predicted_model)
            row=scores[(scores.model=='b4')&(scores.seed==seed)].iloc[0]
            for key in ['balanced_accuracy','accuracy','macro_f1']:self.assertAlmostEqual(m[key],row[key],places=14)
        provenance=json.loads((OUT/'comparison_provenance.json').read_text(encoding='utf-8'))
        self.assertEqual(provenance['primary_control'],'B2')
        self.assertEqual(provenance['mask_sha256'],meta['mask_sha256'])
        boot=json.loads((OUT/'prompt_bootstrap.json').read_text(encoding='utf-8'))
        self.assertEqual((boot['seed'],boot['repeats'],boot['unit']),(20261118,5000,'prompt'))


if __name__=='__main__':
    with patch.object(LinearSVC,'fit',side_effect=AssertionError('Acceptance tests cannot fit classifiers')):
        unittest.main(verbosity=2)
