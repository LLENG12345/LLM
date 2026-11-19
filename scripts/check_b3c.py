"""B3-C branch removal, historical integrity and saved-result acceptance."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from ntl.io import digest
from ntl.models import make_model,model_config,resolve_config,baseline_config
from ntl.style import rich_style_matrix,STYLE_FEATURE_NAMES
from ntl.validation import validate_run
from ntl.evaluation import strict_metrics,load_masks,SPARSE_SEEDS
from analyze_b3c import ROOT,RUN,STAGE,OUT,factor_contrasts

CONFIG=json.loads((ROOT/'configs/s07b_b3c_char_style.json').read_text(encoding='utf-8'))
FORMAL='--formal' in sys.argv
if FORMAL:sys.argv.remove('--formal')


class B3CTests(unittest.TestCase):
    def test_configuration_only_removes_word_actual_params(self):
        b3=json.loads((ROOT/'configs/s07b_b3_rich_style.json').read_text(encoding='utf-8'))
        expected=deepcopy(b3);expected['branches']=['char','style'];self.assertEqual(CONFIG,expected)
        a,b=resolve_config(b3),resolve_config(CONFIG);a['branches']=b['branches'];self.assertEqual(a,b)
        old=model_config(make_model(b3));new=model_config(make_model(CONFIG))
        removed={k for k in old if k=='features__word' or k.startswith('features__word__')}
        self.assertEqual(set(new),set(old)-removed)
        for key in set(new)-{'steps','features','features__transformer_list'}:self.assertEqual(new[key],old[key],key)
        self.assertEqual(new['features__transformer_weights'],{'style':.075})

    def test_word_never_constructed_or_fitted(self):
        built=[];fitted=[];original=TfidfVectorizer.fit_transform
        def construct(**kwargs):built.append(kwargs['analyzer']);return TfidfVectorizer(**kwargs)
        def tracked(obj,*args,**kwargs):fitted.append(obj.analyzer);return original(obj,*args,**kwargs)
        with patch('ntl.features.TfidfVectorizer',side_effect=construct):model=make_model(CONFIG)
        self.assertEqual(built,['char'])
        transforms=model.named_steps['features'];self.assertEqual([n for n,_ in transforms.transformer_list],['char','style'])
        with patch.object(TfidfVectorizer,'fit_transform',tracked):
            result=transforms.fit_transform(['alpha beta first','alpha beta second','gamma delta first','gamma delta second'])
        self.assertEqual(fitted,['char']);self.assertEqual(result.dtype,np.float32)
        self.assertEqual(dict(transforms.transformer_list)['style'].n_features_out_,162)

    def test_same_style_outputs_scaling_and_names(self):
        a=dict(make_model(CONFIG).named_steps['features'].transformer_list)['style']
        b3=json.loads((ROOT/'configs/s07b_b3_rich_style.json').read_text(encoding='utf-8'))
        b=dict(make_model(b3).named_steps['features'].transformer_list)['style']
        self.assertEqual(a.get_params(),b.get_params())
        texts=['A  text.\nNext!','Another long word therefore.','abc123 -- “YES”','']
        np.testing.assert_array_equal(a.fit_transform(texts),b.fit_transform(texts))
        np.testing.assert_array_equal(a.get_feature_names_out(),STYLE_FEATURE_NAMES)
        state=a.scaler_.mean_.copy();a.transform(['UNSEEN '*300]);np.testing.assert_array_equal(a.scaler_.mean_,state)
        self.assertEqual(rich_style_matrix(texts).shape,(4,162))

    def test_default_and_b3_unmodified(self):
        before=json.loads((STAGE/'before.json').read_text(encoding='utf-8'))
        b3=json.loads((ROOT/'configs/s07b_b3_rich_style.json').read_text(encoding='utf-8'))
        self.assertEqual(baseline_config(),before['baseline_config'])
        self.assertEqual(model_config(make_model()),before['baseline_parameters'])
        self.assertEqual(model_config(make_model(b3)),before['b3_parameters'])
        supplied=deepcopy(CONFIG);make_model(CONFIG);self.assertEqual(CONFIG,supplied)

    def test_history_all_sources_and_masks_unchanged(self):
        before=json.loads((STAGE/'before.json').read_text(encoding='utf-8'))
        for path,expected in before['protected_sha256'].items():
            if path=='experiments/experiments.csv':
                self.assertEqual(digest(STAGE/'experiments-before.csv'),expected)
                self.assertTrue((ROOT/path).read_bytes().startswith((STAGE/'experiments-before.csv').read_bytes()))
            else:self.assertEqual(digest(ROOT/path),expected,path)

    def test_factor_formula_and_units_identity(self):
        result=factor_contrasts({'combo':.30,'char':.32,'b3':.38,'b3c':.39})
        expected=[-.02,-.01,.08,.07,.01]
        np.testing.assert_allclose(list(result.values()),expected,atol=1e-15,rtol=0)
        self.assertAlmostEqual(result['interaction_B_minus_A'],result['C_style_with_word']-result['D_style_without_word'])

    def test_bootstrap_preregistered(self):
        plan=json.loads((STAGE/'analysis-plan.json').read_text(encoding='utf-8'))
        self.assertEqual((plan['bootstrap_seed'],plan['bootstrap_repeats']),(20261118,5000))
        self.assertEqual(plan['bootstrap_unit'],'prompt')
        if FORMAL:
            meta=json.loads((ROOT/'artifacts'/RUN/'run_metadata.json').read_text(encoding='utf-8'))
            self.assertLess(plan['frozen_utc'],meta['started_utc'])

    @unittest.skipUnless(FORMAL,'Run --formal after the new official experiment')
    def test_formal_complete_scores_and_identical_scaler_audit(self):
        folder=ROOT/'artifacts'/RUN;validate_run(folder)
        for i in range(1,6):
            old=ROOT/f'artifacts/s07b_b3_rich_style_v1_20261119_01/fold_{i}'
            new=folder/f'fold_{i}'
            self.assertEqual((new/'style_audit.json').read_bytes(),(old/'style_audit.json').read_bytes())
            self.assertEqual((new/'split_audit.json').read_bytes(),(old/'split_audit.json').read_bytes())
            metrics=json.loads((new/'metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(metrics['feature_dimensions'],{'char':80000,'style':162})
            self.assertEqual(metrics['n_features'],80162)

    @unittest.skipUnless(FORMAL,'Run --formal after OOF analysis')
    def test_saved_sparse_factor_and_pair_counts_recomputed(self):
        frame=pd.read_parquet(ROOT/'artifacts'/RUN/'oof_predictions.parquet')
        masks,meta=load_masks(frame[['id','prompt_id','fold']],ROOT/'data/name-that-llm/train.csv',
            ROOT/'experiments/folds/folds_v1.csv',ROOT/'experiments/evaluation_masks/sparse_k2_v1')
        saved=pd.read_csv(OUT/'sparse_metrics.csv');self.assertEqual(len(saved),50)
        self.assertEqual(saved.mask_sha256.nunique(),1);self.assertEqual(saved.mask_sha256.iloc[0],meta['mask_sha256'])
        for seed in SPARSE_SEEDS:
            one=frame.set_index('id').loc[masks.loc[masks.seed==seed,'id']]
            m=strict_metrics(one.true_model,one.predicted_model);row=saved[(saved.model=='b3c')&(saved.seed==seed)].iloc[0]
            for key in ['balanced_accuracy','accuracy','macro_f1']:self.assertAlmostEqual(m[key],row[key],places=14)
        full=pd.read_csv(OUT/'overall.csv').set_index('model').balanced_accuracy.to_dict()
        actual=factor_contrasts(full);factors=pd.read_csv(OUT/'factor_contrasts.csv')
        for contrast,value in actual.items():
            row=factors[(factors.evaluation=='full')&(factors.contrast==contrast)].iloc[0]
            self.assertAlmostEqual(row.delta,value,places=14);self.assertAlmostEqual(row.delta_pp,100*value,places=12)
        transitions=pd.read_csv(OUT/'error_transition_ids.csv');counts=json.loads((OUT/'paired_errors.json').read_text(encoding='utf-8'))
        for transition,key in [('both_correct','both_correct'),('both_wrong','both_wrong'),('b3c_new_error','b3c_new_errors'),('b3c_corrected','b3c_corrected')]:
            self.assertEqual(int(transitions.transition.eq(transition).sum()),counts[key])


if __name__=='__main__':
    with patch.object(LinearSVC,'fit',side_effect=AssertionError('Acceptance tests cannot train classifiers')):
        unittest.main(verbosity=2)
