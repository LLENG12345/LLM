"""Rich162, default-factory compatibility and optional saved B3 run checks.

Tests fit small feature transformers, never classifiers or official models.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ntl.data import read_input_csv
from ntl.evaluation import load_masks
from ntl.io import digest
from ntl.labels import class_order
from ntl.models import baseline_config, make_model, model_config, resolve_config
from ntl.style import (STYLE_FEATURE_NAMES, FUNCTION_WORDS, MARKS, RichStyleTransformer,
                       rich_style_matrix, text_sequence_digest)
from ntl.validation import validate_run

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'artifacts/s07b_b3_20261119_01'
CONFIG=json.loads((ROOT/'configs/s07b_b3_rich_style.json').read_text(encoding='utf-8'))
FORMAL=ROOT/'artifacts/s07b_b3_rich_style_v1_20261119_01'
CHECK_FORMAL='--formal' in sys.argv
if CHECK_FORMAL: sys.argv.remove('--formal')
CHECK_REGRESSIONS='--regressions' in sys.argv
if CHECK_REGRESSIONS: sys.argv.remove('--regressions')
TEXTS=['alpha beta first. We write clearly.','alpha beta second! We write briefly.',
       'gamma delta first. However, a long sentence.','gamma delta second? I, you and we.']


class StyleTests(unittest.TestCase):
    def test_reference_fixture_exact_and_feature_order(self):
        fixture=json.loads((ROOT/'experiments/s07b_b3_definition_v1/reference_fixture.json').read_text(encoding='utf-8'))
        self.assertEqual(list(STYLE_FEATURE_NAMES),fixture['feature_names'])
        self.assertEqual((len(STYLE_FEATURE_NAMES),len(FUNCTION_WORDS),len(MARKS)),(162,81,25))
        self.assertEqual(len(set(STYLE_FEATURE_NAMES)),162)
        actual=rich_style_matrix(fixture['texts'])
        np.testing.assert_array_equal(actual,np.asarray(fixture['expected'],dtype=np.float32))
        self.assertEqual(actual.dtype,np.float32)
        self.assertTrue(fixture['official_raw_comparison']['array_equal'])
        raw=rich_style_matrix(read_input_csv(ROOT/'data/name-that-llm/train.csv','train').text)
        import hashlib
        self.assertEqual(hashlib.sha256(raw.tobytes()).hexdigest(),fixture['official_raw_comparison']['raw_matrix_sha256'])

    def test_boundary_values_finite_and_empty_batch(self):
        texts=['','a','   ','\n\t','!!!','12345','“”’—','中文😀','ＡＢＣ İ １２']
        self.assertTrue(np.isfinite(rich_style_matrix(texts)).all())
        self.assertEqual(rich_style_matrix([]).shape,(0,162))
        # Original method deliberately uses max(word_count,1), even in log1p.
        self.assertAlmostEqual(float(rich_style_matrix([''])[0,1]),np.log(2),places=6)
        for invalid in [None,'single string',[None],[12],pd.DataFrame({'id':['001'],'text':['x']})]:
            with self.assertRaises(ValueError): rich_style_matrix(invalid)

    def test_individual_batch_order_reproducible_text_preserved(self):
        original=[' A  Word!\nNew line.','NULL','NA','İ déjà — “x”']
        saved=deepcopy(original)
        a=rich_style_matrix(original)
        np.testing.assert_array_equal(a,rich_style_matrix(original))
        np.testing.assert_array_equal(a,np.vstack([rich_style_matrix([x]) for x in original]))
        np.testing.assert_array_equal(a[::-1],rich_style_matrix(original[::-1]))
        self.assertEqual(original,saved)

    def test_hand_calculated_named_features(self):
        row=dict(zip(STYLE_FEATURE_NAMES,rich_style_matrix(['A a a!\n123.'])[0]))
        self.assertEqual(row['function_a_per100words'],75)
        self.assertEqual(row['mark_u0021_present'],1)
        self.assertEqual(row['pattern_digit_per100words'],75)
        self.assertEqual(row['pattern_ascii_uppercase_per100words'],25)
        self.assertEqual(row['type_token_ratio'],.5)
        self.assertEqual(row['hapax_token_fraction'],.25)

    def test_scaler_only_sees_training_and_transform_is_readonly(self):
        transformer=RichStyleTransformer(); seen=[]; original=StandardScaler.fit
        def tracked(obj,X,y=None,**kwargs):
            seen.append(X.copy());return original(obj,X,y=y,**kwargs)
        with patch.object(StandardScaler,'fit',tracked):
            actual=transformer.fit_transform(TEXTS,['ignored']*len(TEXTS))
            state=[x.copy() for x in (transformer.scaler_.mean_,transformer.scaler_.var_,transformer.scaler_.scale_)]
            val=transformer.transform(['VALIDATIONONLY '*1000+'!\n'*200])
        self.assertEqual(len(seen),1)
        np.testing.assert_array_equal(seen[0],rich_style_matrix(TEXTS))
        oracle=StandardScaler().fit_transform(rich_style_matrix(TEXTS))
        np.testing.assert_array_equal(actual,np.clip(oracle,-5,5))
        for a,b in zip(state,(transformer.scaler_.mean_,transformer.scaler_.var_,transformer.scaler_.scale_)):
            np.testing.assert_array_equal(a,b)
        self.assertEqual(int(transformer.scaler_.n_samples_seen_),len(TEXTS))
        self.assertEqual(transformer.fit_text_sha256_,text_sequence_digest(TEXTS))
        self.assertTrue(np.abs(val).max()<=5)
        self.assertEqual(transformer.transform([]).shape,(0,162))

    def test_scaler_ignores_labels_and_new_fold_is_unfitted(self):
        a=RichStyleTransformer();b=RichStyleTransformer()
        np.testing.assert_array_equal(a.fit_transform(TEXTS,['a']*4),b.fit_transform(TEXTS,['z']*4))
        copied=clone(a);self.assertFalse(hasattr(copied,'scaler_'))
        with self.assertRaises(ValueError): RichStyleTransformer(clip=0).fit(TEXTS)
        with self.assertRaises(ValueError): RichStyleTransformer(dtype='float64').fit(TEXTS)

    def test_feature_union_weight_sparse_dense_dtype_and_vocabulary_isolation(self):
        features=make_model(CONFIG).named_steps['features']
        joined=features.fit_transform(TEXTS,['a','a','b','b'])
        self.assertTrue(issparse(joined));self.assertEqual(joined.dtype,np.float32)
        transforms=dict(features.transformer_list)
        expected=hstack([transforms['word'].transform(TEXTS),transforms['char'].transform(TEXTS),
                         .075*transforms['style'].transform(TEXTS)],format='csr')
        self.assertEqual((joined!=expected).nnz,0)
        np.testing.assert_array_equal(joined[:,-162:].toarray(),.075*transforms['style'].transform(TEXTS))
        words=deepcopy(transforms['word'].vocabulary_);idf=transforms['word'].idf_.copy()
        features.transform(['validationonly unseenmarker extraordinarily!!!'])
        self.assertEqual(words,transforms['word'].vocabulary_)
        self.assertNotIn('validationonly',words)
        np.testing.assert_array_equal(idf,transforms['word'].idf_)

    def test_baseline_b0_b1_b2_actual_parameters_unchanged(self):
        before=json.loads((STAGE/'before.json').read_text(encoding='utf-8'))
        self.assertEqual(baseline_config(),before['baseline_config'])
        self.assertEqual(model_config(make_model()),before['baseline_parameters'])
        self.assertEqual(model_config(make_model({'branches':['word']})),before['word_parameters'])
        self.assertEqual(model_config(make_model({'branches':['char']})),before['char_parameters'])
        self.assertNotIn('style',baseline_config()['features'])
        self.assertIsNone(make_model().named_steps['features'].transformer_weights)

    def test_b3_controlled_actual_configuration(self):
        supplied=deepcopy(CONFIG); resolved=resolve_config(CONFIG)
        old=json.loads((STAGE/'before.json').read_text(encoding='utf-8'))
        self.assertEqual(resolved['classifier'],old['baseline_config']['classifier'])
        for name in ['word','char']:
            self.assertEqual(resolved['features'][name],old['baseline_config']['features'][name])
        current=model_config(make_model(CONFIG))
        for key,value in old['baseline_parameters'].items():
            if key.startswith(('features__word','features__char','classifier__')):
                self.assertEqual(current[key],value,key)
        self.assertEqual(current['features__transformer_weights'],{'style':.075})
        self.assertEqual(current['features__style__clip'],5)
        self.assertEqual(supplied,CONFIG)
        resolved['features']['word']['min_df']=99
        self.assertEqual(baseline_config()['features']['word']['min_df'],2)

    def test_historical_artifacts_and_archived_before_code_unchanged(self):
        before=json.loads((STAGE/'before.json').read_text(encoding='utf-8'))
        for name,expected in before['protected_sha256'].items():
            if name in before['authorized_source_updates']:
                self.assertEqual(digest(STAGE/'before-source'/name),expected,name)
            elif name=='experiments/experiments.csv':
                self.assertTrue((ROOT/name).read_bytes().startswith((STAGE/'experiments-before.csv').read_bytes()))
            else:
                self.assertEqual(digest(ROOT/name),expected,name)

    def test_historical_oof_validation_no_retraining(self):
        from analyze_s07 import load_runs
        results=load_runs()
        self.assertEqual(set(results),{'combo','word','char'})

    def test_error_comparison_four_cells(self):
        from analyze_b3 import compare_errors
        a,b=class_order[0],class_order[-1]
        counts=compare_errors([a,b,a,b],[a,b,b,a],[a,a,a,a])
        self.assertEqual(counts['both_correct'],1)
        self.assertEqual(counts['b0_only_correct_new_errors'],1)
        self.assertEqual(counts['b3_only_correct_corrected'],1)
        self.assertEqual(counts['both_wrong'],1)
        self.assertEqual(counts['same_provider_errors_b0'],0)
        self.assertEqual(counts['cross_provider_errors_b3'],2)

    @unittest.skipUnless(CHECK_FORMAL,'Use --formal after saved B3 comparison')
    def test_four_model_comparison_fixed_masks_and_error_counts(self):
        from ntl.evaluation import strict_metrics,SPARSE_SEEDS
        folder=ROOT/'experiments'/FORMAL.name
        frame=pd.read_parquet(FORMAL/'oof_predictions.parquet')
        train_path=ROOT/'data/name-that-llm/train.csv'
        masks,meta=load_masks(frame[['id','prompt_id','fold']],train_path,ROOT/'experiments/folds/folds_v1.csv',ROOT/'experiments/evaluation_masks/sparse_k2_v1')
        scores=pd.read_csv(folder/'sparse_metrics.csv')
        self.assertEqual(len(scores),40);self.assertEqual(scores.mask_sha256.nunique(),1)
        self.assertEqual(scores.mask_sha256.iloc[0],meta['mask_sha256'])
        for seed in SPARSE_SEEDS:
            one=frame.set_index('id').loc[masks.loc[masks.seed==seed,'id']]
            actual=strict_metrics(one.true_model,one.predicted_model)
            saved=scores[(scores.model=='b3')&(scores.seed==seed)].iloc[0]
            for name in ['balanced_accuracy','accuracy','macro_f1']:
                self.assertAlmostEqual(actual[name],saved[name],places=14)
        transitions=pd.read_csv(folder/'error_transition_ids.csv',dtype={'id':'string','prompt_id':'string'})
        self.assertEqual(set(transitions.id),set(frame.id))
        counts=json.loads((folder/'error_comparison.json').read_text(encoding='utf-8'))
        for transition,key in [('both_correct','both_correct'),('both_wrong','both_wrong'),
                               ('corrected','b3_only_correct_corrected'),('new_error','b0_only_correct_new_errors')]:
            self.assertEqual(int(transitions.transition.eq(transition).sum()),counts[key])

    @unittest.skipUnless(CHECK_FORMAL,'Use --formal after the new five-fold run')
    def test_formal_oof_and_scaler_audits(self):
        validate_run(FORMAL)
        train=read_input_csv(ROOT/'data/name-that-llm/train.csv','train')
        oof=pd.read_parquet(FORMAL/'oof_predictions.parquet')
        self.assertEqual(len(oof),2699)
        self.assertEqual(oof.id.tolist(),train.id.tolist())
        for fold in range(1,6):
            metrics=json.loads((FORMAL/f'fold_{fold}/metrics.json').read_text(encoding='utf-8'))
            audit=json.loads((FORMAL/f'fold_{fold}/style_audit.json').read_text(encoding='utf-8'))
            self.assertEqual(metrics['feature_dimensions']['style'],162)
            self.assertEqual(metrics['n_features'],sum(metrics['feature_dimensions'].values()))
            self.assertEqual(audit['feature_names'],list(STYLE_FEATURE_NAMES))
            fit=train[~train.id.isin(oof.loc[oof.fold==fold,'id'])]
            self.assertEqual(audit['n_fit_samples'],len(fit))
            self.assertEqual(audit['fit_text_sha256'],text_sequence_digest(fit.text))
            raw=rich_style_matrix(fit.text);scaler=StandardScaler().fit(raw)
            np.testing.assert_array_equal(audit['mean'],scaler.mean_)
            np.testing.assert_array_equal(audit['var'],scaler.var_)
            np.testing.assert_array_equal(audit['scale'],scaler.scale_)
            self.assertEqual(audit['weight'],.075)


if __name__=='__main__':
    with patch.object(LinearSVC,'fit',side_effect=AssertionError('Tests must not train classifiers')):
        if CHECK_REGRESSIONS:
            import check_s07,check_s07a
            loader=unittest.defaultTestLoader
            suite=loader.loadTestsFromTestCase(StyleTests)
            exclusions={check_s07.AblationTests:{'test_only_enabled_branch_fits_and_width_is_observed'},
                check_s07a.ProtocolTests:{'test_existing_resources_and_v1_history_unchanged'}}
            for cls,excluded in exclusions.items():
                for name in loader.getTestCaseNames(cls):
                    if name not in excluded:suite.addTest(cls(name))
            print('Legacy exclusions: synthetic classifier fits avoided; pre-B3 active-source byte freeze replaced by exact B3 before-source archive + unchanged-artifact checks.',flush=True)
            result=unittest.TextTestRunner(verbosity=2).run(suite)
            sys.exit(0 if result.wasSuccessful() else 1)
        else:
            unittest.main(verbosity=2)
