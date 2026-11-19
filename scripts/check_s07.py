"""Ablation configuration, fitted-width and OOF analysis tests; no official fitting."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ntl.experiments import fitted_feature_summary
from ntl.models import baseline_config, make_model, model_config
from analyze_s07 import ROOT, align_oof, check_controls, complementarity, prompt_bootstrap, load_runs


class AblationTests(unittest.TestCase):
    def test_complete_configuration_control_and_default_isolation(self):
        before=baseline_config(); parameters=model_config(make_model())
        for branch in ('word','char'):
            overrides=json.loads((ROOT/f'configs/s07_{branch}_only.json').read_text(encoding='utf-8'))
            untouched=deepcopy(overrides)
            config={'experiment_configuration':__import__('ntl.models',fromlist=['resolve_config']).resolve_config(overrides),
                'model_parameters':model_config(make_model(overrides)),
                'class_order':list(__import__('ntl.labels',fromlist=['class_order']).class_order),'random_seed':20261118}
            check_controls(config,branch)
            self.assertEqual(overrides,untouched)
        self.assertEqual(before,baseline_config()); self.assertEqual(parameters,model_config(make_model()))

    def test_only_enabled_branch_fits_and_width_is_observed(self):
        original=TfidfVectorizer.fit_transform
        for branch in ('word','char'):
            seen=[]
            def tracked(vectorizer,*args,**kwargs):
                seen.append(vectorizer.analyzer)
                return original(vectorizer,*args,**kwargs)
            with patch.object(TfidfVectorizer,'fit_transform',tracked):
                model=make_model({'branches':[branch]})
                model.fit(['alpha beta first','alpha beta second','gamma delta first','gamma delta second'],['a','a','b','b'])
            self.assertEqual(seen,[branch])
            result=fitted_feature_summary(model)
            self.assertEqual(result['feature_branches'],[branch])
            self.assertEqual(result['n_features'],len(model.named_steps['features'].transformer_list[0][1].vocabulary_))
            self.assertEqual(result['n_features'],model.named_steps['classifier'].coef_.shape[1])
            self.assertLess(result['n_features'],40000)
        self.assertNotIn('.fit(',inspect.getsource(fitted_feature_summary))
        self.assertNotIn('.transform(',inspect.getsource(fitted_feature_summary))

    def fixture(self):
        return pd.DataFrame({'id':['001','002','003','004'],'prompt_id':['p1','p1','p2','p2'],
            'fold':[1,1,2,2],'true_model':['a','b','a','b'],'predicted_model':['a','a','a','a']})

    def test_alignment_is_by_string_id_not_position(self):
        base=self.fixture(); other=base.sample(frac=1,random_state=17)
        actual=align_oof({'combo':base,'word':other})
        pd.testing.assert_frame_equal(actual['combo'],actual['word'])
        self.assertEqual(actual['word'].id.tolist(),['001','002','003','004'])

    def test_alignment_rejects_id_and_truth_prompt_fold_errors(self):
        base=self.fixture()
        damaged=[base.iloc[:3],pd.concat([base,base.iloc[[0]]]),base.assign(id=['x','002','003','004']),
            base.assign(prompt_id='wrong'),base.assign(fold=5),base.assign(true_model='wrong'),
            base.assign(id=[1,2,3,4])]
        for frame in damaged:
            with self.subTest(),self.assertRaises(ValueError): align_oof({'combo':base,'word':frame})

    def test_complementarity_all_four_cells_and_combo_rescues(self):
        result=complementarity(['a','b','a','b'],['a','a','a','a'],['a','b','b','a'],['a','b','a','b'])
        self.assertEqual([g['n'] for g in result['groups'].values()],[1,1,1,1])
        self.assertEqual([g['combo_correct'] for g in result['groups'].values()],[1,1,1,1])
        self.assertEqual(result['prediction_agreement_rate'],0.5)
        self.assertEqual(result['at_least_one_correct_rate'],0.75)
        self.assertEqual(result['oracle_balanced_accuracy'],0.75)
        self.assertEqual(result['combo_rescued_both_wrong_count'],1)
        self.assertEqual(result['exclusive_complementary_count'],2)
        self.assertEqual(result['combo_correct_exclusive_count'],2)
        self.assertFalse(result['oracle_is_deployable'])

    def test_complementarity_rejects_empty_or_misaligned_lengths(self):
        for arrays in [([],[],[],[]),(['a'],[],['a'],['a'])]:
            with self.assertRaises(ValueError): complementarity(*arrays)

    def test_prompt_paired_bootstrap_reproducible_identical_predictions(self):
        truth=['a','b','a','b']; groups=['p1','p1','p2','p2']
        predictions={name:truth for name in ('combo','word','char')}
        one=prompt_bootstrap(truth,groups,predictions,repeats=100,seed=17)
        self.assertEqual(one,prompt_bootstrap(truth,groups,predictions,repeats=100,seed=17))
        self.assertEqual(one['n_prompts'],2)
        for interval in one['intervals'].values():
            self.assertEqual(interval,{'observed_delta':0.,'percentile_95_lower':0.,'percentile_95_upper':0.})

    def test_saved_official_runs_all_checks_and_alignment(self):
        runs=load_runs()
        self.assertEqual(len(runs['word']['oof']),2699)
        for branch in ('word','char'):
            self.assertEqual(runs[branch]['meta']['status'],'validated')
            self.assertEqual(runs[branch]['meta']['completed_folds'],[1,2,3,4,5])
            self.assertEqual(runs[branch]['oof'].id.tolist(),runs['combo']['oof'].id.tolist())


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AblationTests))
    raise SystemExit(not result.wasSuccessful())
