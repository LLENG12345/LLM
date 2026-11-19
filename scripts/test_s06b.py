"""Shared-package behavior/config protection, using only tiny synthetic fits."""
import ast
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ntl import data, labels, models, splits, experiments, submission
from ntl.io import digest

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'artifacts/s06b/before-source'


def function_ast(source, name):
    return ast.dump(next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name))


class SharedPackageTests(unittest.TestCase):
    def test_complete_default_configuration(self):
        baseline = json.loads((ROOT / 'artifacts/s04_baseline_v1_20261118_01/config.json').read_text(encoding='utf-8'))
        self.assertEqual(models.model_config(models.make_model()), baseline['model_parameters'])
        self.assertEqual(models.RANDOM_SEED, baseline['random_seed'])
        self.assertEqual(list(labels.class_order), baseline['class_order'])

    def test_configuration_isolation_and_three_branches_on_tiny_corpus(self):
        original = models.baseline_config()
        for branches in (['word'], ['char'], ['word', 'char']):
            overrides = {'branches': branches, 'classifier': {'C': 0.5}}
            before = deepcopy(overrides)
            model = models.make_model(overrides)
            self.assertEqual([n for n, _ in model.named_steps['features'].transformer_list], branches)
            model.fit(['alpha beta first', 'alpha beta second', 'gamma delta first', 'gamma delta second'], ['a','a','b','b'])
            self.assertEqual(len(model.predict(['alpha beta'])), 1)
            self.assertEqual(overrides, before)
        self.assertEqual(models.baseline_config(), original)
        copy = models.resolve_config(); copy['features']['word']['min_df'] = 1
        self.assertEqual(models.baseline_config(), original)

    def test_bad_configuration_rejected(self):
        for config in ({'unknown': 1}, {'classifier': {'unknown': 2}}, {'branches': []},
                       {'branches': ['word','word']}, {'branches': ['new']}, {'branches': 'word'},
                       {'features': {'word': {'dtype': 'float64'}}}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                models.make_model(config)

    def test_verbatim_original_input_contracts(self):
        original = json.loads((ORIGINAL / 'notebooks/01_baseline.ipynb').read_text(encoding='utf-8'))
        source = ''.join(original['cells'][2]['source'])
        for name in ('_locations','_validate_fields','validate_train','validate_test','read_input_csv',
                     'file_fingerprint','validate_sample_submission','load_competition_data'):
            self.assertEqual(function_ast(source,name), function_ast(inspect.getsource(getattr(data,name)),name))
        path_source = ''.join(original['cells'][1]['source'])
        self.assertEqual(function_ast(path_source,'resolve_data_dir'), function_ast(inspect.getsource(data.resolve_data_dir),'resolve_data_dir'))

    def test_verbatim_original_fixed_folds(self):
        original = json.loads((ORIGINAL / 'notebooks/01_baseline.ipynb').read_text(encoding='utf-8'))
        source = ''.join(original['cells'][3]['source'])
        for name in ('generate_fold_map','validate_fold_map','save_fixed_folds','load_fixed_folds','iter_fixed_splits'):
            self.assertEqual(function_ast(source,name), function_ast(inspect.getsource(getattr(splits,name)),name))

    def test_verbatim_original_fold_fit_and_metrics(self):
        source = (ORIGINAL / 'scripts/run_s04.py').read_text(encoding='utf-8')
        for name in ('run_fold','finalise'):
            current = ast.parse(inspect.getsource(getattr(experiments,name)))
            if name == 'run_fold':
                expected_observation = ast.dump(ast.parse('metric.update(fitted_feature_summary(model))').body[0])
                observations = [n for n in current.body[0].body if ast.dump(n) == expected_observation]
                self.assertEqual(len(observations),1)
                current.body[0].body = [n for n in current.body[0].body if ast.dump(n) != expected_observation]
            self.assertEqual(function_ast(source,name),ast.dump(current.body[0]))

    def test_verbatim_original_submission_contract(self):
        original = (ORIGINAL / 'scripts/check_s05.py').read_text(encoding='utf-8')
        for name in ('validate_submission','write_and_verify_submission'):
            self.assertEqual(function_ast(original,name), function_ast(inspect.getsource(getattr(submission,name)),name))

    def test_no_notebook_read_for_factory(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        from _common import load_factory
        real_open = __import__('builtins').open
        def guarded(file, *args, **kwargs):
            if isinstance(file,(str,os.PathLike)) and str(file).endswith('.ipynb'):
                raise AssertionError('Factory must not read a Notebook')
            return real_open(file,*args,**kwargs)
        with patch('builtins.open',guarded):
            namespace, notebook, _ = load_factory()
            self.assertIsNone(notebook)
            self.assertEqual(models.model_config(namespace['make_model']()), models.model_config(models.make_model()))

    def test_generator_does_not_read_local_artifacts(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        from build_submission_notebook import build
        original_open = Path.open
        def guarded(path,*args,**kwargs):
            if Path(path).resolve().is_relative_to(ROOT/'artifacts'):
                raise AssertionError('Submission generation must use shared code and lightweight records')
            return original_open(path,*args,**kwargs)
        with patch.object(Path,'open',guarded):
            payload, provenance = build()
        self.assertIn('src/ntl/models.py',provenance['source_sha256'])
        self.assertIn('submission.parquet',payload)

    def test_independent_import_from_other_working_directory(self):
        with tempfile.TemporaryDirectory(prefix='ntl-import-') as temporary:
            for cwd in (ROOT, ROOT/'notebooks', Path(temporary)):
                result = subprocess.run([sys.executable,'-I','-c',
                    'from ntl.models import make_model; from ntl.data import read_input_csv; from ntl.experiments import run_experiment; print(type(make_model()).__name__)'],
                    cwd=cwd,capture_output=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stderr.decode(errors='replace'))
                self.assertIn(b'Pipeline',result.stdout)

    def test_fixed_v1_id_alignment_and_no_file_changes(self):
        directory = ROOT/'data/name-that-llm'; train_path=directory/'train.csv'
        train = data.read_input_csv(train_path,'train')
        for version in ('folds_v1','folds_v2'):
            csv = ROOT/f'experiments/folds/{version}.csv'; meta=csv.with_name(version+'_metadata.json')
            before = digest(csv),digest(meta)
            aligned,_ = splits.load_fixed_folds(train,train_path,csv,meta)
            reordered = train.sample(frac=1,random_state=17)
            other,_ = splits.load_fixed_folds(reordered,train_path,csv,meta)
            self.assertEqual(dict(zip(train.id,aligned)),dict(zip(reordered.id,other)))
            for fit,valid in splits.iter_fixed_splits(train,train_path,csv,meta):
                self.assertFalse(set(train.iloc[fit].prompt_id)&set(train.iloc[valid].prompt_id))
            self.assertEqual((digest(csv),digest(meta)),before)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SharedPackageTests))
    raise SystemExit(not result.wasSuccessful())
