"""Generate/verify lightweight folds and EDA; never execute official model fitting."""
import argparse
import ast
from contextlib import chdir, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


from _common import load_context
from ntl.analysis import basic_analysis


class FoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns, cls.nb = load_context()
        cls.train_path = cls.ns['DATA_DIR'] / 'train.csv'
        cls.train, cls.test, _, _ = cls.ns['load_competition_data'](cls.ns['DATA_DIR'])
        cls.maps = {v: pd.read_csv(ROOT / f'experiments/folds/{v}.csv', dtype='string') for v in ('folds_v1', 'folds_v2')}

    def test_reproducibility_and_original_v1(self):
        for version, stored in self.maps.items():
            generated, _ = self.ns['generate_fold_map'](self.train, version)
            pd.testing.assert_frame_equal(generated.astype('string'), stored)
        original = self.ns['GroupKFold'](n_splits=5)
        expected = np.empty(len(self.train), dtype=int)
        for fold, (_, valid) in enumerate(original.split(self.train['text'], self.train['model'], self.train['prompt_id']), 1):
            expected[valid] = fold
        aligned, _ = self.ns['validate_fold_map'](self.train, self.maps['folds_v1'])
        np.testing.assert_array_equal(expected, aligned)

    def test_load_shuffle_and_all_boundaries(self):
        for version in self.maps:
            csv = ROOT / f'experiments/folds/{version}.csv'
            meta = ROOT / f'experiments/folds/{version}_metadata.json'
            aligned, metadata = self.ns['load_fixed_folds'](self.train, self.train_path, csv, meta)
            shuffled = self.train.sample(frac=1, random_state=17)
            other, _ = self.ns['load_fixed_folds'](shuffled, self.train_path, csv, meta)
            self.assertEqual(dict(zip(self.train['id'], aligned)), dict(zip(shuffled['id'], other)))
            splits = list(self.ns['iter_fixed_splits'](self.train, self.train_path, csv, meta))
            self.assertEqual(sorted(np.concatenate([v for _, v in splits]).tolist()), list(range(len(self.train))))
            for fit, valid in splits:
                self.assertFalse(set(self.train.iloc[fit]['prompt_id']) & set(self.train.iloc[valid]['prompt_id']))
                self.assertEqual(set(self.train.iloc[fit]['model']), set(self.ns['class_order']))
                self.assertEqual(set(self.train.iloc[valid]['model']), set(self.ns['class_order']))

    def test_corrupt_maps(self):
        mapping = self.maps['folds_v1']
        cases = [mapping.iloc[1:], pd.concat([mapping, mapping.iloc[[0]]]),
                 pd.concat([mapping, pd.DataFrame([{'id': 'extra', 'prompt_id': 'extra', 'fold': '1'}])]),
                 mapping.assign(fold='6'), mapping.assign(fold='1.0'), mapping.assign(fold='1'), mapping.assign(prompt_id='wrong')]
        split_prompt = mapping.copy()
        split_prompt.loc[0, 'fold'] = str(int(mapping.iloc[0]['fold']) % 5 + 1)
        cases.append(split_prompt)
        for broken in cases:
            with self.subTest(), self.assertRaises(ValueError):
                self.ns['validate_fold_map'](self.train, broken)

    def test_file_metadata_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix='ntl-s03-') as tmp:
            tmp = Path(tmp)
            csv, meta = self.ns['save_fixed_folds'](self.train, self.train_path, tmp)
            before = csv.read_bytes(), meta.read_bytes()
            with self.assertRaises(FileExistsError):
                self.ns['save_fixed_folds'](self.train, self.train_path, tmp)
            self.assertEqual(before, (csv.read_bytes(), meta.read_bytes()))
            original = json.loads(meta.read_text(encoding='utf-8'))
            for key, value in [('train_sha256', 'wrong'), ('fold_sha256', 'wrong'), ('n_samples', 1), ('class_order', []), ('algorithm', 'wrong'), ('parameters', {})]:
                changed = dict(original); changed[key] = value
                meta.write_text(json.dumps(changed), encoding='utf-8')
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.ns['load_fixed_folds'](self.train, self.train_path, csv, meta)

    def test_versions_are_not_just_renumbered(self):
        tables = [m.drop_duplicates('prompt_id').set_index('prompt_id')['fold'] for m in self.maps.values()]
        cross = pd.crosstab(tables[0], tables[1])
        self.assertTrue((cross.gt(0).sum(axis=1) > 1).all())

    def test_issue_detector_on_synthetic_data(self):
        frame = self.train.iloc[:3].copy()
        frame.loc[frame.index[1], 'text'] = frame.iloc[0]['text']
        frame.loc[frame.index[1], 'model'] = frame.iloc[0]['model'] + '-different'
        frame.loc[frame.index[2], ['prompt_id', 'model']] = frame.iloc[0][['prompt_id', 'model']].tolist()
        report, _, _ = basic_analysis(frame, frame.iloc[[0]][['id', 'text']])
        self.assertEqual(len(report['issues']['duplicate_text_ids']), 2)
        self.assertEqual(len(report['issues']['conflicting_text_ids']), 2)
        self.assertEqual(len(report['issues']['duplicate_prompt_model_ids']), 2)
        self.assertEqual(len(report['issues']['train_test_id_overlap']), 1)

    def test_leakage_boundary(self):
        from check_baseline_structure import check_baseline_structure
        check_baseline_structure(self.nb, ROOT)
        # Fit a tiny disjoint synthetic corpus, never the official training set.
        model = self.ns['make_model']()
        model.fit(['alpha beta first', 'alpha beta second', 'gamma delta first', 'gamma delta second'], ['a', 'a', 'b', 'b'])
        word = model.named_steps['features'].transformer_list[0][1]
        before = dict(word.vocabulary_), word.idf_.copy()
        model.predict(['validationonlytoken'])
        self.assertNotIn('validationonlytoken', word.vocabulary_)
        self.assertEqual(before[0], word.vocabulary_)
        np.testing.assert_array_equal(before[1], word.idf_)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--generate', action='store_true')
    parser.add_argument('--execute-eda', action='store_true', help='Execute all EDA code cells in a plain Python namespace, without a Jupyter server')
    args = parser.parse_args()
    ns, _ = load_context()
    train, test, _, input_summary = ns['load_competition_data'](ns['DATA_DIR'])
    directory = ROOT / 'experiments/folds'
    if args.generate:
        for version in ('folds_v1', 'folds_v2'):
            if (directory / f'{version}.csv').exists() or (directory / f'{version}_metadata.json').exists():
                raise FileExistsError(f'{version} already exists; generation refuses replacement')
        for version in ('folds_v1', 'folds_v2'):
            ns['save_fixed_folds'](train, ns['DATA_DIR'] / 'train.csv', directory, version)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FoldTests))
    if not result.wasSuccessful(): raise SystemExit(1)
    report, _, _ = basic_analysis(train, test)
    out = Path(__import__('os').environ.get('NAME_THAT_LLM_S03_REPORT_DIR', str(ROOT / 'artifacts/s03'))); out.mkdir(parents=True, exist_ok=True)
    (out / 'basic-summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    previous_path = ROOT / 'artifacts/s02/input-summary.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else input_summary
    for name, metadata in previous['files'].items():
        assert ns['file_fingerprint'](ns['DATA_DIR'] / name)['sha256'] == metadata['sha256']
    print('S03_PASS; ORIGINAL_SHA256_UNCHANGED')
    if args.execute_eda:
        notebook = json.loads((ROOT / 'notebooks/00_eda.ipynb').read_text(encoding='utf-8'))
        scope = {'display': lambda obj: print(obj.to_string() if hasattr(obj, 'to_string') else obj)}
        count = 0
        with chdir(ROOT / 'notebooks'):
            for cell in notebook['cells']:
                if cell['cell_type'] != 'code':
                    continue
                count += 1
                output = io.StringIO()
                with redirect_stdout(output):
                    exec(compile(''.join(cell['source']), '00_eda.ipynb', 'exec'), scope)
                cell['execution_count'] = count
                cell['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': output.getvalue()}]
        (out / '00_eda.python-executed.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
        assert (out / 'text_statistics.csv').exists() and (out / 'model_length_statistics.csv').exists()
        print('EDA_ALL_CODE_CELLS_EXECUTED', count, '(plain Python execution; no Jupyter frontend/kernel claimed)')
    for version in ('folds_v1', 'folds_v2'):
        metadata = json.loads((directory / f'{version}_metadata.json').read_text(encoding='utf-8'))
        print(version, [{k: v for k, v in row.items() if not k.endswith('class_counts')} for row in metadata['fold_statistics']])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
