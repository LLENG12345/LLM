"""Exercise Notebook input contracts without executing model training cells."""
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NB = json.loads((ROOT / 'notebooks/01_baseline.ipynb').read_text(encoding='utf-8'))
NS = {'Path': Path, 'pd': pd, 'np': __import__('numpy')}
INPUT_SOURCE = ''.join(NB['cells'][2]['source'])
exec(compile(INPUT_SOURCE.split('train, test, sample_submission, data_summary =')[0],
             'notebook_input_functions', 'exec'), NS)
ORDER = NS['class_order']


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ntl-s02-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def train(self):
        return pd.DataFrame({'id': ['001', '1'], 'prompt_id': ['0002', '0001'],
                             'text': ['  MiXeD **Bold**!\n  A—B  ', 'NULL'],
                             'model': [ORDER[0], ORDER[1]]}, dtype='string')

    def read(self, frame, kind='train', full=False):
        path = self.base / f'{kind}.csv'
        frame.to_csv(path, index=False)
        return NS['read_input_csv'](path, kind, require_all_labels=full)

    def reject(self, frame, phrase, kind='train', full=False):
        with self.assertRaisesRegex(ValueError, phrase) as caught:
            self.read(frame, kind, full)
        self.assertIn(os.path.normcase(str(self.base.resolve())),
                      os.path.normcase(str(caught.exception)))

    def test_valid_small_train_and_string_ids(self):
        frame = self.train()
        pd.testing.assert_frame_equal(frame, self.read(frame))
        self.assertEqual(self.read(frame)['id'].tolist(), ['001', '1'])

    def test_valid_test_only_id_text(self):
        frame = self.train()[['id', 'text']]
        pd.testing.assert_frame_equal(frame, self.read(frame, 'test'))

    def test_missing_fields(self):
        self.reject(self.train().drop(columns='prompt_id'), 'missing.*prompt_id')
        self.reject(self.train()[['id']], 'missing.*text', 'test')

    def test_true_null_fields(self):
        for field in ('id', 'prompt_id', 'text', 'model'):
            frame = self.train()
            frame.loc[0, field] = pd.NA
            with self.assertRaisesRegex(ValueError, f'{field!r} contains null.*record'):
                NS['validate_train'](frame, 'null-fixture', False)

    def test_empty_and_whitespace_fields(self):
        for field in ('id', 'prompt_id', 'text', 'model'):
            for value in ('', ' \t\n '):
                frame = self.train()
                frame.loc[0, field] = value
                self.reject(frame, f'{field!r} is empty/whitespace.*record')

    def test_duplicate_ids(self):
        frame = self.train()
        frame.loc[1, 'id'] = '001'
        self.reject(frame, 'duplicate id.*record')

    def test_unknown_label_with_location(self):
        frame = self.train()
        frame.loc[0, 'model'] = 'GPT-INVALID'
        self.reject(frame, 'unknown=.*GPT-INVALID.*invalid records=.*001')

    def test_full_label_coverage(self):
        self.reject(self.train(), 'missing=\\[', full=True)
        frame = pd.concat([self.train().iloc[[0]]] * len(ORDER), ignore_index=True)
        frame['id'] = [f'{i:03}' for i in range(len(ORDER))]
        frame['model'] = list(ORDER)
        self.assertEqual(len(self.read(frame, full=True)), len(ORDER))
        frame.loc[0, 'model'] = 'unknown'
        self.reject(frame, 'unknown=.*unknown.*missing=', full=True)

    def test_literal_na_strings(self):
        for value in ('NA', 'NULL', 'null', 'NaN', 'N/A'):
            frame = self.train()[['id', 'text']].copy()
            frame['text'] = value
            self.assertEqual(self.read(frame, 'test')['text'].tolist(), [value, value])

    def test_text_and_order_are_unchanged(self):
        frame = self.train().iloc[::-1].reset_index(drop=True)
        frame.loc[0, 'text'] = '  MiXeD\t**Markdown**!\n  e\u0301 é  '
        actual = self.read(frame)
        pd.testing.assert_frame_equal(frame, actual)
        with (self.base / 'train.csv').open(encoding='utf-8', newline='') as stream:
            raw = list(csv.DictReader(stream))
        self.assertEqual(actual['text'].tolist(), [row['text'] for row in raw])
        self.assertEqual(actual['id'].tolist(), [row['id'] for row in raw])

    def test_validation_does_not_mutate(self):
        frame = self.train()
        before = frame.copy(deep=True)
        NS['validate_train'](frame, 'valid-fixture', False)
        pd.testing.assert_frame_equal(frame, before)

    def test_class_order_matches_documented_official_set(self):
        import re
        section = (ROOT / 'docs/工作计划.md').read_text(encoding='utf-8').split(
            '### 官方标签清单')[1].split('## 二')[0]
        documented = set(re.findall(r'`([^`]+)`', section)) - {'class_order'}
        self.assertEqual(set(ORDER), documented)
        self.assertEqual(len(ORDER), 18)
        self.assertEqual(ORDER, tuple(sorted(ORDER)))

    def test_sample_contract_and_placeholders(self):
        test = self.train()[['id', 'text']]
        sample = pd.DataFrame({'id': test['id'], 'model': ['placeholder', None]})
        path = self.base / 'sample.parquet'
        sample.to_parquet(path, index=False)
        loaded = pd.read_parquet(path)
        NS['validate_sample_submission'](loaded, test, str(path))
        NS['validate_sample_submission'](loaded.iloc[::-1], test, str(path))
        for broken, phrase in ((loaded.drop(columns='model'), 'missing.*model'),
                               (loaded.assign(id=['wrong', '1']), 'id mismatch'),
                               (loaded.assign(id=['001', '001']), 'duplicate id'),
                               (loaded.assign(id=[1, 2]), 'must contain strings')):
            with self.assertRaisesRegex(ValueError, phrase):
                NS['validate_sample_submission'](broken, test, str(path))

    def test_fingerprint_is_raw_file_hash(self):
        path = self.base / 'bytes.csv'
        path.write_bytes(b'id,text\r\n001,NA\r\n')
        first = NS['file_fingerprint'](path)
        self.assertEqual(first, NS['file_fingerprint'](path))
        self.assertEqual(first['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())


def check_official():
    checkpoint = ROOT / 'artifacts/s02/before.json'
    before = json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else None
    directory = ROOT / 'data/name-that-llm'
    hashes_before = {p.name: NS['file_fingerprint'](p)['sha256']
                     for p in directory.iterdir() if p.is_file()}
    train, test, sample, summary = NS['load_competition_data'](directory)
    # Independent CSV reader verifies decoded text and order, not a second pandas read.
    for name, frame in (('train.csv', train), ('test.csv', test)):
        with (directory / name).open(encoding='utf-8-sig', newline='') as stream:
            raw = list(csv.DictReader(stream))
        for column in frame.columns:
            assert frame[column].tolist() == [row[column] for row in raw], (name, column)
    assert hashes_before == {p.name: NS['file_fingerprint'](p)['sha256']
                             for p in directory.iterdir() if p.is_file()}
    if before:
        assert hashes_before == {Path(p).name: value for p, value in before['files'].items()}
        assert NB['cells'][1] == before['notebook']['cells'][1], 'S01 resolver changed'
        from check_baseline_structure import check_baseline_structure
        check_baseline_structure(NB, ROOT)
    assert sample is not None
    output = ROOT / 'artifacts/s02/input-summary.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('OFFICIAL_INPUT_PASS', len(train), len(test), len(sample))
    print('RAW_TEXT_ORDER_AND_SHA256_UNCHANGED; S01_PATH_AND_ALGORITHM_UNCHANGED')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(InputTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    check_official()
