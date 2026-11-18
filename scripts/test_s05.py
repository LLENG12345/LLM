"""Submission contract and real saved-model regression; never fit official data."""
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import psutil

from check_s05 import validate_submission, write_and_verify_submission, validate_s05_run
from run_s04 import ROOT, digest, load_factory, write_json
from run_s05 import build_submission, fit_full_model, model_state_fingerprint, predict_test

RUN = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None


class SubmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns, _, _ = load_factory()
        cls.labels = list(cls.ns['class_order'])
        meta = json.loads((RUN / 'run_metadata.json').read_text(encoding='utf-8'))
        assert meta['status'] == 'validated' and digest(RUN / 'model.joblib') == meta['model_sha256']
        cls.model = joblib.load(RUN / 'model.joblib')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ntl-s05-')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.test = pd.DataFrame({'id': ['0001', '0020'], 'text': ['First text!', 'Other text.']}, dtype='string')
        self.valid = build_submission(self.test, self.labels[:2], self.labels)

    def test_valid_roundtrip_custom_directory_leading_zero(self):
        path = self.base / 'custom' / 'nested' / 'submission.parquet'
        actual = write_and_verify_submission(self.valid, self.test, self.labels, path)
        self.assertEqual(actual['id'].tolist(), ['0001', '0020'])
        self.assertEqual(actual.columns.tolist(), ['id', 'model'])
        pd.testing.assert_frame_equal(actual, self.valid)
        with self.assertRaises(FileExistsError):
            write_and_verify_submission(self.valid, self.test, self.labels, path)

    def test_missing_extra_and_wrong_order_columns(self):
        for frame in (self.valid.drop(columns='model'), self.valid.assign(score=1), self.valid[['model', 'id']]):
            with self.subTest(), self.assertRaisesRegex(ValueError, 'columns'):
                validate_submission(frame, self.test, self.labels)

    def test_duplicate_missing_extra_and_wrong_order_ids(self):
        cases = [self.valid.assign(id=['0001', '0001']), self.valid.iloc[:1],
                 self.valid.assign(id=['0001', 'extra']), self.valid.iloc[::-1],
                 pd.concat([self.valid, self.valid.iloc[[0]].assign(id='extra')])]
        for frame in cases:
            with self.subTest(), self.assertRaises(ValueError):
                validate_submission(frame, self.test, self.labels)

    def test_illegal_labels(self):
        with self.assertRaisesRegex(ValueError, 'illegal model'):
            validate_submission(self.valid.assign(model='not-an-official-model'), self.test, self.labels)

    def test_null_blank_and_nonstring(self):
        for column in ('id', 'model'):
            for value in (None, '', ' \t '):
                frame = self.valid.copy(); frame.loc[0, column] = value
                with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                    validate_submission(frame, self.test, self.labels)
            with self.assertRaisesRegex(ValueError, 'strings'):
                validate_submission(self.valid.assign(**{column: [1, 2]}), self.test, self.labels)

    def test_prediction_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'Prediction count'):
            build_submission(self.test, self.labels[:1], self.labels)

    def test_protected_input_and_sample_paths(self):
        raw = self.base / 'inputs'; raw.mkdir()
        original = raw / 'sample_submission.parquet'
        self.valid.to_parquet(original, index=False)
        before = digest(original)
        for target in (original, raw / 'submission.parquet'):
            with self.subTest(), self.assertRaises(ValueError):
                write_and_verify_submission(self.valid, self.test, self.labels, target, [raw])
        self.assertEqual(before, digest(original))
        self.assertFalse((raw / 'submission.parquet').exists())

    def test_real_prediction_variable_counts_and_no_fit(self):
        fingerprint = model_state_fingerprint(self.model)
        meta = json.loads((RUN / 'run_metadata.json').read_text(encoding='utf-8'))
        train = self.ns['read_input_csv'](Path(meta['data_dir']) / 'train.csv', 'train')
        texts = train['text'].tolist()
        process, measurements = psutil.Process(), []
        with patch.object(self.model, 'fit', side_effect=AssertionError('test cannot fit')):
            for _, transform in self.model.named_steps['features'].transformer_list:
                self.assertNotIn('s05_never_fit_zz', transform.vocabulary_)
            for count in (0, 1, 3, 17, 257, 4000):
                test = pd.DataFrame({'id': pd.Series([f'{i:06}' for i in range(count)], dtype='string'),
                                     'text': pd.Series([texts[i % len(texts)] + ' s05_never_fit_zz' for i in range(count)], dtype='string')})
                rss_before = process.memory_info().rss
                start = time.perf_counter()
                pred, scores = predict_test(self.model, test, self.ns, batch_size=128)
                seconds = time.perf_counter() - start
                self.assertEqual(scores.shape, (count, 18))
                submission = build_submission(test, pred, self.labels)
                write_and_verify_submission(submission, test, self.labels,
                    self.base / str(count) / 'submission.parquet')
                measurements.append({'rows': count, 'batch_size': 128,
                    'total_characters': int(test['text'].str.len().sum()),
                    'predict_and_decision_seconds': seconds, 'rss_before_bytes': rss_before,
                    'rss_after_bytes': process.memory_info().rss,
                    'process_lifetime_peak_wset_bytes': getattr(process.memory_info(), 'peak_wset', None)})
        self.assertEqual(fingerprint, model_state_fingerprint(self.model))
        write_json(ROOT / 'artifacts/s05/capacity_report.json', {
            'run_id': meta['run_id'], 'measurements': measurements,
            'input': 'Training texts cycled with synthetic unique IDs and an unseen suffix; capacity only, no quality estimate',
            'fit_disabled': True, 'fitted_state_unchanged': True,
            'memory_note': 'RSS snapshots and process-lifetime peak, not isolated per-size peak'})

    def test_fit_receives_only_training_text_and_labels(self):
        captured = {}
        class Spy:
            named_steps = {'classifier': SimpleNamespace()}
            def fit(self, x, y):
                captured['x'], captured['y'] = x.tolist(), y.tolist()
        train = self.test.assign(prompt_id=['p0', 'p1'], model=self.labels[:2])
        fit_full_model(Spy, train)
        self.assertEqual(captured, {'x': train['text'].tolist(), 'y': train['model'].tolist()})

    def test_model_reload_predictions_and_scores(self):
        path = self.base / 'own-model.joblib'; joblib.dump(self.model, path, compress=3)
        recorded = digest(path); self.assertEqual(recorded, digest(path))
        own = joblib.load(path)
        before, scores = predict_test(self.model, self.test, self.ns)
        after, other = predict_test(own, self.test, self.ns)
        np.testing.assert_array_equal(before, after); np.testing.assert_array_equal(scores, other)

    def test_score_alignment_is_explicit(self):
        labels = self.labels
        class ReverseClasses:
            classes_ = np.asarray(labels[::-1])
            def predict(self, texts): return np.asarray([labels[0]] * len(texts))
            def decision_function(self, texts): return np.tile(np.arange(18), (len(texts), 1))
        _, scores = predict_test(ReverseClasses(), self.test, self.ns)
        np.testing.assert_array_equal(scores[0], np.arange(18)[::-1])

    def test_runtime_does_not_depend_on_sample_ids(self):
        self.test.to_csv(self.base / 'test.csv', index=False)
        pd.DataFrame({'id': ['outdated-sample-id'], 'model': ['placeholder']}).to_parquet(
            self.base / 'sample_submission.parquet', index=False)
        runtime = self.ns['read_input_csv'](self.base / 'test.csv', 'test')
        pred, _ = predict_test(self.model, runtime, self.ns)
        write_and_verify_submission(build_submission(runtime, pred, self.labels), runtime, self.labels,
                                    self.base / 'output' / 'submission.parquet')

    def test_real_saved_run(self):
        validate_s05_run(RUN)


if __name__ == '__main__':
    if RUN is None: raise SystemExit('Usage: python scripts/test_s05.py RUN_DIRECTORY')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SubmissionTests))
    raise SystemExit(not result.wasSuccessful())
