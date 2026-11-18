"""Saved-run checks plus corruption tests; no training or original-data writes."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import pandas as pd

from check_s04 import validate_run

RUN = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None


class SavedRunTests(unittest.TestCase):
    def test_real_saved_run(self):
        validate_run(RUN)

    def test_corrupted_oof_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='ntl-s04-') as folder:
            copy = Path(folder) / 'run'
            shutil.copytree(RUN, copy)
            path = copy / 'oof_predictions.parquet'
            original = pd.read_parquet(path)
            duplicate = original.copy(); duplicate.loc[1, 'id'] = duplicate.loc[0, 'id']
            invalid_score = original.copy(); invalid_score.loc[0, 'decision_00'] = float('nan')
            invalid_fold = original.copy(); invalid_fold.loc[0, 'fold'] = 6
            for name, frame in [('duplicate', duplicate), ('nonfinite', invalid_score),
                                ('wrong-fold', invalid_fold), ('missing-row', original.iloc[1:])]:
                frame.to_parquet(path, index=False)
                with self.subTest(name=name), self.assertRaises(AssertionError):
                    validate_run(copy)

    def test_false_metrics_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='ntl-s04-') as folder:
            copy = Path(folder) / 'run'; shutil.copytree(RUN, copy)
            path = copy / 'metrics.json'
            metrics = json.loads(path.read_text(encoding='utf-8')); metrics['balanced_accuracy'] = 1.0
            path.write_text(json.dumps(metrics), encoding='utf-8')
            with self.assertRaises(AssertionError):
                validate_run(copy)

    def test_training_overlap_audit_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='ntl-s04-') as folder:
            copy = Path(folder) / 'run'; shutil.copytree(RUN, copy)
            path = copy / 'fold_1/split_audit.json'
            audit = json.loads(path.read_text(encoding='utf-8'))
            audit['train_ids'].append(audit['valid_ids'][0])
            path.write_text(json.dumps(audit), encoding='utf-8')
            with self.assertRaises(AssertionError):
                validate_run(copy)


if __name__ == '__main__':
    if RUN is None: raise SystemExit('Usage: python scripts/test_s04.py RUN_DIRECTORY')
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SavedRunTests))
    raise SystemExit(not result.wasSuccessful())
