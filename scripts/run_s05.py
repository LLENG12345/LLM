"""Full fit and submission candidate from the shared verified baseline; no tuning."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
import psutil

from _common import ROOT, load_factory
from ntl.io import digest, write_json
from ntl.models import model_config
from ntl.submission import fit_full_model, model_state_fingerprint, predict_test, build_submission
from ntl.submission import validate_submission, write_and_verify_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--data-dir')
    parser.add_argument('--output-dir', help='New run directory; existing paths are refused')
    parser.add_argument('--submission-path', help='Optional explicit path ending in submission.parquet')
    parser.add_argument('--baseline-dir', default=str(ROOT / 'artifacts/s04_baseline_v1_20261118_01'))
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    if not args.run_id.replace('_', '').replace('-', '').isalnum():
        parser.error('Invalid run-id')
    if args.batch_size <= 0: parser.error('batch-size must be positive')
    if args.data_dir is not None:
        os.environ['NAME_THAT_LLM_DATA_DIR'] = args.data_dir
    ns, nb, factory_source = load_factory()
    baseline = Path(args.baseline_dir).expanduser().resolve()
    base_meta = json.loads((baseline / 'run_metadata.json').read_text(encoding='utf-8'))
    config = json.loads((baseline / 'config.json').read_text(encoding='utf-8'))
    assert base_meta['status'] == 'validated' and base_meta['mode'] == 'formal'
    assert model_config(ns['make_model']()) == config['model_parameters']
    assert list(ns['class_order']) == config['class_order']
    assert ns['RANDOM_SEED'] == config['random_seed']
    # Training contract and version are fixed; test input is runtime-dependent.
    train_path, test_path = ns['DATA_DIR'] / 'train.csv', ns['DATA_DIR'] / 'test.csv'
    expected_train_sha = next(value for name, value in base_meta['input_sha256'].items() if Path(name).name == 'train.csv')
    assert digest(train_path) == expected_train_sha
    train = ns['read_input_csv'](train_path, 'train')
    test = ns['read_input_csv'](test_path, 'test')
    assert len(train) == 2699 and train['prompt_id'].nunique() == 150
    protected = [ns['DATA_DIR'], baseline, ROOT / 'experiments/folds']
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else ROOT / 'artifacts' / args.run_id
    if any(out.is_relative_to(Path(p).resolve()) for p in protected):
        raise ValueError('Run output cannot be inside input, S04 artifact or fold directories')
    destination = Path(args.submission_path).expanduser().resolve() if args.submission_path else out / 'submission.parquet'
    if destination.exists() or any(destination.is_relative_to(Path(p).resolve()) for p in protected):
        raise ValueError('Submission output exists or is inside a protected directory')
    if destination.name != 'submission.parquet': raise ValueError('Submission filename must be submission.parquet')
    out.mkdir(parents=True, exist_ok=False)
    metadata = {'run_id': args.run_id, 'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'baseline_run_id': base_meta['run_id'], 'baseline_directory': str(baseline),
                'data_dir': str(ns['DATA_DIR']), 'n_train': len(train), 'n_test': len(test),
                'n_prompts': int(train['prompt_id'].nunique()), 'batch_size': args.batch_size,
                'submission_path': str(destination), 'python': platform.python_version(), 'python_executable': sys.executable,
                'versions': {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'pyarrow', 'joblib', 'scipy')},
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
                'input_sha256': {str(p.resolve()): digest(p) for p in [*ns['DATA_DIR'].glob('*'), * (ROOT / 'experiments/folds').glob('*')] if p.is_file()},
                'baseline_artifact_sha256': {str(p.relative_to(baseline)): digest(p) for p in baseline.rglob('*') if p.is_file()},
                'source_sha256': {}, 'execution': 'plain Python; shared ntl modules; no Jupyter repair claimed',
                'purpose': 'full fit of unchanged shared baseline',
                'sample_submission_policy': 'S02 local format example only; runtime predictions never depend on sample IDs or placeholders',
                'serialization_note': 'Only load this project-generated trusted joblib; compatible Python/sklearn/numpy versions are required'}
    source_dir = out / 'source'; source_dir.mkdir()
    for source in [ROOT / 'notebooks/01_baseline.ipynb', *sorted((ROOT / 'scripts').glob('*.py')), ROOT / 'requirements.txt', ROOT / 'requirements-s01.txt', ROOT / 'pyproject.toml', *sorted((ROOT / 'src/ntl').glob('*.py'))]:
        dest = source_dir / source.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest); metadata['source_sha256'][str(source.relative_to(ROOT))] = digest(source)
    write_json(out / 'config.json', config); write_json(out / 'run_metadata.json', metadata)
    write_json(out / 'fit_audit.json', {'train_ids': train['id'].tolist(), 'train_prompts': sorted(set(train['prompt_id'])),
                                      'fit_features': ['text'], 'fit_target': 'model'})
    try:
        process = psutil.Process(); memory_before = process.memory_info().rss
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            start = time.perf_counter(); model = fit_full_model(ns['make_model'], train)
            metadata['fit_seconds'] = time.perf_counter() - start
            assert model_config(model) == config['model_parameters']
            assert set(model.classes_) == set(config['class_order'])
            metadata['model_state_after_fit'] = model_state_fingerprint(model)
            start = time.perf_counter(); pred, scores = predict_test(model, test, ns, args.batch_size)
            metadata['inference_seconds'] = time.perf_counter() - start
            assert model_state_fingerprint(model) == metadata['model_state_after_fit']
            submission = build_submission(test, pred, ns['class_order'])
            write_and_verify_submission(submission, test, ns['class_order'], destination, protected)
            model_path = out / 'model.joblib'; joblib.dump(model, model_path, compress=3)
            metadata['model_sha256'] = digest(model_path)
            reloaded = joblib.load(model_path)
            after_pred, after_scores = predict_test(reloaded, test, ns, args.batch_size)
            np.testing.assert_array_equal(pred, after_pred); np.testing.assert_array_equal(scores, after_scores)
            assert model_state_fingerprint(reloaded) == metadata['model_state_after_fit']
        recorded = [{'category': w.category.__name__, 'message': str(w.message), 'filename': w.filename, 'lineno': w.lineno} for w in caught]
        write_json(out / 'warnings.json', recorded)
        for warning in recorded: print('WARNING', json.dumps(warning, ensure_ascii=False), flush=True)
        metadata.update(warning_count=len(recorded), model_path=str(model_path), submission_sha256=digest(destination),
                        rss_before_bytes=memory_before, rss_after_bytes=process.memory_info().rss,
                        process_peak_wset_bytes=getattr(process.memory_info(), 'peak_wset', None))
        debug = pd.DataFrame(scores, columns=list(config['score_columns']))
        debug.insert(0, 'id', test['id'].tolist()); debug.to_parquet(out / 'test_decision_scores.parquet', index=False)
        write_json(out / 'validation_report.json', {'write_read_equal': True, 'model_reload_equal': True,
                    'submission_rows': len(test), 'decision_score_columns': config['score_columns'],
                    'score_type': 'uncalibrated decision scores, not probability'})
        write_json(out / 'run_metadata.json', metadata)
        from check_s05 import validate_s05_run
        validate_s05_run(out, allow_running=True)
        metadata.update(status='validated', completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(out / 'run_metadata.json', metadata)
        print(submission.to_string(index=False), flush=True)
        print('S05_VALIDATED', out, flush=True)
    except BaseException:
        metadata.update(status='failed', error=traceback.format_exc())
        write_json(out / 'run_metadata.json', metadata)
        (out / 'failure.log').write_text(metadata['error'], encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
