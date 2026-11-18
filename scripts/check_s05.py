"""Strict submission contract and independent saved S05 run verification."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from run_s04 import ROOT, digest, model_config, load_factory


def validate_submission(submission, test, labels):
    if submission.columns.tolist() != ['id', 'model']:
        raise ValueError('Submission columns must be exactly id, model in that order')
    for column in ('id', 'model'):
        values = submission[column]
        if values.isna().any():
            raise ValueError(f'Submission {column} contains null values')
        if not pd.api.types.is_string_dtype(values.dtype) or not all(isinstance(v, str) for v in values):
            raise ValueError(f'Submission {column} must contain strings')
        if values.str.strip().eq('').any():
            raise ValueError(f'Submission {column} contains empty/whitespace strings')
    if submission['id'].duplicated().any():
        raise ValueError('Submission contains duplicate ID')
    missing = set(test['id']) - set(submission['id'])
    extra = set(submission['id']) - set(test['id'])
    if missing or extra or len(submission) != len(test):
        raise ValueError(f'Submission ID coverage mismatch: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}')
    if submission['id'].tolist() != test['id'].tolist():
        raise ValueError('Submission ID order differs from test input')
    unknown = set(submission['model']) - set(labels)
    if unknown:
        raise ValueError(f'Submission contains illegal model labels: {sorted(unknown)}')
    return submission


def write_and_verify_submission(submission, test, labels, path, protected_roots=()):
    path = Path(path).expanduser().resolve()
    if path.name != 'submission.parquet':
        raise ValueError('Output filename must be submission.parquet; sample files are never overwritten')
    if any(path.is_relative_to(Path(root).resolve()) for root in protected_roots):
        raise ValueError(f'Output is inside a protected input/experiment directory: {path}')
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {path}')
    validate_submission(submission, test, labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents a concurrent writer from being overwritten.
    with path.open('xb') as stream:
        submission.to_parquet(stream, index=False)
    if path.stat().st_size == 0:
        raise ValueError(f'Empty submission file: {path}')
    loaded = pd.read_parquet(path)
    validate_submission(loaded, test, labels)
    pd.testing.assert_frame_equal(submission, loaded)
    return loaded


def validate_s05_run(directory, allow_running=False):
    directory = Path(directory).resolve()
    metadata = json.loads((directory / 'run_metadata.json').read_text(encoding='utf-8'))
    assert metadata['status'] == 'validated' or (allow_running and metadata['status'] == 'running')
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    base = Path(metadata['baseline_directory'])
    baseline = json.loads((base / 'config.json').read_text(encoding='utf-8'))
    assert config == baseline
    for name, value in metadata['input_sha256'].items():
        assert digest(name) == value, f'Input changed: {name}'
    for name, value in metadata['baseline_artifact_sha256'].items():
        assert digest(base / name) == value, f'S04 artifact changed: {name}'
    for name, value in metadata['source_sha256'].items():
        assert digest(directory / 'source' / name) == value, f'Source snapshot changed: {name}'
    ns, _, _ = load_factory()
    train = ns['read_input_csv'](Path(metadata['data_dir']) / 'train.csv', 'train')
    test = ns['read_input_csv'](Path(metadata['data_dir']) / 'test.csv', 'test')
    audit = json.loads((directory / 'fit_audit.json').read_text(encoding='utf-8'))
    assert audit['train_ids'] == train['id'].tolist()
    assert audit['train_prompts'] == sorted(set(train['prompt_id']))
    assert audit['fit_features'] == ['text'] and audit['fit_target'] == 'model'
    assert metadata['n_train'] == len(train) == 2699 and metadata['n_prompts'] == train['prompt_id'].nunique() == 150
    assert metadata['n_test'] == len(test)
    model_path = directory / 'model.joblib'
    assert digest(model_path) == metadata['model_sha256']
    # Only the newly generated project model whose recorded hash just passed is loaded.
    model = joblib.load(model_path)
    assert model_config(model) == config['model_parameters']
    assert set(model.classes_) == set(config['class_order'])
    from run_s05 import predict_test, model_state_fingerprint
    fingerprint = model_state_fingerprint(model)
    pred, scores = predict_test(model, test, ns, batch_size=metadata['batch_size'])
    assert model_state_fingerprint(model) == fingerprint == metadata['model_state_after_fit']
    path = Path(metadata['submission_path'])
    assert path.is_file() and path.stat().st_size > 0
    assert digest(path) == metadata['submission_sha256']
    submission = pd.read_parquet(path)
    validate_submission(submission, test, config['class_order'])
    assert submission['model'].tolist() == pred.tolist()
    debug = pd.read_parquet(directory / 'test_decision_scores.parquet')
    assert debug['id'].tolist() == test['id'].tolist()
    np.testing.assert_array_equal(debug[list(config['score_columns'])].to_numpy(), scores)
    warnings = json.loads((directory / 'warnings.json').read_text(encoding='utf-8'))
    assert metadata['warning_count'] == len(warnings)
    validation = json.loads((directory / 'validation_report.json').read_text(encoding='utf-8'))
    assert validation['write_read_equal'] and validation['model_reload_equal']
    assert validation['submission_rows'] == len(test)
    return submission


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('run_directory')
    args = parser.parse_args()
    print(validate_s05_run(args.run_directory).to_string(index=False))
    print('S05_SAVED_RUN_VALIDATION_PASS')
