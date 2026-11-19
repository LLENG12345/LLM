"""Versioned prompt folds; loading never regenerates existing resources."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold
from .labels import class_order
from .data import validate_train, _validate_fields, file_fingerprint


def generate_fold_map(frame, version='folds_v1'):
    validate_train(frame)
    settings = {'folds_v1': {'n_splits': 5, 'shuffle': False, 'random_state': None},
                'folds_v2': {'n_splits': 5, 'shuffle': True, 'random_state': 20261118}}
    if version not in settings:
        raise ValueError(f'Unknown fold version: {version}')
    params = settings[version]
    splitter = GroupKFold(**params)
    result = frame[['id', 'prompt_id']].copy()
    result['fold'] = 0
    for fold, (_, valid) in enumerate(splitter.split(frame['text'], frame['model'],
                                                    groups=frame['prompt_id']), 1):
        result.iloc[valid, result.columns.get_loc('fold')] = fold
    return result.sort_values('id').reset_index(drop=True), params


def validate_fold_map(frame, mapping):
    validate_train(frame)
    required = {'id', 'prompt_id', 'fold'}
    if set(mapping.columns) != required:
        raise ValueError('Fold map must contain exactly id, prompt_id, fold')
    _validate_fields(mapping, ('id', 'prompt_id'), 'fold map')
    missing = set(frame['id']) - set(mapping['id'])
    extra = set(mapping['id']) - set(frame['id'])
    if missing or extra:
        raise ValueError(f'Fold ID coverage mismatch: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}')
    numeric = mapping['fold'].astype('string')
    if not numeric.isin(['1', '2', '3', '4', '5']).all():
        raise ValueError('Invalid fold: expected integer IDs 1..5')
    lookup = mapping.set_index('id').loc[frame['id']]
    if lookup['prompt_id'].tolist() != frame['prompt_id'].tolist():
        raise ValueError('Fold prompt_id mismatch with training data')
    aligned = lookup['fold'].astype(int).to_numpy()
    check = frame[['prompt_id']].copy()
    check['fold'] = aligned
    if (check.groupby('prompt_id')['fold'].nunique() != 1).any():
        raise ValueError('A prompt spans multiple folds')
    stats = []
    for fold in range(1, 6):
        valid = frame.iloc[np.flatnonzero(aligned == fold)]
        fit = frame.iloc[np.flatnonzero(aligned != fold)]
        overlap = set(fit['prompt_id']) & set(valid['prompt_id'])
        if overlap:
            raise ValueError(f'Fold {fold} has prompt leakage')
        if set(valid['model']) != set(class_order) or set(fit['model']) != set(class_order):
            raise ValueError(f'Fold {fold} does not cover all official classes')
        stats.append({'fold': fold, 'n_train': len(fit), 'n_valid': len(valid),
                      'train_prompts': int(fit['prompt_id'].nunique()),
                      'valid_prompts': int(valid['prompt_id'].nunique()),
                      'prompt_overlap': len(overlap),
                      'train_class_counts': {label: int((fit['model'] == label).sum()) for label in class_order},
                      'valid_class_counts': {label: int((valid['model'] == label).sum()) for label in class_order}})
    if sum(row['n_valid'] for row in stats) != len(frame):
        raise ValueError('Validation folds do not partition the training data')
    return aligned, stats


def save_fixed_folds(frame, train_path, directory, version='folds_v1'):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = directory / f'{version}.csv', directory / f'{version}_metadata.json'
    if csv_path.exists() or meta_path.exists():
        raise FileExistsError(f'Refusing to overwrite fold resource: {csv_path}, {meta_path}')
    mapping, params = generate_fold_map(frame, version)
    _, stats = validate_fold_map(frame, mapping)
    # Exclusive creation: an existing resource is never silently replaced.
    with csv_path.open('x', encoding='utf-8', newline='') as stream:
        mapping.to_csv(stream, index=False, lineterminator='\n')
    metadata = {'version': version, 'train_sha256': file_fingerprint(train_path)['sha256'],
                'fold_sha256': file_fingerprint(csv_path)['sha256'],
                'algorithm': 'sklearn.model_selection.GroupKFold', 'parameters': params,
                'sklearn_version': sklearn.__version__, 'n_samples': len(frame),
                'n_prompts': int(frame['prompt_id'].nunique()), 'fold_numbering': '1..5',
                'class_order': list(class_order), 'fold_statistics': stats,
                'generation': 'scripts/check_s03.py --generate; input in official CSV order; output sorted by string id',
                'purpose': 'primary development CV' if version == 'folds_v1' else
                           'predefined stability review; not an independent unseen test set'}
    with meta_path.open('x', encoding='utf-8', newline='') as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return csv_path, meta_path


def load_fixed_folds(frame, train_path, csv_path, meta_path):
    csv_path, meta_path = Path(csv_path), Path(meta_path)
    metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    expected_params = {'folds_v1': {'n_splits': 5, 'shuffle': False, 'random_state': None},
                       'folds_v2': {'n_splits': 5, 'shuffle': True, 'random_state': 20261118}}
    if (metadata['version'] not in expected_params
            or metadata['algorithm'] != 'sklearn.model_selection.GroupKFold'
            or metadata['parameters'] != expected_params[metadata['version']]
            or metadata['fold_numbering'] != '1..5'):
        raise ValueError('Fold metadata algorithm/parameters mismatch')
    if metadata['version'] != csv_path.stem:
        raise ValueError('Fold metadata version mismatch')
    if metadata['train_sha256'] != file_fingerprint(train_path)['sha256']:
        raise ValueError('Training data fingerprint mismatch')
    if metadata['fold_sha256'] != file_fingerprint(csv_path)['sha256']:
        raise ValueError('Fold file fingerprint mismatch')
    if metadata['class_order'] != list(class_order):
        raise ValueError('Fold metadata class_order mismatch')
    mapping = pd.read_csv(csv_path, dtype='string', keep_default_na=False, na_filter=False)
    aligned, stats = validate_fold_map(frame, mapping)
    if (metadata['n_samples'] != len(frame) or metadata['n_prompts'] != frame['prompt_id'].nunique()
            or metadata['fold_statistics'] != stats):
        raise ValueError('Fold metadata statistics mismatch')
    return aligned, metadata


def iter_fixed_splits(frame, train_path, csv_path, meta_path):
    aligned, _ = load_fixed_folds(frame, train_path, csv_path, meta_path)
    for fold in range(1, 6):
        yield np.flatnonzero(aligned != fold), np.flatnonzero(aligned == fold)
