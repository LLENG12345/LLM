"""Frozen OOF-only evaluation. This module neither constructs nor fits a model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .io import digest, write_json
from .labels import class_order

ALGORITHM = 'sha256-rank-per-prompt-k2-v1'
# Chosen before any sparse prediction scores were read. Never select among seeds.
SPARSE_SEEDS = tuple(20261119 + 1009 * i for i in range(10))


def _reference(frame):
    needed = ['id', 'prompt_id', 'fold']
    if not set(needed).issubset(frame):
        raise ValueError('Reference requires id, prompt_id, fold')
    ref = frame[needed].copy()
    for col in ('id', 'prompt_id'):
        if not ref[col].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f'Reference {col} must contain nonempty strings')
    if not len(ref) or not ref.id.is_unique:
        raise ValueError('Reference needs nonempty unique IDs')
    if not ref.fold.map(lambda x: isinstance(x, (int, np.integer)) and not isinstance(x, bool) and 1 <= x <= 5).all():
        raise ValueError('Reference fold must be integer 1..5')
    if ref.groupby('prompt_id').fold.nunique().gt(1).any():
        raise ValueError('Reference prompt crosses folds')
    return ref.sort_values(['prompt_id', 'id']).reset_index(drop=True)


def generate_masks(reference, seeds=SPARSE_SEEDS):
    """Blind to labels/predictions; SHA pseudo-random ranking is order independent."""
    ref = _reference(reference)
    if len(set(seeds)) != len(seeds) or not seeds or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Seeds must be distinct nonnegative Python integers')
    rows = []
    for seed in seeds:
        for prompt, group in ref.groupby('prompt_id', sort=True):
            def key(identifier):
                raw = json.dumps([ALGORITHM, seed, prompt, identifier], ensure_ascii=False, separators=(',', ':')).encode('utf-8')
                return hashlib.sha256(raw).digest(), identifier
            ids = sorted(sorted(group.id, key=key)[:2])
            lookup = group.set_index('id')
            rows.extend({'seed': seed, 'id': identifier, 'prompt_id': prompt,
                         'fold': int(lookup.loc[identifier, 'fold'])} for identifier in ids)
    return pd.DataFrame(rows, columns=['seed', 'id', 'prompt_id', 'fold'])


def validate_masks(reference, masks, seeds=SPARSE_SEEDS):
    if list(masks.columns) != ['seed', 'id', 'prompt_id', 'fold']:
        raise ValueError('Mask columns must be seed, id, prompt_id, fold')
    for col in ('seed', 'fold'):
        if not masks[col].map(lambda x: isinstance(x, (int, np.integer)) and not isinstance(x, bool)).all():
            raise ValueError(f'Mask {col} must contain integers')
    expected = generate_masks(reference, seeds)
    try:
        pd.testing.assert_frame_equal(masks.sort_values(['seed', 'prompt_id', 'id']).reset_index(drop=True),
            expected.sort_values(['seed', 'prompt_id', 'id']).reset_index(drop=True), check_dtype=False)
    except AssertionError as exc:
        raise ValueError('Mask IDs/seeds/prompt/fold differ from frozen algorithm') from exc
    return masks


def strict_metrics(truth, predicted):
    truth, predicted = list(truth), list(predicted)
    if not truth or len(truth) != len(predicted):
        raise ValueError('Metrics need equal nonempty arrays')
    if not set(truth).issubset(class_order) or not set(predicted).issubset(class_order):
        raise ValueError('Metrics contain unknown official labels')
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    supports = {label: int((truth == label).sum()) for label in class_order}
    recalls = {label: float((predicted[truth == label] == label).mean()) if supports[label] else None for label in class_order}
    missing = [label for label in class_order if not supports[label]]
    return {'balanced_accuracy': None if missing else float(np.mean(list(recalls.values()))),
            'accuracy': float(accuracy_score(truth, predicted)),
            'macro_f1': float(f1_score(truth, predicted, labels=list(class_order), average='macro', zero_division=0)),
            'n_samples': len(truth), 'n_classes': 18 - len(missing), 'missing_classes': missing,
            'support': supports, 'recall': recalls,
            'ba_status': 'undefined_missing_official_class' if missing else 'all_18_classes'}


def freeze_masks(reference, labels_by_id, train_path, fold_path, directory):
    """Exclusive creation; support counts are measured AFTER label-blind sampling."""
    directory = Path(directory)
    masks = generate_masks(reference)
    validate_masks(reference, masks)
    if not labels_by_id.index.is_unique or set(labels_by_id.index) != set(reference.id):
        raise ValueError('Support labels must match reference IDs exactly')
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / 'masks.csv'
    with target.open('x', encoding='utf-8', newline='') as stream:
        masks.to_csv(stream, index=False, lineterminator='\n')
    support = []
    for seed in SPARSE_SEEDS:
        one = masks[masks.seed == seed]
        for fold in [0, 1, 2, 3, 4, 5]:
            ids = one.id if fold == 0 else one.loc[one.fold == fold, 'id']
            truth = labels_by_id.loc[ids]
            support.append({'seed': seed, 'fold': 'aggregate' if fold == 0 else fold,
                'n_samples': len(ids), 'support': {label: int(truth.eq(label).sum()) for label in class_order}})
    from datetime import datetime, timezone
    meta = {'protocol': 'sparse_group_cv_k2', 'version': 'v1', 'algorithm': ALGORITHM,
        'seeds': list(SPARSE_SEEDS), 'k': 2, 'sampling_uses_labels_or_scores': False,
        'training': 'unchanged folds_v1; unselected validation rows remain held out',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'score_evaluation_started': False,
        'train_sha256': digest(train_path), 'folds_v1_sha256': digest(fold_path),
        'mask_sha256': digest(target), 'class_order': list(class_order), 'support': support}
    write_json(directory / 'metadata.json', meta)
    return meta


def load_masks(reference, train_path, fold_path, directory):
    directory = Path(directory)
    meta = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))
    expected = {'train_sha256': digest(train_path), 'folds_v1_sha256': digest(fold_path),
                'mask_sha256': digest(directory / 'masks.csv'), 'algorithm': ALGORITHM,
                'seeds': list(SPARSE_SEEDS), 'class_order': list(class_order)}
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f'Frozen mask metadata mismatch: {key}')
    masks = pd.read_csv(directory / 'masks.csv', dtype={'id': 'string', 'prompt_id': 'string'}, keep_default_na=False)
    validate_masks(reference, masks)
    return masks, meta
