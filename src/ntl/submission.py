"""Full fit, ordered runtime prediction and strict Parquet submission."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import validate_test
from .labels import class_order


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

def fit_full_model(factory, train):
    model = factory()
    if hasattr(model.named_steps['classifier'], 'coef_'):
        raise ValueError('Factory must return an unfitted Pipeline')
    model.fit(train['text'].astype(str), train['model'])
    return model


def model_state_fingerprint(model):
    checksum = hashlib.sha256()
    for name, transform in model.named_steps['features'].transformer_list:
        checksum.update(name.encode('utf-8'))
        checksum.update(json.dumps(transform.vocabulary_, sort_keys=True, default=int).encode('utf-8'))
        checksum.update(transform.idf_.tobytes())
    for values in (model.named_steps['classifier'].coef_, model.named_steps['classifier'].intercept_):
        checksum.update(values.tobytes())
    return checksum.hexdigest()


def predict_test(model, test, ns=None, batch_size=256):
    if ns is None:
        ns = {'validate_test': validate_test, 'class_order': class_order}
    ns['validate_test'](test, 'runtime test input')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    order, classes = list(ns['class_order']), list(model.classes_)
    if len(classes) != len(order) or set(classes) != set(order):
        raise ValueError('Model classes differ from official labels')
    columns = [classes.index(label) for label in order]
    predictions, decisions = [], []
    for start in range(0, len(test), batch_size):
        texts = test.iloc[start:start + batch_size]['text'].astype(str)
        predictions.extend(model.predict(texts).tolist())
        decisions.append(model.decision_function(texts)[:, columns])
    pred = np.asarray(predictions, dtype=object)
    scores = np.concatenate(decisions) if decisions else np.empty((0, len(order)), dtype=float)
    if scores.shape != (len(test), len(order)) or not np.isfinite(scores).all():
        raise ValueError('Invalid runtime decision score matrix')
    if len(pred) != len(test) or not set(pred).issubset(order):
        raise ValueError('Invalid prediction labels/count')
    return pred, scores


def build_submission(test, predictions, labels):
    if len(test) != len(predictions):
        raise ValueError('Prediction count differs from test rows')
    result = pd.DataFrame({'id': test['id'].tolist(), 'model': list(predictions)}, dtype='string')
    return validate_submission(result, test, labels)
