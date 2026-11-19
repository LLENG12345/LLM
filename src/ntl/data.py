"""Original-string input contracts and local/Kaggle path resolution."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .labels import OFFICIAL_MODEL_LABELS, class_order
KAGGLE_DATA = Path('/kaggle/input/competitions/name-that-llm')
REQUIRED_DATA_FILES = ('train.csv', 'test.csv')


def resolve_data_dir(explicit_dir=None):
    def missing_files(directory):
        return [name for name in REQUIRED_DATA_FILES if not (directory / name).is_file()]

    if explicit_dir is not None:
        if not str(explicit_dir).strip():
            raise ValueError('NAME_THAT_LLM_DATA_DIR cannot be empty')
        directory = Path(explicit_dir).expanduser().resolve()
        missing = missing_files(directory)
        if missing:
            raise FileNotFoundError(
                f'Explicit data directory {directory} is invalid; missing: {", ".join(missing)}. '
                'No fallback was used.'
            )
        return directory

    start = Path.cwd().resolve()
    project_root = next((
        parent for parent in (start, *start.parents)
        if (parent / 'notebooks' / '01_baseline.ipynb').is_file()
        and (parent / 'requirements.txt').is_file()
    ), None)
    candidates = [KAGGLE_DATA]
    if project_root is not None:
        candidates.extend([project_root / 'data' / 'name-that-llm', project_root / 'data'])
    for directory in candidates:
        if not missing_files(directory):
            return directory.resolve()
    details = '; '.join(
        f'{directory.resolve()}: missing {", ".join(missing_files(directory))}'
        for directory in candidates
    )
    raise FileNotFoundError(
        f'Could not find competition input files. Checked: {details}. '
        'Set NAME_THAT_LLM_DATA_DIR explicitly when outside the project.'
    )

def _locations(frame, mask):
    positions = np.flatnonzero(np.asarray(mask))[:5]
    return [{'record': int(i) + 1,
             'id': repr(frame.iloc[i]['id']) if 'id' in frame else '<missing column>'}
            for i in positions]


def _validate_fields(frame, required, source):
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f'{source}: missing required fields {missing}')
    for field in required:
        values = frame[field]
        null = values.isna()
        if null.any():
            raise ValueError(f'{source}: field {field!r} contains null; records {_locations(frame, null)}')
        non_string = ~values.map(lambda value: isinstance(value, str)).astype(bool)
        if non_string.any():
            raise ValueError(f'{source}: field {field!r} must contain strings; records {_locations(frame, non_string)}')
        blank = values.str.strip().eq('')
        if blank.any():
            raise ValueError(f'{source}: field {field!r} is empty/whitespace; records {_locations(frame, blank)}')
    duplicate = frame['id'].duplicated(keep=False)
    if duplicate.any():
        raise ValueError(f'{source}: duplicate id; records {_locations(frame, duplicate)}')


def validate_train(frame, source='<train>', require_all_labels=True):
    _validate_fields(frame, ('id', 'prompt_id', 'text', 'model'), source)
    actual = set(frame['model'])
    unknown = sorted(actual - set(class_order))
    missing = sorted(set(class_order) - actual) if require_all_labels else []
    if unknown or missing:
        invalid = ~frame['model'].isin(class_order)
        raise ValueError(f'{source}: field model label mismatch; unknown={unknown}; '
                         f'missing={missing}; invalid records={_locations(frame, invalid)}')
    return frame


def validate_test(frame, source='<test>'):
    _validate_fields(frame, ('id', 'text'), source)
    return frame


def read_input_csv(path, kind, require_all_labels=True):
    path = Path(path).expanduser().resolve()
    if kind not in ('train', 'test'):
        raise ValueError(f'{path}: unknown input kind {kind!r}')
    try:
        # Empty fields remain empty; NA/NULL/nan are literal strings, not null markers.
        frame = pd.read_csv(path, dtype='string', keep_default_na=False,
                            na_filter=False, skip_blank_lines=False, encoding='utf-8-sig')
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f'{path}: could not read {kind} CSV: {exc}') from exc
    if kind == 'train':
        return validate_train(frame, str(path), require_all_labels)
    return validate_test(frame, str(path))


def file_fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def validate_sample_submission(sample, test_frame, source='<sample_submission>'):
    # Predictions here are format examples/placeholders, not evaluated predictions.
    missing = sorted({'id', 'model'} - set(sample.columns))
    if missing:
        raise ValueError(f'{source}: missing required fields {missing}')
    _validate_fields(sample, ('id',), source)
    expected, actual = set(test_frame['id']), set(sample['id'])
    if len(sample) != len(test_frame) or actual != expected:
        raise ValueError(f'{source}: id mismatch with test; '
                         f'missing IDs={sorted(expected - actual)[:5]}; '
                         f'extra IDs={sorted(actual - expected)[:5]}; '
                         f'rows={len(sample)}, expected={len(test_frame)}')
    return sample


def load_competition_data(directory, require_all_labels=True):
    directory = Path(directory).resolve()
    train_path, test_path = directory / 'train.csv', directory / 'test.csv'
    train_frame = read_input_csv(train_path, 'train', require_all_labels)
    test_frame = read_input_csv(test_path, 'test')
    summary = {'class_order': list(class_order), 'files': {}}
    for name, frame in (('train.csv', train_frame), ('test.csv', test_frame)):
        summary['files'][name] = {
            **file_fingerprint(directory / name), 'rows': len(frame),
            'columns': frame.columns.tolist(),
            'dtypes': {column: str(dtype) for column, dtype in frame.dtypes.items()}}
    summary['files']['train.csv']['class_counts'] = {
        label: int((train_frame['model'] == label).sum()) for label in class_order}
    summary['files']['train.csv']['prompt_count'] = int(train_frame['prompt_id'].nunique())
    sample_path = directory / 'sample_submission.parquet'
    sample = None
    if sample_path.exists():
        try:
            sample = pd.read_parquet(sample_path)
        except (OSError, ValueError) as exc:
            raise ValueError(f'{sample_path}: could not read sample Parquet: {exc}') from exc
        validate_sample_submission(sample, test_frame, str(sample_path))
        summary['files'][sample_path.name] = {
            **file_fingerprint(sample_path), 'rows': len(sample),
            'columns': sample.columns.tolist(),
            'dtypes': {column: str(dtype) for column, dtype in sample.dtypes.items()},
            'id_order_matches_test': sample['id'].tolist() == test_frame['id'].tolist(),
            'placeholder_value_counts': {str(value): int(count) for value, count in
                                         sample['model'].value_counts(dropna=False).items()},
            'placeholder_null_count': int(sample['model'].isna().sum()),
            'prediction_labels_validated': False}
    else:
        summary['sample_submission_status'] = 'not provided; prediction input does not require sample'
    return train_frame, test_frame, sample, summary


def resolve_submission_data(explicit_dir=None):
    def missing(directory):
        return [name for name in REQUIRED_DATA_FILES if not (directory / name).is_file()]
    if explicit_dir is not None:
        if not str(explicit_dir).strip():
            raise ValueError('NAME_THAT_LLM_DATA_DIR cannot be empty')
        directory = Path(explicit_dir).expanduser().resolve()
        absent = missing(directory)
        if absent:
            raise FileNotFoundError(f'Explicit data directory {directory}: missing {absent}; no fallback used')
        return directory
    if not missing(KAGGLE_DATA):
        return KAGGLE_DATA.resolve()
    # A mount can differ between notebook environments. Do not guess among datasets.
    input_root = Path('/kaggle/input')
    candidates = sorted({p.parent.resolve() for p in input_root.rglob('train.csv')
                         if not missing(p.parent)}) if input_root.is_dir() else []
    if len(candidates) == 1:
        print('DISCOVERED_INPUT_PAIR; training SHA-256 will be checked:', candidates[0])
        return candidates[0]
    raise FileNotFoundError(
        f'Competition inputs unavailable or ambiguous. Official candidate {KAGGLE_DATA}: '
        f'missing {missing(KAGGLE_DATA)}; complete candidate directories={list(map(str, candidates))}. '
        'Attach the official competition data and inspect Input; set NAME_THAT_LLM_DATA_DIR explicitly if needed.')
