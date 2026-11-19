"""Freeze and evaluate sparse validation masks using historical OOF only."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from ntl.data import read_input_csv
from ntl.evaluation import SPARSE_SEEDS, freeze_masks, load_masks, strict_metrics
from ntl.io import digest, write_json
from ntl.labels import class_order
from ntl.splits import load_fixed_folds
from ntl.validation import validate_run
from analyze_s07 import RUNS, align_oof, complementarity

ROOT = Path(__file__).resolve().parents[1]
MASKS = ROOT / 'experiments/evaluation_masks/sparse_k2_v1'
RESULT = ROOT / 'experiments/s07a_protocol_20261119_01'


def reference_data():
    train_path = ROOT / 'data/name-that-llm/train.csv'
    fold_path = ROOT / 'experiments/folds/folds_v1.csv'
    train = read_input_csv(train_path, 'train')
    folds, _ = load_fixed_folds(train, train_path, fold_path, fold_path.with_name('folds_v1_metadata.json'))
    reference = train[['id', 'prompt_id']].copy()
    reference['fold'] = folds
    return train, reference, train_path, fold_path


def evaluate(mask_dir=MASKS, output=RESULT):
    started = datetime.now(timezone.utc).isoformat()
    train, reference, train_path, fold_path = reference_data()
    masks, meta = load_masks(reference, train_path, fold_path, mask_dir)
    if started < meta['created_utc']:
        raise ValueError('Evaluation predates frozen masks')
    # Read scores ONLY after loading the already-frozen mask resource.
    frames, audit = {}, {}
    for name, run_id in RUNS.items():
        folder = ROOT / 'artifacts' / run_id
        validate_run(folder)  # Saved split IDs, scores, metrics, recalls, CM, config/source/data.
        frames[name] = pd.read_parquet(folder / 'oof_predictions.parquet')
        config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
        scores = list(config['score_columns'])
        assert len(frames[name]) == 2699 and len(scores) == 18
        audit[name] = {'run_id': run_id, 'oof_rows': len(frames[name]), 'score_dimensions': len(scores),
            'score_type': 'LinearSVC decision_function margins; aligned official class_order',
            'oof_sha256': digest(folder / 'oof_predictions.parquet'),
            'config_sha256': digest(folder / 'config.json'), 'full_artifact_validation': 'passed'}
    frames = align_oof(frames)
    full, sparse, recalls, coverage, groups, mask_use = [], [], [], [], [], []
    labels_by_id = train.set_index('id').model
    for row in meta['support']:
        subset = masks[masks.seed.eq(row['seed'])]
        if row['fold'] != 'aggregate':
            subset = subset[subset.fold.eq(row['fold'])]
        actual = {label: int(labels_by_id.loc[subset.id].eq(label).sum()) for label in class_order}
        if actual != row['support'] or len(subset) != row['n_samples']:
            raise ValueError('Frozen support metadata differs from official input')
        coverage.append({'seed': row['seed'], 'fold': row['fold'], 'n_samples': len(subset),
                         'classes_present': sum(v > 0 for v in actual.values()),
                         'missing_classes': '|'.join(k for k, v in actual.items() if not v), **actual})
    for name, frame in frames.items():
        metrics = strict_metrics(frame.true_model, frame.predicted_model)
        saved = json.loads((ROOT / 'artifacts' / RUNS[name] / 'metrics.json').read_text(encoding='utf-8'))
        for key in ('balanced_accuracy', 'accuracy', 'macro_f1'):
            np.testing.assert_allclose(metrics[key], saved[key], rtol=0, atol=1e-15)
        full.append({'model': name, 'run_id': RUNS[name], **{k: metrics[k] for k in ('balanced_accuracy', 'accuracy', 'macro_f1')}})
        for label in class_order:
            recalls.append({'model': name, 'seed': 'full', 'label': label, 'support': metrics['support'][label], 'recall': metrics['recall'][label]})
        lookup = frame.set_index('id')
        for seed in SPARSE_SEEDS:
            one = masks[masks.seed == seed]
            subset = lookup.loc[one.id]
            assert subset.prompt_id.tolist() == one.prompt_id.tolist() and subset.fold.tolist() == one.fold.tolist()
            m = strict_metrics(subset.true_model, subset.predicted_model)
            sparse.append({'model': name, 'seed': seed, **{k: m[k] for k in ('balanced_accuracy', 'accuracy', 'macro_f1', 'n_samples', 'n_classes', 'ba_status')}})
            mask_use.append({'model': name, 'seed': seed, 'mask_sha256': meta['mask_sha256'], 'n_samples': len(subset)})
            for label in class_order:
                recalls.append({'model': name, 'seed': seed, 'label': label, 'support': m['support'][label], 'recall': m['recall'][label]})
    full, sparse = pd.DataFrame(full), pd.DataFrame(sparse)
    summary = []
    for name, frame in sparse.groupby('model', sort=False):
        row = full[full.model == name].iloc[0].to_dict()
        for key in ('balanced_accuracy', 'accuracy', 'macro_f1'):
            values = frame[key].to_numpy(dtype=float)
            row[f'sparse_{key}_mean'] = float(values.mean()) if np.isfinite(values).all() else None
            row[f'sparse_{key}_std_ddof0'] = float(values.std(ddof=0)) if np.isfinite(values).all() else None
            row[f'sparse_{key}_min'] = float(values.min()) if np.isfinite(values).all() else None
            row[f'sparse_{key}_max'] = float(values.max()) if np.isfinite(values).all() else None
        row['sparse_minus_full_ba'] = None if row['sparse_balanced_accuracy_mean'] is None else row['sparse_balanced_accuracy_mean'] - row['balanced_accuracy']
        summary.append(row)
    wide = sparse.pivot(index='seed', columns='model', values='balanced_accuracy')
    paired = []
    for a, b in [('char','combo'),('word','combo'),('char','word')]:
        difference = wide[a] - wide[b]
        paired.append({'a': a, 'b': b, 'mean_difference': float(difference.mean()) if difference.notna().all() else None,
            'std_ddof0': float(difference.std(ddof=0)) if difference.notna().all() else None,
            'min': float(difference.min()) if difference.notna().all() else None,
            'max': float(difference.max()) if difference.notna().all() else None,
            'wins': int(difference.gt(0).sum()), 'ties': int(difference.eq(0).sum()),
            'losses': int(difference.lt(0).sum()), 'undefined': int(difference.isna().sum())})
    rankings = []
    for seed, row in wide.iterrows():
        rankings.append({'seed': int(seed), 'ranks': {k: float(v) if pd.notna(v) else None for k,v in row.rank(ascending=False,method='min').items()},
            'descending_ba': row.dropna().sort_values(ascending=False).index.tolist(),
            'ties_present': bool(row.dropna().duplicated().any())})
    for seed in ['full', *SPARSE_SEEDS]:
        ids = frames['combo'].id if seed == 'full' else masks.loc[masks.seed == seed, 'id']
        data = {name: f.set_index('id').loc[ids] for name,f in frames.items()}
        groups.append({'seed': seed, **complementarity(data['combo'].true_model,
            data['word'].predicted_model,data['char'].predicted_model,data['combo'].predicted_model)})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in [('full_metrics',full),('sparse_metrics',sparse),('summary',pd.DataFrame(summary)),
                        ('class_recall',pd.DataFrame(recalls)),('class_coverage',pd.DataFrame(coverage)),
                        ('mask_use',pd.DataFrame(mask_use)),('paired_comparisons',pd.DataFrame(paired))]:
        frame.to_csv(output / f'{name}.csv', index=False, lineterminator='\n')
    write_json(output / 'rankings.json', rankings)
    write_json(output / 'complementarity.json', groups)
    write_json(output / 'historical_audit.json', audit)
    sources = [Path(__file__), ROOT/'scripts/analyze_s07.py', ROOT/'src/ntl/evaluation.py', ROOT/'src/ntl/labels.py']
    for source in sources:
        target = output / 'source' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    write_json(output / 'evaluation_metadata.json', {'evaluation_id': output.name, 'kind': 'historical OOF resampling; not a trained-model experiment',
        'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'new_model_fits': 0, 'fit_guards_enabled_by_cli': True, 'masks_frozen_utc': meta['created_utc'],
        'mask_sha256': meta['mask_sha256'], 'mask_metadata_sha256': digest(Path(mask_dir)/'metadata.json'),
        'train_sha256': digest(train_path), 'folds_v1_sha256': digest(fold_path),
        'class_order': list(class_order), 'source_sha256': {p.relative_to(ROOT).as_posix(): digest(p) for p in sources},
        'primary_protocol': 'full_group_cv', 'sparse_role': 'supplementary candidate only',
        'public_correlation': 'not estimable: one independent public feedback score',
        'uncertainty': '10 overlapping masks describe subsampling variation conditional on fixed OOF; not independent test/fit replicates; no CI calculated'})
    print(pd.DataFrame(summary).to_string(index=False))
    print('Paired:', paired)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--masks', type=Path, default=MASKS)
    parser.add_argument('--output', type=Path, default=RESULT)
    args = parser.parse_args()
    with ExitStack() as stack:
        for owner, method in [(Pipeline,'fit'),(LinearSVC,'fit'),(ExtraTreesClassifier,'fit'),
                              (TfidfVectorizer,'fit'),(TfidfVectorizer,'fit_transform')]:
            stack.enter_context(patch.object(owner, method, side_effect=AssertionError('S07-A forbids model/feature fitting')))
        if args.freeze:
            train, ref, train_path, fold_path = reference_data()
            print(freeze_masks(ref,train.set_index('id').model,train_path,fold_path,args.masks)['mask_sha256'])
        else:
            evaluate(args.masks, args.output)


if __name__ == '__main__':
    main()
