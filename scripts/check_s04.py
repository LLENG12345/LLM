"""Independently verify saved OOF, split provenance and recomputed metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, recall_score

from run_s04 import ROOT, config_hash, digest, load_factory, scores_summary


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def validate_run(out, allow_running=False):
    out = Path(out)
    metadata = json.loads((out / 'run_metadata.json').read_text(encoding='utf-8'))
    assert metadata['status'] == 'validated' or (allow_running and metadata['status'] == 'running')
    config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
    assert config_hash(config) == metadata['config_sha256']
    for name, expected in metadata['input_sha256'].items():
        assert digest(ROOT / name) == expected, f'Changed input: {name}'
    for name, expected in metadata['source_sha256'].items():
        assert digest(out / 'source' / name) == expected, f'Changed snapshot: {name}'
    ns, _, _ = load_factory()
    train_path = Path(metadata['data_dir']) / 'train.csv'
    train = ns['read_input_csv'](train_path, 'train')
    order = list(ns['class_order'])
    assert config['class_order'] == order and len(order) == 18
    columns = list(config['score_columns'])
    assert list(config['score_columns'].values()) == order and len(columns) == 18
    aligned, _ = ns['load_fixed_folds'](train, train_path,
        ROOT / 'experiments/folds/folds_v1.csv', ROOT / 'experiments/folds/folds_v1_metadata.json')
    expected_folds = [1] if metadata['mode'] == 'smoke' else list(range(1, 6))
    assert sorted(int(p.name.split('_')[1]) for p in out.glob('fold_*') if p.is_dir()) == expected_folds
    frames, fold_records = [], []
    for fold in expected_folds:
        folder = out / f'fold_{fold}'
        frame = pd.read_parquet(folder / 'oof.parquet')
        metric = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        audit = json.loads((folder / 'split_audit.json').read_text(encoding='utf-8'))
        valid, fit = train.iloc[np.flatnonzero(aligned == fold)], train.iloc[np.flatnonzero(aligned != fold)]
        assert audit['train_ids'] == fit['id'].tolist() and audit['valid_ids'] == valid['id'].tolist()
        assert set(audit['train_prompts']) == set(fit['prompt_id'])
        assert set(audit['valid_prompts']) == set(valid['prompt_id'])
        assert set(audit['train_prompts']).isdisjoint(audit['valid_prompts'])
        assert metric['config_sha256'] == config_hash(config['model_parameters'])
        assert set(metric['model_classes']) == set(order) and len(metric['model_classes']) == 18
        assert metric['fold'] == fold and metric['n_train'] == len(fit) and metric['n_valid'] == len(valid)
        assert metric['train_prompts'] == fit['prompt_id'].nunique() and metric['valid_prompts'] == valid['prompt_id'].nunique()
        assert frame['id'].tolist() == valid['id'].tolist()
        assert frame['prompt_id'].tolist() == valid['prompt_id'].tolist()
        assert frame['true_model'].tolist() == valid['model'].tolist()
        assert frame['fold'].eq(fold).all()
        assert not frame.isna().any().any() and not frame['id'].duplicated().any()
        assert set(frame['predicted_model']).issubset(order)
        assert np.isfinite(frame[columns].to_numpy()).all()
        assert frame['predicted_model'].tolist() == np.asarray(order)[frame[columns].to_numpy().argmax(axis=1)].tolist()
        for name, value in scores_summary(frame.true_model, frame.predicted_model, order).items():
            close(metric[name], value)
        close([metric['recall'][label] for label in order],
              recall_score(frame.true_model, frame.predicted_model, labels=order, average=None, zero_division=0))
        assert metric['warning_count'] == len(json.loads((folder / 'warnings.json').read_text(encoding='utf-8')))
        frames.append(frame); fold_records.append(metric)
    assert len(json.loads((out / 'warnings.json').read_text(encoding='utf-8'))) == sum(m['warning_count'] for m in fold_records)
    if metadata['mode'] == 'smoke':
        assert not (out / 'oof_predictions.parquet').exists()
        return
    oof = pd.read_parquet(out / 'oof_predictions.parquet')
    assert len(oof) == len(train) == 2699 and oof['id'].is_unique
    assert oof['id'].tolist() == train['id'].tolist()
    assert oof['true_model'].tolist() == train['model'].tolist()
    assert oof['prompt_id'].tolist() == train['prompt_id'].tolist()
    np.testing.assert_array_equal(oof['fold'].to_numpy(), aligned)
    expected = pd.concat(frames, ignore_index=True).set_index('id').loc[train['id']].reset_index()
    pd.testing.assert_frame_equal(oof, expected)
    assert not oof.isna().any().any() and np.isfinite(oof[columns].to_numpy()).all()
    metrics = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
    for name, value in scores_summary(oof.true_model, oof.predicted_model, order).items(): close(metrics[name], value)
    close(metrics['mean_fold_balanced_accuracy'], np.mean([m['balanced_accuracy'] for m in fold_records]))
    close(metrics['std_fold_balanced_accuracy_ddof0'], np.std([m['balanced_accuracy'] for m in fold_records], ddof=0))
    for key in ('fit', 'predict', 'decision', 'inference'):
        close(metrics[f'total_{key}_seconds'], sum(m[f'{key}_seconds'] for m in fold_records))
    saved_folds = pd.read_csv(out / 'fold_metrics.csv')
    assert saved_folds.fold.tolist() == expected_folds
    close(saved_folds.balanced_accuracy, [m['balanced_accuracy'] for m in fold_records])
    report = pd.read_csv(out / 'classification_report.csv', index_col=0)
    computed = pd.DataFrame(classification_report(oof.true_model, oof.predicted_model,
        labels=order, output_dict=True, zero_division=0)).T.loc[order]
    assert report.index.tolist() == order
    close(report.to_numpy(), computed.to_numpy())
    close(report['recall'].mean(), metrics['balanced_accuracy'])
    cm = pd.read_csv(out / 'confusion_matrix.csv', index_col=0)
    norm = pd.read_csv(out / 'confusion_matrix_normalized.csv', index_col=0)
    assert cm.index.tolist() == cm.columns.tolist() == order
    assert norm.index.tolist() == norm.columns.tolist() == order
    expected_cm = confusion_matrix(oof.true_model, oof.predicted_model, labels=order)
    np.testing.assert_array_equal(cm.to_numpy(), expected_cm)
    assert cm.to_numpy().sum() == len(train)
    np.testing.assert_array_equal(cm.sum(axis=1).to_numpy(), train['model'].value_counts().reindex(order).to_numpy())
    close(norm.to_numpy(), expected_cm / expected_cm.sum(axis=1, keepdims=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('run_directory')
    args = parser.parse_args(); validate_run(args.run_directory)
    print('S04_SAVED_RUN_VALIDATION_PASS', args.run_directory)
