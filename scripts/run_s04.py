"""Run the Notebook's unchanged baseline factory on fixed v1; no full-data fit/test prediction."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)

from check_s03 import ROOT, load_context


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalise(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [normalise(item) for item in value]
    if isinstance(value, dict):
        return {str(key): normalise(item) for key, item in value.items()}
    if isinstance(value, type):
        return value.__module__ + '.' + value.__name__
    return repr(value)


def model_config(model):
    return normalise(model.get_params(deep=True))


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def load_factory():
    ns, nb = load_context()
    source = ''.join(nb['cells'][3]['source'])
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'make_model')
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'notebook_factory', 'exec'), ns)
    return ns, nb, ast.get_source_segment(source, node)


def scores_summary(true, pred, order):
    return {'balanced_accuracy': balanced_accuracy_score(true, pred),
            'accuracy': accuracy_score(true, pred),
            'macro_f1': f1_score(true, pred, labels=order, average='macro', zero_division=0)}


def run_fold(ns, train, fit, valid, fold, out, expected_config):
    model = ns['make_model']()
    assert not hasattr(model.named_steps['classifier'], 'coef_')
    assert model_config(model) == expected_config
    fit_frame, valid_frame = train.iloc[fit], train.iloc[valid]
    assert set(fit_frame['prompt_id']).isdisjoint(valid_frame['prompt_id'])
    folder = out / f'fold_{fold}'; folder.mkdir()
    write_json(folder / 'split_audit.json', {
        'train_ids': fit_frame['id'].tolist(), 'valid_ids': valid_frame['id'].tolist(),
        'train_prompts': sorted(set(fit_frame['prompt_id'])),
        'valid_prompts': sorted(set(valid_frame['prompt_id']))})
    process = psutil.Process(); rss_before = process.memory_info().rss
    cpu_before = sum(process.cpu_times()[:2])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        start = time.perf_counter()
        model.fit(fit_frame['text'].astype(str), fit_frame['model'])
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        pred = model.predict(valid_frame['text'].astype(str))
        predict_seconds = time.perf_counter() - start
        start = time.perf_counter()
        raw_scores = model.decision_function(valid_frame['text'].astype(str))
        decision_seconds = time.perf_counter() - start
    classes = list(model.classes_); order = list(ns['class_order'])
    assert len(classes) == len(order) == 18 and set(classes) == set(order)
    scores = raw_scores[:, [classes.index(label) for label in order]]
    assert scores.shape == (len(valid), len(order)) and np.isfinite(scores).all()
    assert set(pred).issubset(order)
    assert np.array_equal(np.asarray(order)[scores.argmax(axis=1)], pred)
    frame = valid_frame[['id', 'prompt_id']].copy()
    frame['fold'] = fold
    frame['true_model'] = valid_frame['model'].to_numpy()
    frame['predicted_model'] = pred
    for i in range(len(order)):
        frame[f'decision_{i:02}'] = scores[:, i]
    frame.to_parquet(folder / 'oof.parquet', index=False)
    captured = [{'fold': fold, 'category': w.category.__name__, 'message': str(w.message),
                 'filename': w.filename, 'lineno': w.lineno} for w in caught]
    memory = process.memory_info()
    metric = {'fold': fold, 'n_train': len(fit), 'n_valid': len(valid),
              'train_prompts': fit_frame['prompt_id'].nunique(),
              'valid_prompts': valid_frame['prompt_id'].nunique(),
              **scores_summary(valid_frame['model'], pred, order),
              'fit_seconds': fit_seconds, 'predict_seconds': predict_seconds,
              'decision_seconds': decision_seconds,
              'inference_seconds': predict_seconds + decision_seconds,
              'cpu_seconds': sum(process.cpu_times()[:2]) - cpu_before,
              'rss_before_bytes': rss_before, 'rss_after_bytes': memory.rss,
              'process_peak_wset_bytes': getattr(memory, 'peak_wset', None),
              'warning_count': len(captured), 'config_sha256': config_hash(expected_config),
              'model_classes': classes, 'n_iter': int(model.named_steps['classifier'].n_iter_),
              'recall': dict(zip(order, recall_score(valid_frame['model'], pred, labels=order,
                                                    average=None, zero_division=0).tolist()))}
    # pandas/numpy count scalars are converted before JSON encoding.
    metric['train_prompts'] = int(metric['train_prompts']); metric['valid_prompts'] = int(metric['valid_prompts'])
    write_json(folder / 'metrics.json', metric)
    write_json(folder / 'warnings.json', captured)
    for warning in captured:
        print('WARNING', json.dumps(warning, ensure_ascii=False), flush=True)
    print(f'FOLD {fold} COMPLETE BA={metric["balanced_accuracy"]:.6f} fit={fit_seconds:.2f}s inference={metric["inference_seconds"]:.2f}s', flush=True)
    return frame, metric, captured


def finalise(out, train, frames, folds, order):
    oof = pd.concat(frames, ignore_index=True).set_index('id').loc[train['id']].reset_index()
    oof.to_parquet(out / 'oof_predictions.parquet', index=False)
    report = pd.DataFrame(classification_report(oof.true_model, oof.predicted_model,
                          labels=order, output_dict=True, zero_division=0)).T.loc[order]
    report.index.name = 'model'; report.to_csv(out / 'classification_report.csv')
    report[['recall']].to_csv(out / 'class_recall.csv')
    cm = confusion_matrix(oof.true_model, oof.predicted_model, labels=order)
    for name, matrix in [('confusion_matrix.csv', cm),
                         ('confusion_matrix_normalized.csv', cm / cm.sum(axis=1, keepdims=True))]:
        pd.DataFrame(matrix, index=order, columns=order).to_csv(out / name, index_label='true_model')
    pd.DataFrame([{k: v for k, v in metric.items() if k not in ('recall', 'model_classes')}
                  for metric in folds]).to_csv(out / 'fold_metrics.csv', index=False)
    metrics = {**scores_summary(oof.true_model, oof.predicted_model, order),
               'mean_fold_balanced_accuracy': float(np.mean([f['balanced_accuracy'] for f in folds])),
               'std_fold_balanced_accuracy_ddof0': float(np.std([f['balanced_accuracy'] for f in folds], ddof=0)),
               'std_definition': 'population standard deviation across five folds, ddof=0',
               'total_fit_seconds': sum(f['fit_seconds'] for f in folds),
               'total_predict_seconds': sum(f['predict_seconds'] for f in folds),
               'total_decision_seconds': sum(f['decision_seconds'] for f in folds),
               'total_inference_seconds': sum(f['inference_seconds'] for f in folds),
               'warning_count': sum(f['warning_count'] for f in folds), 'oof_rows': len(oof),
               'confusion_matrix_png': 'not generated: matplotlib is not installed'}
    if importlib.util.find_spec('matplotlib'):
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        fig, ax = plt.subplots(figsize=(15, 13))
        ConfusionMatrixDisplay(cm, display_labels=order).plot(ax=ax, xticks_rotation=90, include_values=False)
        fig.tight_layout(); fig.savefig(out / 'confusion_matrix.png', dpi=150); plt.close(fig)
        metrics['confusion_matrix_png'] = 'generated'
    write_json(out / 'metrics.json', metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'formal'], required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--smoke-run')
    args = parser.parse_args()
    if not args.run_id.replace('_', '').replace('-', '').isalnum():
        parser.error('run-id must contain only letters, digits, underscore or hyphen')
    ns, nb, factory = load_factory()
    train_path = ns['DATA_DIR'] / 'train.csv'
    train = ns['read_input_csv'](train_path, 'train')
    folder = ROOT / 'experiments/folds'
    csv_path, meta_path = folder / 'folds_v1.csv', folder / 'folds_v1_metadata.json'
    splits = list(ns['iter_fixed_splits'](train, train_path, csv_path, meta_path))
    config = {'model_parameters': model_config(ns['make_model']()), 'random_seed': ns['RANDOM_SEED'],
              'seed_change': 'LinearSVC random_state: None -> 20261118; no other model parameter changed',
              'factory_source': factory, 'class_order': list(ns['class_order']),
              'score_columns': {f'decision_{i:02}': name for i, name in enumerate(ns['class_order'])},
              'score_type': 'LinearSVC uncalibrated decision_function, NOT probability',
              'fold_version': 'folds_v1'}
    fingerprints = {str(p.resolve()): digest(p) for p in
                    [*sorted(ns['DATA_DIR'].glob('*')), csv_path, meta_path] if p.is_file()}
    if args.mode == 'formal':
        if not args.smoke_run: parser.error('Formal run requires a successful matching --smoke-run')
        prior = json.loads((ROOT / 'artifacts' / args.smoke_run / 'run_metadata.json').read_text(encoding='utf-8'))
        assert prior['status'] == 'validated' and prior['mode'] == 'smoke'
        assert prior['config_sha256'] == config_hash(config) and prior['input_sha256'] == fingerprints
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    metadata = {'run_id': args.run_id, 'mode': args.mode, 'status': 'running',
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'execution': 'plain Python; Notebook factory and input/fold functions compiled directly',
                'python_executable': sys.executable, 'python': platform.python_version(),
                'data_dir': str(ns['DATA_DIR']),
                'versions': {name: importlib.metadata.version(name) for name in
                             ('numpy', 'pandas', 'scikit-learn', 'scipy', 'pyarrow', 'notebook', 'psutil')},
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
                'input_sha256': fingerprints, 'config_sha256': config_hash(config),
                'resource_note': 'RSS before/after each fold; Windows process lifetime peak when available, not isolated fold peak',
                'smoke_run': args.smoke_run}
    snapshots = out / 'source'; snapshots.mkdir()
    paths = [ROOT / 'notebooks/01_baseline.ipynb', *sorted((ROOT / 'scripts').glob('*.py')),
             ROOT / 'requirements.txt', ROOT / 'requirements-s01.txt']
    metadata['source_sha256'] = {}
    for source in paths:
        target = snapshots / source.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target); metadata['source_sha256'][str(source.relative_to(ROOT))] = digest(source)
    (snapshots / 'working-tree.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary'], cwd=ROOT))
    write_json(out / 'config.json', config); write_json(out / 'run_metadata.json', metadata)
    frames, folds, captured = [], [], []
    try:
        for fold, (fit, valid) in enumerate(splits, 1):
            if args.mode == 'smoke' and fold != 1: break
            frame, metric, recorded = run_fold(ns, train, fit, valid, fold, out, config['model_parameters'])
            frames.append(frame); folds.append(metric); captured.extend(recorded)
        write_json(out / 'warnings.json', captured)
        if args.mode == 'formal': finalise(out, train, frames, folds, list(ns['class_order']))
        else: write_json(out / 'metrics.json', folds[0])
        assert all(digest(ROOT / name) == expected for name, expected in fingerprints.items())
        from check_s04 import validate_run
        validate_run(out, allow_running=True)
        metadata['status'] = 'validated'; metadata['completed_utc'] = datetime.now(timezone.utc).isoformat()
        metadata['completed_folds'] = [f['fold'] for f in folds]
        write_json(out / 'run_metadata.json', metadata)
        print('VALIDATED', args.run_id, flush=True)
    except BaseException:
        metadata['status'] = 'failed'; metadata['error'] = traceback.format_exc()
        write_json(out / 'run_metadata.json', metadata)
        (out / 'failure.log').write_text(metadata['error'], encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
