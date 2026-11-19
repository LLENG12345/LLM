from pathlib import Path
from datetime import datetime, timezone
from functools import partial
from types import SimpleNamespace
import importlib.metadata
import inspect
import os
import platform
import shutil
import subprocess
import sys
import traceback
from .data import read_input_csv, resolve_data_dir
from .labels import class_order
from .models import make_model, resolve_config
from .splits import iter_fixed_splits
from .io import digest
"""Existing fold fit, OOF collection and reporting, moved without algorithm changes."""
import importlib.util
import json
import time
import warnings
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from .io import write_json
from .models import model_config, config_hash
from .metrics import scores_summary


def fitted_feature_summary(model):
    """Observe fitted widths; never fit/transform or use configured vocabulary caps."""
    dimensions = {name: int(transform.n_features_out_) if name == 'style' else
                  len(transform.named_steps['tfidf'].vocabulary_) if name == 'morphology' else len(transform.vocabulary_)
                  for name, transform in model.named_steps['features'].transformer_list}
    total = int(model.named_steps['classifier'].coef_.shape[1])
    assert total > 0 and total == sum(dimensions.values())
    return {'feature_dimensions': dimensions, 'n_features': total,
            'feature_branches': list(dimensions)}


def run_fold(ns, train, fit, valid, fold, out, expected_config):
    model = ns['make_model']()
    assert not hasattr(model.named_steps['classifier'], 'coef_')
    style = dict(model.named_steps['features'].transformer_list).get('style')
    morphology = dict(model.named_steps['features'].transformer_list).get('morphology')
    if morphology is not None:
        assert not hasattr(morphology.named_steps['tfidf'],'vocabulary_'), 'Each fold requires a fresh morphology TF-IDF'
    if style is not None:
        assert not hasattr(style, 'scaler_'), 'Each fold requires a new unfitted style scaler'
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
        if morphology is not None:
            from .morphology import morphology_state_digest
            morphology_state = morphology_state_digest(morphology)
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
    metric.update(fitted_feature_summary(model))
    if morphology is not None:
        from .morphology import write_morphology_audit
        assert morphology_state_digest(morphology) == morphology_state
        write_morphology_audit(morphology,fit_frame['text'].astype(str),folder,
            model.named_steps['features'].transformer_weights['morphology'])
    if style is not None:
        from .style import text_sequence_digest, STYLE_FEATURE_NAMES, STYLE_VERSION
        assert int(style.scaler_.n_samples_seen_) == len(fit_frame)
        assert style.fit_text_sha256_ == text_sequence_digest(fit_frame['text'].astype(str))
        assert style.n_features_out_ == len(STYLE_FEATURE_NAMES) == 162
        assert np.isfinite(style.scaler_.mean_).all() and np.isfinite(style.scaler_.scale_).all()
        write_json(folder / 'style_audit.json', {'definition_version': STYLE_VERSION,
            'feature_names': list(STYLE_FEATURE_NAMES), 'n_features': style.n_features_out_,
            'n_fit_samples': int(style.scaler_.n_samples_seen_), 'fit_text_sha256': style.fit_text_sha256_,
            'training_text_digest_matches': True, 'scaler': 'StandardScaler; with_mean=True; with_std=True',
            'clip': style.clip, 'weight': model.named_steps['features'].transformer_weights['style'],
            'mean': style.scaler_.mean_.tolist(), 'var': style.scaler_.var_.tolist(),
            'scale': style.scaler_.scale_.tolist()})
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


def run_experiment(run_id, mode='formal', config=None, data_dir=None, project_root=None,
                   smoke_run=None, purpose='baseline'):
    """Run the existing fixed-v1 experiment, with explicit factory configuration."""
    ROOT = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[2]
    if mode not in ('smoke', 'formal'):
        raise ValueError('mode must be smoke or formal')
    args = SimpleNamespace(run_id=run_id, mode=mode, smoke_run=smoke_run)
    if not args.run_id.replace('_', '').replace('-', '').isalnum():
        raise ValueError('run-id must contain only letters, digits, underscore or hyphen')
    resolved = resolve_config(config)
    selected = resolve_data_dir(data_dir if data_dir is not None else os.environ.get('NAME_THAT_LLM_DATA_DIR'))
    ns = {'DATA_DIR': selected, 'read_input_csv': read_input_csv, 'iter_fixed_splits': iter_fixed_splits,
          'make_model': partial(make_model, config=resolved), 'RANDOM_SEED': resolved['classifier']['random_state'],
          'class_order': class_order}
    factory = inspect.getsource(make_model)
    train_path = ns['DATA_DIR'] / 'train.csv'
    train = ns['read_input_csv'](train_path, 'train')
    folder = ROOT / 'experiments/folds'
    csv_path, meta_path = folder / 'folds_v1.csv', folder / 'folds_v1_metadata.json'
    splits = list(ns['iter_fixed_splits'](train, train_path, csv_path, meta_path))
    config = {'model_parameters': model_config(ns['make_model']()), 'random_seed': ns['RANDOM_SEED'],
              'implementation': 'shared ntl modules; default baseline unchanged', 'experiment_configuration': resolved,
              'factory_source': factory, 'class_order': list(ns['class_order']),
              'score_columns': {f'decision_{i:02}': name for i, name in enumerate(ns['class_order'])},
              'score_type': 'LinearSVC uncalibrated decision_function, NOT probability',
              'fold_version': 'folds_v1'}
    fingerprints = {str(p.resolve()): digest(p) for p in
                    [*sorted(ns['DATA_DIR'].glob('*')), csv_path, meta_path] if p.is_file()}
    if args.mode == 'formal':
        if not args.smoke_run: raise ValueError('Formal run requires a successful matching smoke_run')
        prior = json.loads((ROOT / 'artifacts' / args.smoke_run / 'run_metadata.json').read_text(encoding='utf-8'))
        assert prior['status'] == 'validated' and prior['mode'] == 'smoke'
        assert prior['config_sha256'] == config_hash(config) and prior['input_sha256'] == fingerprints
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    metadata = {'run_id': args.run_id, 'mode': args.mode, 'status': 'running',
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'execution': 'plain Python; shared ntl modules, no Notebook execution', 'purpose': purpose,
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
             ROOT / 'requirements.txt', ROOT / 'requirements-s01.txt', ROOT / 'pyproject.toml',
             *sorted((ROOT / 'src/ntl').glob('*.py'))]
    metadata['source_sha256'] = {}
    for source in paths:
        if not source.is_file(): continue
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
        from .validation import validate_run
        validate_run(out, allow_running=True)
        metadata['status'] = 'validated'; metadata['completed_utc'] = datetime.now(timezone.utc).isoformat()
        metadata['completed_folds'] = [f['fold'] for f in folds]
        write_json(out / 'run_metadata.json', metadata)
        print('VALIDATED', args.run_id, flush=True)
        return out
    except BaseException:
        metadata['status'] = 'failed'; metadata['error'] = traceback.format_exc()
        write_json(out / 'run_metadata.json', metadata)
        (out / 'failure.log').write_text(metadata['error'], encoding='utf-8')
        raise
