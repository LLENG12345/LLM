"""Compare real refactor runs to the immutable S04/S05 reference results."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ntl.io import digest, write_json
from ntl.validation import validate_run, validate_s05_run

ROOT = Path(__file__).resolve().parents[1]


def equivalence(cv_directory, full_directory, output):
    cv_directory, full_directory, output = map(Path, (cv_directory,full_directory,output))
    old_cv = ROOT/'artifacts/s04_baseline_v1_20261118_01'
    old_full = ROOT/'artifacts/s05_full_baseline_20261118_01'
    validate_run(cv_directory); validate_s05_run(full_directory)
    original = pd.read_parquet(old_cv/'oof_predictions.parquet')
    current = pd.read_parquet(cv_directory/'oof_predictions.parquet')
    key_columns = ['id','prompt_id','fold','true_model','predicted_model']
    pd.testing.assert_frame_equal(original[key_columns],current[key_columns])
    config = json.loads((old_cv/'config.json').read_text(encoding='utf-8'))
    new_config = json.loads((cv_directory/'config.json').read_text(encoding='utf-8'))
    assert config['model_parameters'] == new_config['model_parameters']
    scores = list(config['score_columns'])
    delta = float(np.max(np.abs(current[scores].to_numpy()-original[scores].to_numpy())))
    # Same environment/seed: require exact values rather than expanding tolerance.
    np.testing.assert_array_equal(current[scores].to_numpy(),original[scores].to_numpy())
    old_metrics = json.loads((old_cv/'metrics.json').read_text(encoding='utf-8'))
    new_metrics = json.loads((cv_directory/'metrics.json').read_text(encoding='utf-8'))
    metric_names = ['balanced_accuracy','accuracy','macro_f1','mean_fold_balanced_accuracy','std_fold_balanced_accuracy_ddof0']
    for key in metric_names: assert old_metrics[key] == new_metrics[key], key
    for name in ('classification_report.csv','class_recall.csv','confusion_matrix.csv','confusion_matrix_normalized.csv'):
        pd.testing.assert_frame_equal(pd.read_csv(old_cv/name),pd.read_csv(cv_directory/name),check_exact=True)
    old_submission, new_submission = pd.read_parquet(old_full/'submission.parquet'),pd.read_parquet(full_directory/'submission.parquet')
    pd.testing.assert_frame_equal(old_submission,new_submission)
    before = pd.read_parquet(old_full/'test_decision_scores.parquet')
    after = pd.read_parquet(full_directory/'test_decision_scores.parquet')
    pd.testing.assert_frame_equal(before,after,check_exact=True)
    full_delta = float(np.max(np.abs(after[scores].to_numpy()-before[scores].to_numpy())))
    np.testing.assert_array_equal(after[scores].to_numpy(),before[scores].to_numpy())
    report = {'status':'exact_equivalence_pass','kind':'baseline refactor verification, not model improvement',
        'cv_directory':str(cv_directory),'full_directory':str(full_directory),'oof_rows':len(current),
        'oof_labels_equal':True,'folds_and_ids_equal':True,'oof_decision_scores_equal':True,
        'oof_max_absolute_score_error':delta,'decision_score_tolerance':0,
        'classification_report_recall_confusion_equal':True,'metrics':{key:new_metrics[key] for key in metric_names},
        'test_labels_equal':True,'test_scores_equal':True,'test_max_absolute_score_error':full_delta,
        'submission_contents_equal':True,'kaggle_submission':'not executed'}
    output.parent.mkdir(parents=True,exist_ok=True); write_json(output,report)
    return report


def check_integrity(output):
    checkpoint = json.loads((ROOT/'artifacts/s06b/before.json').read_text(encoding='utf-8'))
    hashes = checkpoint['protected_sha256']
    current = {name:digest(ROOT/name) for name in hashes}
    assert current == hashes, 'Historical data/artifacts changed'
    write_json(output,{'all_unchanged':True,'protected_files':len(hashes),'sha256':current})
    return len(hashes)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--cv',type=Path,required=True)
    parser.add_argument('--full',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/s06b/equivalence.json')
    args=parser.parse_args(); report=equivalence(args.cv,args.full,args.output)
    count=check_integrity(args.output.with_name('integrity_report.json'))
    print('S06B_EXACT_EQUIVALENCE_PASS',report['oof_rows'],'protected',count)
