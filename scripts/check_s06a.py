"""Validate standalone generated cells in isolated Python, never claim Jupyter/Kaggle execution."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import nbformat
import numpy as np
import pandas as pd

from build_submission_notebook import ROOT, TARGET, build, extract
from check_s05 import validate_submission


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


WORKER = r'''import contextlib, io, json, os, pathlib, socket, sys, time
sys.dont_write_bytecode = True
directory = pathlib.Path.cwd().resolve()
repository = pathlib.Path(sys.argv[1]).resolve()
environment = pathlib.Path(sys.prefix).resolve()
sys.path[:] = [p for p in sys.path if not (pathlib.Path(p).resolve().is_relative_to(repository)
               and not pathlib.Path(p).resolve().is_relative_to(environment))]
notebook = json.loads((directory / '99_submission.ipynb').read_text(encoding='utf-8'))
os.environ.pop('PYTHONPATH', None)
os.environ['NAME_THAT_LLM_DATA_DIR'] = str(directory / 'input')
os.environ['NAME_THAT_LLM_OUTPUT_DIR'] = str(directory / 'output')
denied = {'network': 0, 'repository': 0, 'process': 0}
denied_details = []

def audit(event, args):
    # gethostname reads local OS identity, with no DNS or network I/O. Some
    # installed dependencies inspect it on import. DNS/connect/send remain denied.
    if event.startswith('socket.') and event not in ('socket.__new__', 'socket.gethostname'):
        denied['network'] += 1
        denied_details.append({'kind': 'network', 'event': event})
        raise PermissionError('S06A_NETWORK_DISABLED')
    if event in ('subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn', 'os.spawn'):
        denied['process'] += 1
        denied_details.append({'kind': 'process', 'event': event, 'executable': str(args[0]), 'arguments': str(args[1])})
        raise PermissionError('S06A_CHILD_PROCESSES_DISABLED')
    if event in ('open', 'os.listdir', 'os.scandir', 'os.chdir') and args:
        path = args[0]
        if isinstance(path, (str, bytes, os.PathLike)):
            resolved = pathlib.Path(os.fsdecode(path)).resolve()
            # Installed dependencies in the interpreter environment are allowed;
            # project code, models, artifacts, folds and .git remain inaccessible.
            if resolved.is_relative_to(repository) and not resolved.is_relative_to(environment):
                denied['repository'] += 1
                denied_details.append({'kind': 'repository', 'event': event, 'path': str(resolved)})
                raise PermissionError('S06A_REPOSITORY_DISABLED')

sys.addaudithook(audit)
try:
    socket.create_connection(('example.invalid', 443), timeout=1)
except PermissionError as exc:
    assert 'S06A_NETWORK_DISABLED' in str(exc)
else:
    raise AssertionError('Network guard did not intercept its probe')
try:
    open(repository / 'requirements.txt', 'rb')
except PermissionError as exc:
    assert 'S06A_REPOSITORY_DISABLED' in str(exc)
else:
    raise AssertionError('Repository guard did not intercept its probe')
probes = denied.copy()

def verify_denials():
    assert denied['network'] == probes['network'], f'Unexpected network attempt: {denied_details}'
    assert denied['repository'] == probes['repository'], f'Unexpected project access: {denied_details}'
    processes = [item for item in denied_details if item['kind'] == 'process']
    # Python platform._syscmd_ver tries these local Windows version fallbacks.
    # All are still blocked by the audit hook; no child process is permitted.
    for item in processes:
        assert item['event'] == 'subprocess.Popen' and item['executable'].lower().endswith('cmd.exe'), item
        assert item['arguments'] in [item['executable'] + ' /c "' + command + '"'
                                     for command in ('ver', 'command /c ver', 'cmd /c ver')], item
    assert denied['process'] == len(processes)

namespace = {'__name__': '__main__'}
start = time.perf_counter()
try:
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        output, error = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            exec(compile(source, f'99_submission:cell{i}', 'exec'), namespace)
        cell['execution_count'] = sum(c['cell_type'] == 'code' for c in notebook['cells'][:i+1])
        cell['outputs'] = [{'output_type': 'stream', 'name': name, 'text': stream.getvalue()}
                           for name, stream in [('stdout', output), ('stderr', error)] if stream.getvalue()]
        print(f'CELL {i} OK', output.getvalue(), error.getvalue(), flush=True)
        verify_denials()
except ValueError as exc:
    if '--expect-training-version-error' not in sys.argv or 'training version differs' not in str(exc):
        raise
    verify_denials()
    (directory / 'result.json').write_text(json.dumps({'training_version_rejected': True,
        'error': str(exc), 'network_guard_probes': probes, 'denied_events': denied}), encoding='utf-8')
    print('EXPECTED_TRAINING_VERSION_REJECTION', flush=True)
    raise SystemExit(0)
if '--expect-training-version-error' in sys.argv:
    raise AssertionError('Changed training version was accepted')

import numpy as np
import pandas as pd
from unittest.mock import patch
ns = namespace
tests = []
if '--variable-counts' in sys.argv:
    before = ns['model_state_fingerprint'](ns['model'])
    texts = ns['train']['text'].tolist()
    with patch.object(ns['model'], 'fit', side_effect=AssertionError('Prediction must never fit')):
        for count in (0, 1, 257, 4000):
            frame = pd.DataFrame({'id': pd.Series([f'{i:06}' for i in range(count)], dtype='string'),
                'text': pd.Series([texts[i % len(texts)] + ' s06a_unseen_token_zz' for i in range(count)], dtype='string')})
            started = time.perf_counter()
            pred, scores = ns['predict_test'](ns['model'], frame, ns, batch_size=128)
            submission = ns['build_submission'](frame, pred, ns['class_order'])
            ns['write_and_verify_submission'](submission, frame, ns['class_order'],
                directory / 'capacity' / str(count) / 'submission.parquet', [directory / 'input'])
            assert scores.shape == (count, 18)
            tests.append({'rows': count, 'batch_size': 128, 'seconds': time.perf_counter()-started})
    assert ns['model_state_fingerprint'](ns['model']) == before
try:
    ns['resolve_submission_data'](directory / 'nonexistent')
except FileNotFoundError as exc:
    assert 'no fallback' in str(exc)
else:
    raise AssertionError('Invalid explicit input did not fail')
verify_denials()
report = {'execution': 'isolated Python -I; sequential cells, not a Jupyter kernel',
    'runtime': ns['runtime_summary'], 'actual_parameters': ns['actual_parameters'],
    'class_order': list(ns['class_order']), 'predictions': ns['submission'].to_dict('list'),
    'decision_scores': ns['decision_scores'].tolist(), 'variable_counts': tests,
    'seconds': time.perf_counter()-start, 'network_guard_probes': probes,
    'denied_events': denied, 'notebook_network_attempts': denied['network']-probes['network'],
    'blocked_system_version_probes': [item for item in denied_details if item['kind'] == 'process'],
    'child_processes_executed': 0,
    'notebook_repository_attempts': denied['repository']-probes['repository'],
    'initial_input_files': sorted(p.name for p in (directory / 'input').iterdir()),
    'project_on_sys_path': any(pathlib.Path(p).resolve() == repository for p in sys.path if p),
    'output_files': sorted(p.name for p in (directory / 'output').iterdir())}
(directory / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
(directory / '99_submission.python-executed.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
print('ISOLATED_CELLS_PASS', flush=True)
'''


def isolated_case(name, destination, data_dir, mode=None):
    case_out = destination / name; case_out.mkdir()
    with tempfile.TemporaryDirectory(prefix='ntl-s06a-') as temporary:
        sandbox = Path(temporary)
        inputs = sandbox / 'input'; inputs.mkdir()
        shutil.copyfile(data_dir / 'train.csv', inputs / 'train.csv')
        shutil.copyfile(data_dir / 'test.csv', inputs / 'test.csv')
        if mode == 'different-test':
            frame = pd.read_csv(inputs / 'test.csv', dtype='string', keep_default_na=False)
            synthetic = pd.DataFrame({'id': [f'{i:06}' for i in range(17)],
                'text': [frame.iloc[i % len(frame)]['text'] for i in range(17)]}, dtype='string')
            synthetic.to_csv(inputs / 'test.csv', index=False)
        if mode == 'changed-train':
            with (inputs / 'train.csv').open('ab') as stream: stream.write(b'\n')
        shutil.copyfile(TARGET, sandbox / '99_submission.ipynb')
        (sandbox / 'worker.py').write_text(WORKER, encoding='utf-8')
        command = [sys.executable, '-I', '-X', 'utf8', str(sandbox / 'worker.py'), str(ROOT)]
        if mode is None: command.append('--variable-counts')
        if mode == 'changed-train': command.append('--expect-training-version-error')
        environment = dict(os.environ, PYTHONIOENCODING='utf-8')
        environment.pop('PYTHONPATH', None)
        result = subprocess.run(command, cwd=sandbox, env=environment, capture_output=True, timeout=120)
        (case_out / 'run.log').write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f'Isolated case {name} failed ({result.returncode}); see {case_out / "run.log"}')
        report = json.loads((sandbox / 'result.json').read_text(encoding='utf-8'))
        write_json(case_out / 'result.json', report)
        if mode != 'changed-train':
            shutil.copyfile(sandbox / 'output/submission.parquet', case_out / 'submission.parquet')
            shutil.copyfile(sandbox / '99_submission.python-executed.ipynb', case_out / '99_submission.python-executed.ipynb')
            loaded = pd.read_parquet(case_out / 'submission.parquet')
            test = pd.read_csv(inputs / 'test.csv', dtype='string', keep_default_na=False, na_filter=False)
            validate_submission(loaded, test, report['class_order'])
            assert report['initial_input_files'] == ['test.csv', 'train.csv']
            assert not report['project_on_sys_path']
            assert report['notebook_network_attempts'] == report['notebook_repository_attempts'] == 0
            assert report['output_files'] == ['submission.parquet']
        else:
            assert report['training_version_rejected']
        print('ISOLATED_CASE_PASS', name, flush=True)
        return report


def check_structure():
    notebook = nbformat.read(TARGET, as_version=4); nbformat.validate(notebook)
    payload, _ = build()
    assert payload.encode('utf-8') == TARGET.read_bytes(), 'Generated output is stale'
    manifest = json.loads(TARGET.with_suffix('.sources.json').read_text(encoding='utf-8'))
    assert manifest['notebook_sha256'] == digest(TARGET), 'Notebook byte fingerprint mismatch'
    source = '\n'.join(c.source for c in notebook.cells if c.cell_type == 'code')
    tree = ast.parse(source)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            assert cell.execution_count is None and cell.outputs == []
    allowed_imports = {'pathlib', 'hashlib', 'importlib', 'json', 'os', 'platform', 'time',
                      'warnings', 'numpy', 'pandas', 'pyarrow', 'sklearn', 'copy'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(a.name.split('.')[0] in allowed_imports for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module.split('.')[0] in allowed_imports
    for forbidden in ('GroupKFold', 'cross_val', 'load_fixed_folds', 'joblib', 'subprocess',
                      'pip install', 'sys.path', 'D:/', 'D:\\', 'C:\\', 'C:/'):
        assert forbidden not in source, forbidden
    baseline = json.loads((ROOT / 'artifacts/s04_baseline_v1_20261118_01/config.json').read_text(encoding='utf-8'))
    expected = extract((ROOT / 'src/ntl/models.py').read_text(encoding='utf-8'), ['make_model'])
    actual = extract(source, ['make_model'])
    assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(expected))
    assert source.count('OFFICIAL_MODEL_LABELS =') == 1
    return baseline, len(notebook.cells), sum(c.cell_type == 'code' for c in notebook.cells)


def regressions(output):
    jobs = [('s01', ['scripts/check_s01.py']), ('s02', ['scripts/check_s02.py']),
            ('s03', ['scripts/check_s03.py']),
            ('s04-check', ['scripts/check_s04.py', 'artifacts/s04_baseline_v1_20261118_01']),
            ('s04-test', ['scripts/test_s04.py', 'artifacts/s04_baseline_v1_20261118_01']),
            ('s05-check', ['scripts/check_s05.py', 'artifacts/s05_full_baseline_20261118_01'])]
    # S05 tests write a capacity report; redirect only that report root to new S06A
    # evidence so the original S05 run and its validation evidence stay immutable.
    redirected = output / 's05-report-root'
    (redirected / 'artifacts/s05').mkdir(parents=True)
    wrapper = ("import pathlib,sys,unittest; sys.path.insert(0,'scripts'); "
        "sys.argv=['test_s05.py','artifacts/s05_full_baseline_20261118_01']; import test_s05; "
        f"test_s05.ROOT=pathlib.Path({str(redirected)!r}); "
        "r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(test_s05.SubmissionTests)); "
        "raise SystemExit(not r.wasSuccessful())")
    jobs.append(('s05-test', ['-c', wrapper]))
    results = {}
    for name, args in jobs:
        started = time.perf_counter()
        result = subprocess.run([sys.executable, *args], cwd=ROOT,
            env=dict(os.environ, PYTHONIOENCODING='utf-8',
                     NAME_THAT_LLM_S02_REPORT=str(output / 's02-input-summary.json'),
                     NAME_THAT_LLM_S03_REPORT_DIR=str(output / 's03-report')),
            capture_output=True, timeout=120)
        (output / f'{name}.log').write_bytes(result.stdout + result.stderr)
        results[name] = {'exit_code': result.returncode, 'seconds': time.perf_counter()-started}
        assert result.returncode == 0, f'{name} failed; see its log'
        print('REGRESSION_PASS', name, flush=True)
    return results


def integrity(checkpoint, output):
    before = json.loads(checkpoint.read_text(encoding='utf-8'))
    expected = before.get('protected_sha256', before.get('hashes'))
    assert expected, 'Empty integrity checkpoint'
    current = {name: digest(ROOT / name) for name in expected}
    assert current == expected, 'Protected files changed'
    result = {'protected_files': len(current), 'all_unchanged': True, 'sha256': current}
    write_json(output / 'integrity_report.json', result)
    return result['protected_files']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/name-that-llm')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/s06a/validation')
    parser.add_argument('--regressions', action='store_true')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'artifacts/s06a/before.json')
    args = parser.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=False)
    try:
        config, cells, code_cells = check_structure()
        official = isolated_case('official', out, args.data_dir.resolve())
        assert official['actual_parameters'] == config['model_parameters']
        assert official['class_order'] == config['class_order']
        expected = pd.read_parquet(ROOT / 'artifacts/s05_full_baseline_20261118_01/submission.parquet')
        actual = pd.read_parquet(out / 'official/submission.parquet')
        pd.testing.assert_frame_equal(actual, expected)
        reference_scores = pd.read_parquet(ROOT / 'artifacts/s05_full_baseline_20261118_01/test_decision_scores.parquet')
        np.testing.assert_array_equal(official['decision_scores'], reference_scores[list(config['score_columns'])].to_numpy())
        changed = isolated_case('runtime-test-17', out, args.data_dir.resolve(), 'different-test')
        assert changed['runtime']['n_test'] == 17
        assert changed['actual_parameters'] == config['model_parameters']
        rejected = isolated_case('changed-training-rejected', out, args.data_dir.resolve(), 'changed-train')
        regression = regressions(out) if args.regressions else {'status': 'not executed'}
        count = integrity(args.checkpoint.resolve(), out)
        diff = subprocess.run(['git', 'diff', '--check'], cwd=ROOT, capture_output=True)
        (out / 'diff-check.log').write_bytes(diff.stdout + diff.stderr)
        assert diff.returncode == 0
        summary = {'status': 'local_notebook_validated', 'cells': cells, 'code_cells': code_cells,
            'execution': 'isolated ordinary Python -I; not Jupyter, container or Kaggle',
            'jupyter': 'not retested; prior kernel start limitation retained',
            'kaggle_runtime': 'not executed', 'kaggle_submission': 'not executed',
            'full_model_parameters_equal': True, 's05_labels_equal': True, 's05_decision_scores_equal': True,
            'official_run': official['runtime'], 'different_test_run': changed['runtime'],
            'variable_counts': official['variable_counts'], 'changed_training_rejected': rejected['training_version_rejected'],
            'guards': {'network_denial_probe_pass': True, 'repository_denial_probe_pass': True,
                       'notebook_network_attempts': 0, 'notebook_repository_attempts': 0,
                       'note': 'Python audit guards, not an OS/container sandbox; installed environment packages allowed'},
            'regressions': regression, 'protected_files_unchanged': count, 'git_diff_check_exit_code': diff.returncode,
            'notebook_sha256': digest(TARGET)}
        write_json(out / 'validation-summary.json', summary)
        print('S06A_LOCAL_NOTEBOOK_PASS', out, flush=True)
    except BaseException as exc:
        write_json(out / 'failure.json', {'status': 'failed', 'type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
