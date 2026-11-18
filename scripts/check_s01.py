"""S01 smoke checks: imports, paths and I/O only; never fit a model."""
import hashlib
from contextlib import chdir
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import notebook
import numpy
import pandas as pd
import pyarrow
import sklearn


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / 'notebooks' / '01_baseline.ipynb'


def hashes(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.iterdir() if p.is_file()}


def main():
    versions = {'python': sys.version.split()[0], 'numpy': numpy.__version__,
                'pandas': pd.__version__, 'scikit-learn': sklearn.__version__,
                'pyarrow': pyarrow.__version__, 'notebook': notebook.__version__}
    print('VERSIONS', json.dumps(versions))
    nb = json.loads(NB_PATH.read_text(encoding='utf-8'))
    original = json.loads(subprocess.check_output(
        ['git', 'show', 'HEAD:notebooks/01_baseline.ipynb'], cwd=ROOT))
    from check_baseline_structure import check_baseline_structure
    check_baseline_structure(nb, ROOT)
    source = ''.join(nb['cells'][1]['source'])
    data_dir = ROOT / 'data' / 'name-that-llm'
    before = hashes(data_dir)
    snapshot = ROOT / 'artifacts' / 's01' / 'before.json'
    if snapshot.exists():
        expected = json.loads(snapshot.read_text(encoding='utf-8'))['data']
        assert before == {Path(p).name: h for p, h in expected.items()}

    previous_cwd = Path.cwd()
    previous_override = os.environ.pop('NAME_THAT_LLM_DATA_DIR', None)
    try:
        results = []
        for cwd in (ROOT, ROOT / 'notebooks'):
            os.chdir(cwd)
            namespace = {}
            exec(compile(source, str(NB_PATH), 'exec'), namespace)
            selected = namespace['DATA_DIR']
            assert selected == data_dir.resolve()
            train = pd.read_csv(selected / 'train.csv')
            test = pd.read_csv(selected / 'test.csv')
            results.append((selected, hashes(selected), len(train), len(test)))
        assert results[0] == results[1]
        print('ROOT_AND_NOTEBOOKS_SAME_DATA', results[0][2:])

        # Restore a directory outside the fixture before Windows removes it.
        with tempfile.TemporaryDirectory(prefix='ntl-s01-') as tmp, chdir(ROOT):
            base = Path(tmp)
            project = base / 'project'
            (project / 'notebooks').mkdir(parents=True)
            (project / 'notebooks' / '01_baseline.ipynb').touch()
            (project / 'requirements.txt').touch()
            modern = project / 'data' / 'name-that-llm'
            legacy = project / 'data'
            kaggle = base / 'kaggle'
            explicit = base / 'explicit'
            for directory in (modern, legacy, kaggle, explicit):
                directory.mkdir(parents=True, exist_ok=True)
                for filename in ('train.csv', 'test.csv'):
                    (directory / filename).touch()
            os.chdir(project)
            namespace['KAGGLE_DATA'] = kaggle
            resolve = namespace['resolve_data_dir']
            assert resolve(explicit) == explicit.resolve()
            assert resolve() == kaggle.resolve()
            (kaggle / 'test.csv').unlink()
            assert resolve() == modern.resolve()
            (modern / 'train.csv').unlink()
            assert resolve() == legacy.resolve()
            try:
                resolve(base / 'invalid')
            except FileNotFoundError as exc:
                assert 'Explicit' in str(exc) and 'No fallback' in str(exc)
                print('INVALID_EXPLICIT_REJECTED', str(exc))
            else:
                raise AssertionError('Invalid explicit directory silently fell back')
            (legacy / 'test.csv').unlink()
            try:
                resolve()
            except FileNotFoundError as exc:
                assert all(str(p.resolve()) in str(exc) for p in (kaggle, modern, legacy))
                print('MISSING_CANDIDATES_REPORTED')
            else:
                raise AssertionError('Missing input not rejected')
            frame = pd.DataFrame({'id': ['001', 'abc'], 'value': [1.25, 2.5]})
            output = base / 'roundtrip.parquet'
            frame.to_parquet(output, index=False)
            pd.testing.assert_frame_equal(frame, pd.read_parquet(output))
            print('PARQUET_ROUNDTRIP_OK')
            os.environ['NAME_THAT_LLM_DATA_DIR'] = str(explicit)
            env_namespace = {}
            exec(compile(source, str(NB_PATH), 'exec'), env_namespace)
            assert env_namespace['DATA_DIR'] == explicit.resolve()
        print('ALL_FOUR_PATH_PRIORITIES_OK')
    finally:
        os.chdir(previous_cwd)
        if previous_override is None:
            os.environ.pop('NAME_THAT_LLM_DATA_DIR', None)
        else:
            os.environ['NAME_THAT_LLM_DATA_DIR'] = previous_override
    assert before == hashes(data_dir), 'Original data changed'
    print('ORIGINAL_DATA_SHA256_UNCHANGED', json.dumps(before))
    print('ALGORITHM_CELLS_UNCHANGED; S01_SMOKE_PASS')


if __name__ == '__main__':
    main()
