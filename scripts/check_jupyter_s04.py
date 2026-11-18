"""Bounded tiny-notebook diagnostic with an explicit project interpreter."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/s04/jupyter'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if '--worker' in sys.argv:
        print('WORKER_IMPORT_START', flush=True)
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client.kernelspec import KernelSpecManager
        spec = KernelSpecManager().get_kernel_spec('ntl-s04')
        assert Path(spec.argv[0]).resolve() == Path(sys.executable).resolve()
        print('EXPLICIT_KERNEL', spec.argv, flush=True)
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
            "import sys\nprint(sys.executable)\nassert 1 + 1 == 2\nprint('TINY_NOTEBOOK_OK')")])
        client = NotebookClient(notebook, kernel_name='ntl-s04', timeout=20, startup_timeout=20,
                                on_cell_start=lambda **kw: print('CELL_START', flush=True))
        print('KERNEL_START_REQUESTED', flush=True)
        try:
            client.execute(cwd=str(ROOT))
        except Exception as exc:
            print('EXECUTION_ERROR', type(exc).__name__, str(exc), flush=True)
            raise
        nbformat.write(notebook, OUT / 'tiny.executed.ipynb')
        print('TINY_EXECUTION_PASS', flush=True)
        return
    kernel_dir = OUT / 'specs/kernels/ntl-s04'
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Temporary S04 project Python', 'language': 'python'}), encoding='utf-8')
    env = dict(os.environ)
    env.update(JUPYTER_PATH=str(OUT / 'specs'), JUPYTER_RUNTIME_DIR=str(OUT / 'runtime'),
               IPYTHONDIR=str(OUT / 'ipython'), PYTHONIOENCODING='utf-8')
    start = time.perf_counter()
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                env=env, capture_output=True, text=True, encoding='utf-8',
                                errors='replace', timeout=50,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        record = {'status': 'passed' if result.returncode == 0 else 'failed',
                  'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    except subprocess.TimeoutExpired as exc:
        decode = lambda value: value.decode('utf-8', errors='replace') if isinstance(value, bytes) else (value or '')
        record = {'status': 'outer_timeout_50s', 'stdout': decode(exc.stdout), 'stderr': decode(exc.stderr)}
    record['elapsed_seconds'] = time.perf_counter() - start
    record['project_python'] = sys.executable
    (OUT / 'diagnostic.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
