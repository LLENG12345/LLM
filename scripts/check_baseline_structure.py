"""Protect unchanged model and fitting logic while allowing fixed-fold wiring."""
import ast
import copy
import json
import subprocess
from pathlib import Path


def check_baseline_structure(nb, root):
    old = json.loads(subprocess.check_output(
        ['git', 'show', 'HEAD:notebooks/01_baseline.ipynb'], cwd=root))
    assert nb['cells'][1:3] == old['cells'][1:3], 'Path/input contract changed'
    assert nb['cells'][4:] == old['cells'][4:], 'Final prediction logic changed'
    current = ast.parse(''.join(nb['cells'][3]['source']))
    original = ast.parse(''.join(old['cells'][3]['source']))
    for tree in (current, original):
        assert sum(isinstance(node, ast.For) for node in tree.body) == 1
    for name in ('make_model',):
        now = next(node for node in current.body if isinstance(node, ast.FunctionDef) and node.name == name)
        was = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == name)
        # S04 permits only the explicit deterministic LinearSVC seed.
        seed = next(n for n in current.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'RANDOM_SEED' for t in n.targets))
        assert ast.literal_eval(seed.value) == 20261118
        now = copy.deepcopy(now)
        classifier = next(n for n in ast.walk(now) if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Name) and n.func.id == 'LinearSVC')
        keywords = [k for k in classifier.keywords if k.arg == 'random_state']
        assert len(keywords) == 1 and isinstance(keywords[0].value, ast.Name)
        assert keywords[0].value.id == 'RANDOM_SEED'
        classifier.keywords = [k for k in classifier.keywords if k.arg != 'random_state']
        # Permit comparing against a later committed snapshot with this same seed.
        was = copy.deepcopy(was)
        previous_classifier = next(n for n in ast.walk(was) if isinstance(n, ast.Call)
                                   and isinstance(n.func, ast.Name) and n.func.id == 'LinearSVC')
        previous_classifier.keywords = [k for k in previous_classifier.keywords if k.arg != 'random_state']
        assert ast.dump(now) == ast.dump(was), 'Model configuration changed beyond approved seed'
    now_loop = next(node for node in current.body if isinstance(node, ast.For))
    old_loop = next(node for node in original.body if isinstance(node, ast.For))
    assert [ast.dump(node) for node in now_loop.body] == [ast.dump(node) for node in old_loop.body], 'CV fit/predict body changed'
    assert ast.dump(now_loop.target) == ast.dump(old_loop.target)
    assert 'fixed_splits' in ast.unparse(now_loop.iter), 'CV does not use fixed folds'
