"""Public-behavior/config protection replacing historical Notebook cell AST checks."""
import ast
import inspect
import json
from pathlib import Path
import numpy as np
from ntl.data import read_input_csv
from ntl.models import make_model, model_config, RANDOM_SEED
from ntl.submission import fit_full_model, predict_test, write_and_verify_submission
from ntl.experiments import run_fold

def check_baseline_structure(nb, root):
    baseline = json.loads((Path(root) / 'artifacts/s04_baseline_v1_20261118_01/config.json').read_text(encoding='utf-8'))
    assert model_config(make_model()) == baseline['model_parameters'], 'Default model configuration changed'
    assert RANDOM_SEED == baseline['random_seed'] == 20261118
    # Protect the actual shared fit/predict bodies, not obsolete cell positions.
    fold_source = inspect.getsource(run_fold)
    assert "model.fit(fit_frame['text'].astype(str), fit_frame['model'])" in fold_source
    assert "model.predict(valid_frame['text'].astype(str))" in fold_source
    assert "fit_frame['prompt_id']" in fold_source and "valid_frame['prompt_id']" in fold_source
    full = inspect.getsource(fit_full_model)
    assert "model.fit(train['text'].astype(str), train['model'])" in full
    assert '.fit(' not in inspect.getsource(predict_test)
    assert 'index=False' in inspect.getsource(write_and_verify_submission)
    if nb is not None:
        source = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
        assert 'from ntl.' in source and 'def make_model' not in source
    # Independent tiny corpus proves validation-only tokens cannot change vocab/IDF.
    model = make_model()
    model.fit(['alpha beta first', 'alpha beta second', 'gamma delta first', 'gamma delta second'], ['a','a','b','b'])
    word = model.named_steps['features'].transformer_list[0][1]
    vocab, idf = dict(word.vocabulary_), word.idf_.copy()
    model.predict(['validationonlytoken'])
    assert 'validationonlytoken' not in word.vocabulary_ and word.vocabulary_ == vocab
    np.testing.assert_array_equal(word.idf_, idf)
