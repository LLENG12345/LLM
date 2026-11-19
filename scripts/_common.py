"""Compatibility context for stage scripts, backed only by the installed ntl package."""
from functools import partial
import inspect
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from ntl import data, labels, models, splits

ROOT = Path(__file__).resolve().parents[1]


def load_context(config=None):
    namespace = {'Path': Path, 'np': np, 'pd': pd, 'GroupKFold': GroupKFold}
    for module in (labels, data, splits, models):
        namespace.update({name: value for name, value in vars(module).items()
                          if not name.startswith('__')})
    if config is not None:
        namespace['make_model'] = partial(models.make_model, config=models.resolve_config(config))
    namespace['DATA_DIR'] = data.resolve_data_dir(os.environ.get('NAME_THAT_LLM_DATA_DIR'))
    namespace['OUTPUT_DIR'] = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
    return namespace, None


def load_factory(config=None):
    namespace, _ = load_context(config)
    return namespace, None, inspect.getsource(models.make_model)
