"""One baseline configuration and factory; overrides never mutate defaults."""
from copy import deepcopy
import hashlib
import json
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from .features import feature_defaults, make_features

RANDOM_SEED = 20261118


def baseline_config():
    return {'branches': ['word', 'char'], 'features': feature_defaults(),
            'classifier': {'C': 1.0, 'class_weight': 'balanced', 'max_iter': 3000,
                'tol': 0.0001, 'penalty': 'l2', 'loss': 'squared_hinge', 'multi_class': 'ovr',
                'dual': 'auto', 'fit_intercept': True, 'intercept_scaling': 1,
                'verbose': 0, 'random_state': RANDOM_SEED}}


def resolve_config(overrides=None):
    result = baseline_config()
    # Introduce optional defaults only when explicitly enabled; old resolved
    # configs and get_params remain byte-for-byte equivalent as JSON values.
    if isinstance(overrides, dict) and isinstance(overrides.get('branches'), list) and 'style' in overrides['branches']:
        from .style import style_defaults
        result['features']['style'] = style_defaults()
    if isinstance(overrides, dict) and isinstance(overrides.get('branches'), list) and 'morphology' in overrides['branches']:
        from .morphology import morphology_defaults
        result['features']['morphology'] = morphology_defaults()
    def merge(target, supplied, prefix=''):
        if not isinstance(supplied, dict):
            raise ValueError(f'{prefix or "config"} must be a dictionary')
        for key, value in supplied.items():
            if key not in target:
                raise ValueError(f'Unknown experiment setting: {prefix}{key}')
            if isinstance(target[key], dict):
                merge(target[key], value, prefix + key + '.')
            else:
                target[key] = deepcopy(value)
    if overrides is not None:
        merge(result, overrides)
    branches = result['branches']
    if (not isinstance(branches, list) or not branches or len(branches) != len(set(branches))
            or not set(branches).issubset({'word', 'char', 'style', 'morphology'})):
        raise ValueError('branches must be a nonempty unique list of word, char, style, morphology')
    return result


def make_model(config=None):
    settings = resolve_config(config)
    return Pipeline([('features', make_features(settings)),
                     ('classifier', LinearSVC(**settings['classifier']))])


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
