"""Configurable raw word/character, rich-style and morphology branches."""
from copy import deepcopy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

_FEATURE_DEFAULTS = {
    'word': {'analyzer': 'word', 'ngram_range': [1, 2], 'min_df': 2,
             'max_features': 40000, 'sublinear_tf': True, 'dtype': 'float32',
             'lowercase': True, 'norm': 'l2', 'use_idf': True, 'smooth_idf': True, 'max_df': 1.0},
    'char': {'analyzer': 'char', 'ngram_range': [2, 5], 'min_df': 2,
             'max_features': 80000, 'sublinear_tf': True, 'dtype': 'float32',
             'lowercase': True, 'norm': 'l2', 'use_idf': True, 'smooth_idf': True, 'max_df': 1.0},
}


def feature_defaults():
    return deepcopy(_FEATURE_DEFAULTS)


def make_features(config):
    transforms = []
    weights = {}
    for name in config['branches']:
        settings = deepcopy(config['features'][name])
        if name == 'morphology':
            from sklearn.pipeline import Pipeline
            from .morphology import MorphologyTransformer
            version = settings.pop('definition_version')
            weight = settings.pop('weight')
            if isinstance(weight,bool) or not isinstance(weight,(int,float)) or not np.isfinite(weight) or weight<=0:
                raise ValueError('Morphology weight must be a finite positive number')
            if settings.pop('dtype') != 'float32':
                raise ValueError('Morphology TF-IDF dtype must be float32')
            settings['ngram_range'] = tuple(settings['ngram_range'])
            transforms.append((name,Pipeline([('shape',MorphologyTransformer(definition_version=version)),
                ('tfidf',TfidfVectorizer(dtype=np.float32,**settings))])))
            weights[name] = weight
            continue
        if name == 'style':
            from .style import RichStyleTransformer
            weight = settings.pop('weight')
            if isinstance(weight, bool) or not isinstance(weight, (int,float)) or not np.isfinite(weight) or weight <= 0:
                raise ValueError('Style weight must be a finite positive number')
            transforms.append((name, RichStyleTransformer(**settings)))
            weights[name] = weight
            continue
        if settings.pop('dtype') != 'float32':
            raise ValueError('Only the verified float32 TF-IDF dtype is supported')
        settings['ngram_range'] = tuple(settings['ngram_range'])
        transforms.append((name, TfidfVectorizer(dtype=np.float32, **settings)))
    return FeatureUnion(transforms, transformer_weights=weights or None)
