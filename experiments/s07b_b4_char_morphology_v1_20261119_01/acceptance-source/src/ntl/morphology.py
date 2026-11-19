r"""Independent shape03-v1 implementation of the pinned 03 character mapping.

Unicode alphanumeric/non-decimal numeric characters follow the reference's
[^\W\d_] class; whitespace and punctuation are preserved by this converter.
TF-IDF's own char analyzer separately normalizes repeated whitespace.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

MORPHOLOGY_VERSION = 'shape03-v1'


def morphology_defaults():
    return {'definition_version': MORPHOLOGY_VERSION, 'weight': 0.65,
        'analyzer': 'char', 'ngram_range': [2, 6], 'min_df': 2, 'max_df': 1.0,
        'max_features': 20000, 'lowercase': False, 'sublinear_tf': True,
        'dtype': 'float32', 'norm': 'l2', 'use_idf': True, 'smooth_idf': True}


def morphology_text(text):
    if not isinstance(text, str):
        raise ValueError('Morphology input must be a string')
    # isdecimal() matches Unicode Nd (regex \d), rather than broader isdigit().
    # isalnum() includes numeric Unicode characters such as superscripts/Roman
    # numerals; those are in the reference letter-like regex class as well.
    return ''.join('0' if c.isdecimal() else ('A' if c.isupper() else 'a') if c.isalnum() else c for c in text)


def _batch(values):
    if isinstance(values,(str,bytes)) or getattr(values,'ndim',1)!=1:
        raise ValueError('Morphology batch must be a one-dimensional string sequence')
    try:
        texts=list(values)
    except TypeError as exc:
        raise ValueError('Morphology batch must be iterable') from exc
    for i,text in enumerate(texts):
        if not isinstance(text,str):
            raise ValueError(f'Morphology row {i} must be a string, got {type(text).__name__}')
    return texts


def morphology_batch(values):
    return [morphology_text(text) for text in _batch(values)]


def morphology_input_digest(values):
    return hashlib.sha256(json.dumps(list(values),ensure_ascii=False,separators=(',',':')).encode('utf-8')).hexdigest()


class MorphologyTransformer(TransformerMixin,BaseEstimator):
    """No learned transformation; fit records input identity solely for auditing."""
    def __init__(self,definition_version=MORPHOLOGY_VERSION):
        self.definition_version=definition_version

    def fit(self,X,y=None):
        if self.definition_version!=MORPHOLOGY_VERSION:
            raise ValueError('Unsupported morphology definition version')
        texts=_batch(X)
        self.n_fit_samples_=len(texts)
        self.fit_text_sha256_=morphology_input_digest(texts)
        return self

    def transform(self,X):
        if self.definition_version!=MORPHOLOGY_VERSION:
            raise ValueError('Unsupported morphology definition version')
        return morphology_batch(X)

    def fit_transform(self,X,y=None,**fit_params):
        if fit_params:raise ValueError('Morphology does not accept extra fit parameters')
        texts=_batch(X)
        return self.fit(texts,y).transform(texts)


def morphology_state_digest(branch):
    vec=branch.named_steps['tfidf']
    pairs=sorted((text,int(index)) for text,index in vec.vocabulary_.items())
    h=hashlib.sha256(json.dumps(pairs,ensure_ascii=False,separators=(',',':')).encode('utf-8'))
    h.update(vec.idf_.tobytes())
    return h.hexdigest()


def write_morphology_audit(branch,fit_texts,folder,weight):
    """Save actual fold vocabulary/IDF without additional feature/model fitting."""
    import pandas as pd
    from .io import write_json
    converter,vec=branch.named_steps['shape'],branch.named_steps['tfidf']
    assert converter.n_fit_samples_==len(fit_texts)
    assert converter.fit_text_sha256_==morphology_input_digest(fit_texts)
    assert vec.lowercase is False and np.isfinite(vec.idf_).all()
    n=converter.n_fit_samples_
    df_float=(n+1)/np.exp(vec.idf_.astype(np.float64)-1)-1
    df=np.rint(df_float).astype(int)
    assert np.max(np.abs(df_float-df))<.01 and (df>=2).all() and (df<=n).all()
    names=vec.get_feature_names_out()
    pd.DataFrame({'index':np.arange(len(names)),'ngram':names,'idf':vec.idf_,
        'document_frequency_from_smoothed_idf':df}).to_csv(Path(folder)/'morphology_vocabulary.csv',index=False,lineterminator='\n')
    write_json(Path(folder)/'morphology_audit.json',{'definition_version':MORPHOLOGY_VERSION,
        'n_fit_samples':n,'raw_fit_text_sha256':converter.fit_text_sha256_,
        'converted_fit_text_sha256':morphology_input_digest(morphology_batch(fit_texts)),
        'vocabulary_size':len(names),'idf_dtype':str(vec.idf_.dtype),'weight':weight,
        'contains_uppercase_A':any('A' in name for name in names),'contains_lowercase_a':any('a' in name for name in names),
        'learned_state_sha256':morphology_state_digest(branch),'vocabulary_idf_unchanged_after_validation':True,
        'df_origin':'derived by inverse smoothed-IDF formula and integer rounding; not raw term frequency'})
