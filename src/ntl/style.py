"""Independent rich162 statistics, following the pinned 03 method definition.

Reference: 051019zhao/LLM@e83b9e3, 03_optimized_submission.ipynb,
style_features/rich_style/scale_style. No reference source is vendored here.
Only this branch learns across documents: its StandardScaler, within fit().
"""
import hashlib
import re
from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

STYLE_VERSION = 'rich162-v1'
FUNCTION_WORDS = tuple(('a an the this that those these it its there one on in at over under to of for with from by as into '
    'is are was be been being can could may might must shall should will would or and but if while so '
    'because although then than also even just still often only very more most some any such which who '
    'what how when all each both no not like between through about across within without we you i').split())
MARKS = tuple('.,;:!?()[]{}') + ('—', '–', '-', "'", '’', '"', '“', '”', '*', '`', '_', '/', '\n')
_PATTERNS = (
    ('comma_semicolon_colon', r'[,;:]'), ('question_exclamation', r'[?!]'),
    ('unicode_dash', r'[—–]'), ('hyphen', r'-'), ('ellipsis', r'…|\.\.\.'),
    ('curly_quote', r'[’‘“”]'), ('straight_quote', r'''['"]'''),
    ('parenthesis', r'[()]'), ('square_curly_bracket', r'[\[\]{}]'),
    ('markdown_bold_pair', r'\*\*'), ('backtick', r'`'), ('newline', r'\n'),
    ('list_line', r'(?m)^\s*(?:[-*] |\d+\. )'), ('heading_line', r'(?m)^#{1,6} '),
    ('digit', r'\d'), ('ascii_uppercase', r'[A-Z]'),
    ('personal_pronoun', r'(?i)\b(?:i|me|my|we|our|you|your)\b'),
    ('discourse_word', r'(?i)\b(?:however|therefore|indeed|thus|rather|simply)\b'),
)
_COMPILED = tuple(re.compile(pattern) for _, pattern in _PATTERNS)
_TOKEN = re.compile(r"\b[\w’']+\b")
_SENTENCE = re.compile(r'[.!?]+(?:\s|$)')
STYLE_FEATURE_NAMES = (
    'log1p_char_count', 'log1p_safe_word_count', 'word_length_mean', 'word_length_std',
    'sentence_word_count_mean', 'sentence_word_count_std', 'type_token_ratio',
    *(f'pattern_{name}_per100words' for name, _ in _PATTERNS),
    *(f'function_{word}_per100words' for word in FUNCTION_WORDS),
    *(f'mark_u{ord(mark):04x}_per100words' for mark in MARKS),
    *(f'mark_u{ord(mark):04x}_present' for mark in MARKS),
    'sentence_word_count_median', 'first_sentence_word_count', 'last_sentence_word_count',
    'long_word_ge8_fraction', 'short_word_le3_fraction', 'hapax_token_fraction',
)
assert len(STYLE_FEATURE_NAMES) == len(set(STYLE_FEATURE_NAMES)) == 162


def style_defaults():
    return {'weight': 0.075, 'clip': 5.0, 'dtype': 'float32', 'definition_version': STYLE_VERSION}


def _texts(values):
    if isinstance(values, (str, bytes)) or getattr(values, 'ndim', 1) != 1:
        raise ValueError('Style input must be a one-dimensional sequence of strings')
    try:
        texts = list(values)
    except TypeError as exc:
        raise ValueError('Style input must be an iterable of strings') from exc
    for row, value in enumerate(texts):
        if not isinstance(value, str):
            raise ValueError(f'Style input row {row} must be a string, got {type(value).__name__}')
    return texts


def text_sequence_digest(texts):
    h = hashlib.sha256()
    for text in texts:
        raw = text.encode('utf-8')
        h.update(len(raw).to_bytes(8, 'big')); h.update(raw)
    return h.hexdigest()


def rich_style_matrix(values):
    """Stateless, order-preserving float32 N x 162, including zero-row inputs.

    Lowercased tokens are temporary statistics only; original texts are untouched.
    Population std (ddof=0); safe token denominator=max(count,1). Counts are not
    clipped or scaled here. Sentence token counts use original-case substrings.
    """
    texts = _texts(values)
    result = np.empty((len(texts), len(STYLE_FEATURE_NAMES)), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = _TOKEN.findall(text.lower())
        count = len(tokens); denominator = max(count, 1)
        histogram = Counter(tokens)
        word_lengths = np.asarray([len(token) for token in tokens] or [0])
        sentence_sizes = np.asarray([len(_TOKEN.findall(part)) for part in _SENTENCE.split(text) if part.strip()] or [0])
        result[row, :7] = (np.log1p(len(text)), np.log1p(denominator), word_lengths.mean(),
            word_lengths.std(ddof=0), sentence_sizes.mean(), sentence_sizes.std(ddof=0), len(histogram)/denominator)
        result[row, 7:25] = [100 * len(pattern.findall(text))/denominator for pattern in _COMPILED]
        result[row, 25:106] = [100 * histogram[word]/denominator for word in FUNCTION_WORDS]
        result[row, 106:131] = [100 * text.count(mark)/denominator for mark in MARKS]
        result[row, 131:156] = [mark in text for mark in MARKS]
        result[row, 156:] = (np.median(sentence_sizes), sentence_sizes[0], sentence_sizes[-1],
            sum(len(token)>=8 for token in tokens)/denominator,
            sum(len(token)<=3 for token in tokens)/denominator,
            sum(frequency==1 for frequency in histogram.values())/denominator)
    if not np.isfinite(result).all():
        raise ValueError('Style statistics contain a non-finite value')
    return result


class RichStyleTransformer(TransformerMixin, BaseEstimator):
    """Document statistics -> train-only StandardScaler -> symmetric clipping."""
    def __init__(self, clip=5.0, dtype='float32', definition_version=STYLE_VERSION):
        self.clip = clip
        self.dtype = dtype
        self.definition_version = definition_version

    def _fit_raw(self, texts):
        if self.dtype != 'float32' or self.definition_version != STYLE_VERSION:
            raise ValueError('Unsupported rich style dtype or definition version')
        if not isinstance(self.clip, (int,float)) or not np.isfinite(self.clip) or self.clip <= 0:
            raise ValueError('Style clip must be a finite positive number')
        raw = rich_style_matrix(texts)
        self.scaler_ = StandardScaler().fit(raw)
        self.n_features_out_ = raw.shape[1]
        self.fit_text_sha256_ = text_sequence_digest(texts)
        return raw

    def fit(self, X, y=None):
        self._fit_raw(_texts(X))
        return self

    def fit_transform(self, X, y=None, **fit_params):
        if fit_params:
            raise ValueError('Rich style does not accept extra fit parameters')
        raw = self._fit_raw(_texts(X))
        return np.clip(self.scaler_.transform(raw), -self.clip, self.clip)

    def transform(self, X):
        check_is_fitted(self, 'scaler_')
        raw = rich_style_matrix(X)
        if len(raw) == 0:
            return raw
        return np.clip(self.scaler_.transform(raw), -self.clip, self.clip)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(STYLE_FEATURE_NAMES, dtype=object)
