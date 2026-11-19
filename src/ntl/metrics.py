"""Shared metric definitions; decision values are not probabilities."""
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def scores_summary(true, pred, order):
    return {'balanced_accuracy': balanced_accuracy_score(true, pred),
            'accuracy': accuracy_score(true, pred),
            'macro_f1': f1_score(true, pred, labels=order, average='macro', zero_division=0)}
