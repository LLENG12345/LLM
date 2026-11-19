"""Low-cost existing EDA; no filtering, features or model fitting."""
import pandas as pd


def basic_analysis(train, test):
    duplicate_text = train['text'].duplicated(keep=False)
    conflict_text = train.groupby('text')['model'].transform('nunique') > 1
    duplicate_pair = train.duplicated(['prompt_id', 'model'], keep=False)
    counts = train.groupby('prompt_id').size()
    issues = {'duplicate_ids': train.loc[train['id'].duplicated(keep=False), 'id'].tolist(),
              'duplicate_text_ids': train.loc[duplicate_text, 'id'].tolist(),
              'conflicting_text_ids': train.loc[conflict_text, 'id'].tolist(),
              'duplicate_prompt_model_ids': train.loc[duplicate_pair, 'id'].tolist(),
              'train_test_id_overlap': sorted(set(train['id']) & set(test['id'])),
              'train_test_text_overlap_ids': train.loc[train['text'].isin(test['text']), 'id'].tolist()}
    lengths = pd.DataFrame({'id': train['id'], 'model': train['model'],
                            'characters': train['text'].str.len(),
                            'words': train['text'].str.split().str.len(),
                            'newlines': train['text'].str.count('\n'),
                            'ascii_punctuation': train['text'].str.count(r'[.,;:!?]')})
    by_model = lengths.groupby('model')[['characters', 'words', 'newlines', 'ascii_punctuation']].agg(['mean', 'median', 'min', 'max'])
    return {'rows': len(train), 'test_example_rows': len(test),
            'class_counts': {str(k): int(v) for k, v in train['model'].value_counts().sort_index().items()},
            'prompt_count': int(len(counts)),
            'prompt_size_distribution': {str(k): int(v) for k, v in counts.value_counts().sort_index().items()},
            'issues': issues, 'length_summary': lengths[['characters', 'words', 'newlines', 'ascii_punctuation']].describe().to_dict(),
            'near_duplicate_status': 'deferred; only exact comparisons performed'}, lengths, by_model
