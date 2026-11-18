"""Generate a standalone CPU submission from authoritative baseline functions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'notebooks/99_submission.ipynb'
BASELINE = ROOT / 'artifacts/s04_baseline_v1_20261118_01'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract(source, names):
    """Keep selected top-level definitions verbatim, in their original order."""
    wanted, found, segments = set(names), set(), []
    for node in ast.parse(source).body:
        keys = {node.name} if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else set()
        if isinstance(node, ast.Assign):
            keys = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if keys & wanted:
            segments.append(ast.get_source_segment(source, node))
            found.update(keys & wanted)
    if found != wanted:
        raise ValueError(f'Cannot extract definitions: missing={sorted(wanted - found)}')
    return '\n\n\n'.join(segments) + '\n'


IMPORTS = '''from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import warnings
import numpy as np
import pandas as pd
import pyarrow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

runtime_versions = {name: importlib.metadata.version(name)
                    for name in ('numpy', 'pandas', 'scikit-learn', 'scipy', 'pyarrow')}
print('ENVIRONMENT', json.dumps({'python': platform.python_version(), **runtime_versions}))
'''

PATHS = '''# Explicit overrides are for controlled local runs or a verified Kaggle mount.
# Leave both unset for the normal Kaggle submission.
KAGGLE_DATA = Path('/kaggle/input/competitions/name-that-llm')
REQUIRED_DATA_FILES = ('train.csv', 'test.csv')

def resolve_submission_data(explicit_dir=None):
    def missing(directory):
        return [name for name in REQUIRED_DATA_FILES if not (directory / name).is_file()]
    if explicit_dir is not None:
        if not str(explicit_dir).strip():
            raise ValueError('NAME_THAT_LLM_DATA_DIR cannot be empty')
        directory = Path(explicit_dir).expanduser().resolve()
        absent = missing(directory)
        if absent:
            raise FileNotFoundError(f'Explicit data directory {directory}: missing {absent}; no fallback used')
        return directory
    if not missing(KAGGLE_DATA):
        return KAGGLE_DATA.resolve()
    # A mount can differ between notebook environments. Do not guess among datasets.
    input_root = Path('/kaggle/input')
    candidates = sorted({p.parent.resolve() for p in input_root.rglob('train.csv')
                         if not missing(p.parent)}) if input_root.is_dir() else []
    if len(candidates) == 1:
        print('DISCOVERED_INPUT_PAIR; training SHA-256 will be checked:', candidates[0])
        return candidates[0]
    raise FileNotFoundError(
        f'Competition inputs unavailable or ambiguous. Official candidate {KAGGLE_DATA}: '
        f'missing {missing(KAGGLE_DATA)}; complete candidate directories={list(map(str, candidates))}. '
        'Attach the official competition data and inspect Input; set NAME_THAT_LLM_DATA_DIR explicitly if needed.')

DATA_DIR = resolve_submission_data(os.environ.get('NAME_THAT_LLM_DATA_DIR'))
output_override = os.environ.get('NAME_THAT_LLM_OUTPUT_DIR')
if output_override is not None and not output_override.strip():
    raise ValueError('NAME_THAT_LLM_OUTPUT_DIR cannot be empty')
OUTPUT_DIR = (Path(output_override).expanduser() if output_override is not None else
              Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()).resolve()
print('DATA_DIR', DATA_DIR)
print('OUTPUT_DIR', OUTPUT_DIR)
'''


def build():
    research = nbformat.read(ROOT / 'notebooks/01_baseline.ipynb', as_version=4)
    s04 = (ROOT / 'scripts/run_s04.py').read_text(encoding='utf-8')
    s05 = (ROOT / 'scripts/run_s05.py').read_text(encoding='utf-8')
    checker = (ROOT / 'scripts/check_s05.py').read_text(encoding='utf-8')
    config = json.loads((BASELINE / 'config.json').read_text(encoding='utf-8'))
    meta = json.loads((BASELINE / 'run_metadata.json').read_text(encoding='utf-8'))
    if meta['status'] != 'validated' or meta['mode'] != 'formal':
        raise ValueError('An independently validated formal S04 baseline is required for generation')
    train_sha = next(v for k, v in meta['input_sha256'].items() if Path(k).name == 'train.csv')
    contract = extract(research.cells[2].source, ['OFFICIAL_MODEL_LABELS', 'class_order', '_locations',
        '_validate_fields', 'validate_train', 'validate_test', 'read_input_csv', 'file_fingerprint'])
    factory = extract(research.cells[3].source, ['RANDOM_SEED', 'make_model'])
    parameters = extract(s04, ['normalise', 'model_config'])
    fitting = extract(s05, ['fit_full_model', 'model_state_fingerprint'])
    prediction = extract(s05, ['predict_test', 'build_submission'])
    submission = extract(checker, ['validate_submission', 'write_and_verify_submission'])
    # Verify the actual extracted factory without reading project data or fitting.
    ns = {}; exec(IMPORTS + contract + factory + parameters, ns)
    if ns['model_config'](ns['make_model']()) != config['model_parameters']:
        raise ValueError('Extracted factory differs from formal S04 parameters')
    if list(ns['class_order']) != config['class_order'] or ns['RANDOM_SEED'] != config['random_seed']:
        raise ValueError('Extracted labels/seed differ from S04')
    inputs = ['notebooks/01_baseline.ipynb', 'scripts/run_s04.py', 'scripts/run_s05.py',
              'scripts/check_s05.py', 'scripts/build_submission_notebook.py']
    provenance = {'generator': 'scripts/build_submission_notebook.py', 'baseline_run_id': meta['run_id'],
        'source_sha256': {name: sha256(ROOT / name) for name in inputs},
        'baseline_config_sha256': sha256(BASELINE / 'config.json'),
        'extraction': {'input_and_labels': '01_baseline cell 2', 'model': '01_baseline cell 3',
            'parameter_normalisation': 'run_s04.py', 'fit_and_prediction': 'run_s05.py',
            'submission_contract': 'check_s05.py'}}
    cells = []
    def section(title, description, code):
        cells.append(nbformat.v4.new_markdown_cell(f'## {title}\n\n{description}'))
        cells.append(nbformat.v4.new_code_cell(code.rstrip() + '\n'))
    cells.append(nbformat.v4.new_markdown_cell(
        '# Name That LLM — CPU submission\n\n'
        '由 scripts/build_submission_notebook.py 生成，请修改权威源码后重新生成。'
        '仅含全量拟合、运行时预测及严格写后校验。CPU，Internet Off；无需本地项目或模型。'
        '本地验收不等于 Kaggle 运行或提交成功。\n\n'
        '官方依据：[比赛概览](https://www.kaggle.com/competitions/name-that-llm/overview/description)、'
        '[数据](https://www.kaggle.com/competitions/name-that-llm/data)。'
        'Rules、账号资格、团队和配额请在正式提交前人工确认。'))
    section('1 依赖和环境', '使用环境已安装的依赖，不下载、不安装。打印实际版本；配置不兼容时明确失败。', IMPORTS)
    section('2 数据与输出路径',
        '优先显式 NAME_THAT_LLM_DATA_DIR，再查官方挂载；发现多个候选时要求明确指定。'
        'Kaggle 输出目录为 /kaggle/working。本地请在新目录执行或设置 NAME_THAT_LLM_OUTPUT_DIR。', PATHS)
    section('3 输入契约和官方标签',
        '标签和输入函数从研究 Notebook 提取；保留原始文本、字符串 ID 和顺序。'
        '训练 SHA 是已验证数据版本检查；测试不限制行数/ID/指纹，也不读取样例占位值。',
        contract + f"\nEXPECTED_TRAIN_SHA256 = {train_sha!r}\n" + '''train_path, test_path = DATA_DIR / 'train.csv', DATA_DIR / 'test.csv'
train_fingerprint = file_fingerprint(train_path)
if train_fingerprint['sha256'] != EXPECTED_TRAIN_SHA256:
    raise ValueError('Official training version differs from S04; inspect data attachment/update before changing the baseline')
train = read_input_csv(train_path, 'train')
test = read_input_csv(test_path, 'test')
if len(train) != 2699 or train['prompt_id'].nunique() != 150:
    raise ValueError('S04 official training version expects 2699 records and 150 prompts')
test_fingerprint = file_fingerprint(test_path)
print('INPUT_CHECKED', len(train), train['prompt_id'].nunique(), len(class_order), 'test rows', len(test))
''')
    section('4 创建已验证基线',
        '唯一模型工厂来自原基线；EXPECTED_MODEL_PARAMETERS 为 S04 生成时的完整配置断言，'
        '不用于另建模型。若依赖导致参数差异，停止并报告，不静默替换。',
        factory + parameters + '\nEXPECTED_MODEL_PARAMETERS = ' + repr(config['model_parameters']) + '''
try:
    probe = make_model()
    actual_parameters = model_config(probe)
except (TypeError, ValueError) as exc:
    raise RuntimeError(f'Baseline construction incompatible with environment {runtime_versions}: {exc}') from exc
if actual_parameters != EXPECTED_MODEL_PARAMETERS:
    differences = sorted(k for k in set(actual_parameters) | set(EXPECTED_MODEL_PARAMETERS)
                         if actual_parameters.get(k) != EXPECTED_MODEL_PARAMETERS.get(k))
    raise RuntimeError(f'Baseline parameter compatibility mismatch: {differences}; versions={runtime_versions}')
print('FULL_MODEL_PARAMETERS_MATCH_S04')
''')
    section('5 全量训练', '创建未拟合 Pipeline；只用训练 text 和监督标签。记录真实耗时和告警，不运行 CV/EDA。',
        fitting + '''
with warnings.catch_warnings(record=True) as fit_warnings:
    warnings.simplefilter('always')
    started = time.perf_counter()
    model = fit_full_model(make_model, train)
    fit_seconds = time.perf_counter() - started
if model_config(model) != EXPECTED_MODEL_PARAMETERS or set(model.classes_) != set(class_order):
    raise RuntimeError('Fitted model parameters/classes differ from the verified baseline')
fitted_state = model_state_fingerprint(model)
for warning in fit_warnings:
    print('FIT_WARNING', warning.category.__name__, str(warning.message))
print('FULL_TRAIN_FINISHED', fit_seconds, 'seconds; warnings', len(fit_warnings))
''')
    section('6 运行时预测和提交契约',
        '复用 S05 预测及严格提交函数。只以运行时 test.csv 的 id/text 为准，'
        '不调用 fit；决策分数按 class_order 对齐并留在内存，不作为概率或额外提交列。',
        submission + prediction + '''
with warnings.catch_warnings(record=True) as prediction_warnings:
    warnings.simplefilter('always')
    started = time.perf_counter()
    predictions, decision_scores = predict_test(model, test, {'validate_test': validate_test, 'class_order': class_order})
    prediction_seconds = time.perf_counter() - started
if model_state_fingerprint(model) != fitted_state:
    raise RuntimeError('Prediction changed the fitted model state')
submission = build_submission(test, predictions, class_order)
for warning in prediction_warnings:
    print('PREDICT_WARNING', warning.category.__name__, str(warning.message))
''')
    section('7 写出、回读与结束标记',
        '仅输出 submission.parquet；真实回读后检查两列字符串、完整唯一 ID、顺序、合法标签及写前一致。'
        '已有输出拒绝覆盖，重新运行请用新会话或仅清理自己生成的候选输出。', '''submission_path = OUTPUT_DIR / 'submission.parquet'
reloaded_submission = write_and_verify_submission(submission, test, class_order, submission_path, [DATA_DIR])
runtime_summary = {'status': 'local_runtime_validated', 'n_train': len(train), 'n_prompts': int(train['prompt_id'].nunique()),
    'n_test': len(test), 'fit_seconds': fit_seconds, 'prediction_seconds': prediction_seconds,
    'fit_warning_count': len(fit_warnings), 'prediction_warning_count': len(prediction_warnings),
    'python': platform.python_version(), 'versions': runtime_versions, 'class_order': list(class_order),
    'model_parameters_sha256': hashlib.sha256(json.dumps(actual_parameters, sort_keys=True).encode()).hexdigest(),
    'train_sha256': train_fingerprint['sha256'], 'test_sha256': test_fingerprint['sha256'],
    'submission_path': str(submission_path), 'submission_sha256': file_fingerprint(submission_path)['sha256'],
    'write_read_equal': True, 'fitted_state_unchanged': True}
print('SUBMISSION_VALIDATED', json.dumps(runtime_summary, ensure_ascii=False))
''')
    for i, cell in enumerate(cells):
        cell.id = f's06a-{i:02}'
        if cell.cell_type == 'code': ast.parse(cell.source)
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'name_that_llm': provenance})
    nbformat.validate(notebook)
    return nbformat.writes(notebook), provenance


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    payload, provenance = build()
    manifest = {**provenance, 'notebook_sha256': hashlib.sha256(payload.encode()).hexdigest()}
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + '\n'
    manifest_path = TARGET.with_suffix('.sources.json')
    if args.check:
        if TARGET.read_bytes() != payload.encode('utf-8') or manifest_path.read_bytes() != manifest_text.encode('utf-8'):
            raise SystemExit('Generated Notebook/manifest differs; review sources and regenerate')
        print('GENERATED_NOTEBOOK_CURRENT')
    else:
        if TARGET.exists() and nbformat.read(TARGET, as_version=4).metadata.get('name_that_llm', {}).get('generator') != 'scripts/build_submission_notebook.py':
            raise FileExistsError('Refusing to overwrite a Notebook not owned by this generator')
        TARGET.write_text(payload, encoding='utf-8', newline='\n')
        manifest_path.write_text(manifest_text, encoding='utf-8', newline='\n')
        print('GENERATED', TARGET)


if __name__ == '__main__':
    main()
