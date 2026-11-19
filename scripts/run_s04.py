"""Compatibility CLI for the shared fixed-v1 experiment runner."""
import argparse
import json
from pathlib import Path
from _common import ROOT, load_factory
from ntl.io import digest, write_json
from ntl.models import normalise, model_config, config_hash
from ntl.metrics import scores_summary
from ntl.experiments import run_fold, finalise, run_experiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'formal'], required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--smoke-run')
    parser.add_argument('--data-dir')
    parser.add_argument('--config', type=Path, help='JSON overrides; defaults remain unchanged')
    parser.add_argument('--purpose', default='baseline')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8')) if args.config else None
    run_experiment(args.run_id, args.mode, config, args.data_dir, ROOT, args.smoke_run, args.purpose)

if __name__ == '__main__':
    main()
