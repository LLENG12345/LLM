"""Compatibility CLI: validation now lives in ntl."""
import argparse
from ntl.validation import close, validate_run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('run_directory')
    args = parser.parse_args(); validate_run(args.run_directory)
    print('S04_SAVED_RUN_VALIDATION_PASS', args.run_directory)
