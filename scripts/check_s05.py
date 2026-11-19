"""Compatibility CLI: validation now lives in ntl."""
import argparse
from ntl.validation import validate_s05_run
from ntl.submission import validate_submission, write_and_verify_submission

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('run_directory')
    args = parser.parse_args()
    print(validate_s05_run(args.run_directory).to_string(index=False))
    print('S05_SAVED_RUN_VALIDATION_PASS')
