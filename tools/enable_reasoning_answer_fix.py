#!/usr/bin/env python3
"""Enable one bounded continuation for a thinking-only completion."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import time


def enable(path, *, source=None):
    path = Path(path).resolve()
    source = Path(source) if source else Path(__file__).resolve().parents[1] / 'serve/server.py'
    code = source.read_bytes()
    if b'reasoning_close_retry v2' not in code or b'class IncompleteAnswer' not in code:
        raise ValueError('Extract the matching server update before enabling the answer fix')
    before = json.loads(path.read_text())
    if not isinstance(before, dict): raise ValueError('The config must be a JSON object')
    after = copy.deepcopy(before)
    after['reasoning_close_retry'] = True
    backup = path.with_name(path.name + '.before-reasoning-answer-' + str(time.time_ns()))
    shutil.copy2(path, backup)
    fd, temporary = tempfile.mkstemp(prefix='.strata-reasoning-answer-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as output:
            json.dump(after, output, indent=2); output.write('\n'); output.flush(); os.fsync(output.fileno())
        shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return after, backup


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'strata-swift-abl-iq2_xs.json')
    args = parser.parse_args()
    try:
        config, backup = enable(args.config)
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + '\n')
    print('Thinking-only completion recovery enabled (reasoning_close_retry v2).')
    print('Config backup:', backup)
    print('Restart the Strata server to load the updated config.')
