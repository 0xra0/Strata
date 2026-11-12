#!/usr/bin/env python3
"""Enable cache-safe prompt reads after building its matching engine."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import time


def configure(path):
    path = Path(path).resolve()
    before = json.loads(path.read_text())
    if '--resident-experts' not in before.get('args', []):
        raise ValueError('Prompt streaming requires the existing --resident-experts mode')
    exe = Path(before['exe'])
    if not exe.is_absolute(): exe = Path(before.get('cwd') or path.parent) / exe
    code = exe.read_bytes()
    for marker in (b'idle GPU shadows: live ownership v2', b'resident swap prefetch v1', b'STRATA_PROMPT_STREAM', b'prompt file reader v1'):
        if marker not in code: raise ValueError('Build the prompt-stream engine before enabling this profile')
    after = copy.deepcopy(before)
    if after.get('env') is None: after['env'] = {}
    if not isinstance(after['env'], dict): raise ValueError('The config env must be an object')
    after['env'].setdefault('STRATA_PROMPT_STREAM', '1')
    after['runtime_memory_log'] = True
    assert after['args'] == before['args']
    backup = path.with_name(path.name + '.before-prompt-stream-' + str(time.time_ns()))
    shutil.copy2(path, backup)
    fd, temp = tempfile.mkstemp(prefix='.strata-prompt-stream-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(after, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
        shutil.copymode(path, temp)
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
    return after, backup


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'strata-swift-abl-iq2_xs.json')
    args = p.parse_args()
    try:
        config, backup = configure(args.config)
    except (OSError, ValueError, KeyError) as e: p.exit(1, str(e) + '\n')
    print('Prompt stream profile saved. Model, memory, thinking and vision settings preserved.')
    print('Config backup:', backup)
    if config['env'].get('STRATA_PROMPT_STREAM') == '0':
        print('An explicit STRATA_PROMPT_STREAM=0 was preserved; the new reader is disabled.')
    if config['env'].get('STRATA_PROMPT_DIRECT') == '1':
        print('STRATA_PROMPT_DIRECT=1 was preserved; it takes precedence over the new reader.')
