#!/usr/bin/env python3
"""Enable the bounded RAM working set and independent default thinking allowance."""
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
        raise ValueError('The working-set profile requires the existing --resident-experts mode')
    exe = Path(before['exe'])
    if not exe.is_absolute(): exe = Path(before.get('cwd') or path.parent) / exe
    code = exe.read_bytes()
    for marker in (b'idle GPU shadows: live ownership v2', b'RAM working set v1'):
        if marker not in code: raise ValueError('Build the corrected long-context engine before enabling this profile')
    after = copy.deepcopy(before)
    if after.get('env') is None: after['env'] = {}
    if not isinstance(after['env'], dict): raise ValueError('The config env must be an object')
    after['env']['STRATA_MEMORY_LIFECYCLE'] = '1'
    after['env']['STRATA_RAM_WORKING_SET'] = '1'
    after['env']['STRATA_DECODE_TIMING'] = '1'
    after.setdefault('reasoning_auto_budget_tokens', 4096)
    allowance = after['reasoning_auto_budget_tokens']
    if isinstance(allowance, bool) or not isinstance(allowance, int) or allowance < 0:
        raise ValueError('reasoning_auto_budget_tokens must be a whole number >= 0')
    # Existing explicit reasoning budgets, image settings and all engine arguments
    # remain authoritative. New policy affects only the default allowance.
    assert after['args'] == before['args']
    backup = path.with_name(path.name + '.before-long-context-' + str(time.time_ns()))
    shutil.copy2(path, backup)
    fd, temp = tempfile.mkstemp(prefix='.strata-long-context-', dir=path.parent)
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
    print('Long-context policy enabled; model, context, KV, cache and vision settings preserved.')
    print('Config backup:', backup)
    print('Default thinking allowance:', config['reasoning_auto_budget_tokens'], '(explicit budgets take precedence)')
    if 'STRATA_RAM_ADAPT' in config['env']:
        print('An explicit STRATA_RAM_ADAPT setting is preserved and takes precedence over the new RAM policy.')
