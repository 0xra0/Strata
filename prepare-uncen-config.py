#!/usr/bin/env python3
"""Create a separate DatKurai Uncen Strata config from the user's working Coder config.

Uses only the Python standard library; never overwrites the original configuration.
"""
import json
import sys
from pathlib import Path

ROOT = Path('/mnt/ssd/Strata')
DATA = Path('/mnt/ssd/Strata-data')
ORIGINAL = ROOT / 'strata-coder-iq1_m.json'
OUTPUT = ROOT / 'strata-coder-uncen-iq1_m.json'
MODEL_DIR = DATA / 'models' / 'coder-uncen-IQ1_M'
BASENAME = 'Qwen3.8-Flash-Next-GSQ-RCO-Coder-Uncen-IQ1_M'
SHARD1 = MODEL_DIR / f'{BASENAME}-00001-of-00002.gguf'
SHARD2 = MODEL_DIR / f'{BASENAME}-00002-of-00002.gguf'
PROFILE = MODEL_DIR / 'expert-profile-coder.bin'
PACK = DATA / 'packs' / 'coder-uncen-iq1_m'
TOK = PACK / 'tokenizer'


def fail(message):
    sys.exit(f'ERROR: {message}')


def main():
    if not ORIGINAL.is_file():
        fail(f'Working Coder config not found: {ORIGINAL}')
    if OUTPUT.exists():
        fail(f'Output already exists, refusing to overwrite: {OUTPUT}')

    needed = (SHARD1, SHARD2, PROFILE, PACK / 'index.txt')
    for item in needed:
        if not item.is_file():
            fail(f'Missing input file: {item}')
    if not TOK.is_dir():
        fail(f'Missing native tokenizer directory: {TOK}')

    try:
        config = json.loads(ORIGINAL.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        fail(f'Cannot read working config: {exc}')

    args = config.get('args')
    if not isinstance(args, list):
        fail('Expected an args array in working config')

    replacements = {
        '--pack': str(PACK),
        '--native': str(SHARD1),
        '--ple-gguf': str(SHARD2),
        '--expert-profile': str(PROFILE),
    }
    for flag, new_value in replacements.items():
        indices = [i for i, a in enumerate(args) if a == flag]
        if len(indices) != 1 or indices[0] + 1 >= len(args):
            fail(f'Expected exactly one {flag} followed by its value in original args')
        args[indices[0] + 1] = new_value

    # Everything else in args is copied unchanged: MTP, spec, KV, context,
    # expert strategy, prefill, GPU VRAM reserve, and the --vision flag.
    config['tokenizer'] = str(TOK)
    config['model_name'] = 'qwen3.8-flash-next-coder-uncen-iq1_m'
    config['log'] = str(ROOT / 'strata-coder-uncen-iq1_m.log')
    config['port'] = 8099

    if 'vision' in config and isinstance(config['vision'], dict):
        config['vision']['model'] = str(SHARD1)
        # vision.mmproj, gpu, threads and max_tokens remain unchanged.

    OUTPUT.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Created: {OUTPUT}')
    print('Kept original Coder args (except pack/native/PLE/profile), GPU env, MTP and vision projector.')
    print('API port: 8099. Stop the old Coder server before launching this one.')
    print('WARNING: Existing long-context/CPU-vision OOM limitations may remain.')


if __name__ == '__main__':
    main()
