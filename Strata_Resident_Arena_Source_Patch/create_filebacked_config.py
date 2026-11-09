#!/usr/bin/env python3
"""Create a separate DatKurai configuration with exactly the same inference args."""
import copy
import json
from pathlib import Path

root = Path('/mnt/ssd/Strata')
src = root / 'strata-coder-uncen-iq1_m.json'
out = root / 'strata-coder-uncen-iq1_m-filebacked.json'
model = json.loads(src.read_text())
if '--resident-experts' not in model.get('args', []):
    raise SystemExit('Expected --resident-experts in the original configuration. Did not write the new config.')
if out.exists():
    raise SystemExit(f'{out} already exists. Rename it to preserve your previous test.')
model['env'] = dict(model.get('env', {}))
model['env']['STRATA_RESIDENT_PIN'] = '0'
model['env']['STRATA_RESIDENT_FILE_DIR'] = '/mnt/ssd/Strata-data'
model['log'] = '/mnt/ssd/Strata/strata-coder-uncen-iq1_m-filebacked.log'
model['model_name'] = model.get('model_name', 'coder-uncen-iq1_m') + '-filebacked'
# All engine args (context, kv, resident experts, prefill, cache, MTP, GPU reserve) stay unchanged.
# Vision configuration, native pack, expert profile, ROCm env and API port also stay unchanged.
out.write_text(json.dumps(model, indent=2) + '\n')
print('Created', out)
print('Expert/model/MTP/vision/context/prefill/VRAM options unchanged.')
