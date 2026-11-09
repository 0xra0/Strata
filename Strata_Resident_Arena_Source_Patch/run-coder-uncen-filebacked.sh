#!/bin/sh
set -e
cd /mnt/ssd/Strata
exec /mnt/ssd/Strata/.venv/bin/python /mnt/ssd/Strata/serve/server.py \
  --engine strata \
  --config /mnt/ssd/Strata/strata-coder-uncen-iq1_m-filebacked.json \
  --port 8099
