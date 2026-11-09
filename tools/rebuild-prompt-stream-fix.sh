#!/usr/bin/env bash
# Incremental engine-only build, keeping the existing inference/vision profile.
set -euo pipefail
strata_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
strata_build="$strata_root/build-memory-fix-hip"
if ! command -v cmake >/dev/null; then
    printf '%s\n' 'Activate the existing Strata .venv (cmake is required), then retry.' >&2
    exit 1
fi
if [[ ! -f "$strata_build/CMakeCache.txt" ]]; then
    printf '%s\n' 'Run tools/rebuild-memory-fix.sh for the initial build first, then retry this script.' >&2
    exit 1
fi
cmake -S "$strata_root" -B "$strata_build" \
    -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
    -DCMAKE_HIP_ARCHITECTURES="${STRATA_HIP_ARCH:-gfx1030}" \
    -DSTRATA_PREFILL_MMQ=ON -DSTRATA_PORTABLE=OFF \
    -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CONVERSATION_TESTS=OFF \
    -DSTRATA_GGML_DIR="$strata_root/third_party/llama.cpp"
cmake --build "$strata_build" --target strata -j "${STRATA_BUILD_JOBS:-2}"
"$strata_build/strata" --help >/dev/null 2>&1
for strata_marker in 'idle GPU shadows: live ownership v2' 'resident swap prefetch v1' 'STRATA_PROMPT_STREAM' 'prompt file reader v1'; do
    if ! LC_ALL=C grep -aFq "$strata_marker" "$strata_build/strata"; then
        printf '%s\n' 'The built engine lacks the correction; the installed engine was not changed.' >&2
        exit 1
    fi
done
strata_new="$(mktemp "$strata_root/engine/.strata-prompt-stream-new.XXXXXX")"
trap 'rm -f -- "$strata_new"' EXIT
cp -- "$strata_build/strata" "$strata_new"
chmod 755 "$strata_new"
strata_backup="$(mktemp "$strata_root/engine/strata.pre-prompt-stream.XXXXXX")"
cp -p -- "$strata_root/engine/strata" "$strata_backup"
mv -f -- "$strata_new" "$strata_root/engine/strata"
printf 'Previous engine saved as %s\n' "$strata_backup"
strata_python=python3
if [[ -x "$strata_root/.venv/bin/python" ]]; then strata_python="$strata_root/.venv/bin/python"; fi
"$strata_python" "$strata_root/tools/configure_prompt_stream.py"
printf '%s\n' 'Engine rebuilt. Restart run-coder-uncen-iq1_m.sh.'
