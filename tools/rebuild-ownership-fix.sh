#!/usr/bin/env bash
# Incremental engine-only correction for the memory-lifecycle ownership bug.
set -euo pipefail
strata_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
strata_build="$strata_root/build-memory-fix-hip"
strata_jobs="${STRATA_BUILD_JOBS:-2}"
strata_arch="${STRATA_HIP_ARCH:-gfx1030}"
if ! command -v cmake >/dev/null; then
    printf '%s\n' 'Activate the existing Strata .venv (cmake is required), then retry.' >&2
    exit 1
fi
if [[ ! -f "$strata_build/CMakeCache.txt" ]]; then
    printf '%s\n' 'The previous memory-fix build is missing. Run tools/rebuild-memory-fix.sh for the initial build.' >&2
    exit 1
fi
cmake -S "$strata_root" -B "$strata_build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
    -DCMAKE_HIP_ARCHITECTURES="$strata_arch" \
    -DSTRATA_PREFILL_MMQ=ON -DSTRATA_PORTABLE=OFF \
    -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CONVERSATION_TESTS=OFF \
    -DSTRATA_GGML_DIR="$strata_root/third_party/llama.cpp"
cmake --build "$strata_build" --target strata -j "$strata_jobs"
"$strata_build/strata" --help >/dev/null 2>&1
if ! LC_ALL=C grep -aFq 'idle GPU shadows: live ownership v2' "$strata_build/strata"; then
    printf '%s\n' 'The built engine lacks the ownership correction; the installed engine was not changed.' >&2
    exit 1
fi
mkdir -p "$strata_root/engine"
strata_new="$(mktemp "$strata_root/engine/.strata-ownership-new.XXXXXX")"
trap 'rm -f -- "$strata_new"' EXIT
cp -- "$strata_build/strata" "$strata_new"
chmod 755 "$strata_new"
if [[ -f "$strata_root/engine/strata" ]]; then
    strata_backup="$(mktemp "$strata_root/engine/strata.pre-ownership-fix.XXXXXX")"
    cp -p -- "$strata_root/engine/strata" "$strata_backup"
    printf 'Previous engine saved as %s\n' "$strata_backup"
fi
mv -f -- "$strata_new" "$strata_root/engine/strata"
printf '%s\n' 'Installed the live-ownership correction. Vision and configuration were preserved.' \
    'Restart run-coder-uncen-iq1_m.sh and use New chat to discard the earlier corrupted conversation state.'
