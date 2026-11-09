#!/usr/bin/env bash
# Build on the AMD PC; the archive's pre-existing engine binary is not this build.
set -euo pipefail
strata_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
strata_build="$strata_root/build-memory-fix-hip"
strata_vision_build="$strata_root/build-memory-fix-vision"
strata_arch="${STRATA_HIP_ARCH:-gfx1030}"
strata_jobs="${STRATA_BUILD_JOBS:-2}"

if ! command -v cmake >/dev/null; then
    printf '%s\n' 'cmake is required. Activate the existing Strata .venv, or install cmake, then retry.' >&2
    exit 1
fi
if [[ ! -f "$strata_root/third_party/llama.cpp/ggml/CMakeLists.txt" ]]; then
    printf '%s\n' 'The bundled third_party/llama.cpp checkout is missing.' >&2
    exit 1
fi

# gfx1030 and MMQ=ON match the supplied build-hip/CMakeCache.txt. Use a fresh
# folder: the uploaded build cache contains paths and compilers from another PC.
cmake -S "$strata_root" -B "$strata_build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
    -DCMAKE_HIP_ARCHITECTURES="$strata_arch" \
    -DSTRATA_PREFILL_MMQ=ON -DSTRATA_PORTABLE=OFF \
    -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CONVERSATION_TESTS=OFF \
    -DSTRATA_GGML_DIR="$strata_root/third_party/llama.cpp"
cmake --build "$strata_build" --target strata -j "$strata_jobs"
"$strata_build/strata" --help >/dev/null 2>&1

# The encoder needs the matching mtmd workspace-release API. Keep its old native
# compiler where available instead of reusing an uploaded absolute build cache.
strata_vision_compilers=()
for strata_language in C CXX; do
    strata_compiler=""
    if [[ -f "$strata_root/build-vision/CMakeCache.txt" ]]; then
        strata_compiler="$(sed -n "s/^CMAKE_${strata_language}_COMPILER:FILEPATH=//p" "$strata_root/build-vision/CMakeCache.txt")"
    fi
    if [[ -n "$strata_compiler" && -x "$strata_compiler" ]]; then
        strata_vision_compilers+=("-DCMAKE_${strata_language}_COMPILER=$strata_compiler")
    fi
done
cmake -S "$strata_root/tools/vision" -B "$strata_vision_build" \
    -DCMAKE_BUILD_TYPE=Release -DSTRATA_PORTABLE=OFF -DSTRATA_VISION_CUDA=OFF \
    -DSTRATA_VISION_WORKSPACE_TESTS=OFF \
    -DLLAMA_DIR="$strata_root/third_party/llama.cpp" "${strata_vision_compilers[@]}"
cmake --build "$strata_vision_build" --target strata-vision -j "$strata_jobs"
"$strata_vision_build/bin/strata-vision" --help >/dev/null 2>&1

mkdir -p "$strata_root/engine"
strata_new="$(mktemp "$strata_root/engine/.strata-new.XXXXXX")"
strata_vision_new="$(mktemp "$strata_root/engine/.strata-vision-new.XXXXXX")"
trap 'rm -f -- "$strata_new" "$strata_vision_new"' EXIT
cp -- "$strata_build/strata" "$strata_new"
cp -- "$strata_vision_build/bin/strata-vision" "$strata_vision_new"
chmod 755 "$strata_new" "$strata_vision_new"
for strata_binary in strata strata-vision; do
    if [[ -f "$strata_root/engine/$strata_binary" ]]; then
        strata_backup="$(mktemp "$strata_root/engine/$strata_binary.pre-memory-fix.XXXXXX")"
        cp -p -- "$strata_root/engine/$strata_binary" "$strata_backup"
        printf 'Previous binary saved as %s\n' "$strata_backup"
    fi
done
mv -f -- "$strata_new" "$strata_root/engine/strata"
mv -f -- "$strata_vision_new" "$strata_root/engine/strata-vision"
if [[ -f "$strata_root/strata-coder-uncen-iq1_m.json" ]]; then
    strata_python=python3
    if [[ -x "$strata_root/.venv/bin/python" ]]; then strata_python="$strata_root/.venv/bin/python"; fi
    "$strata_python" "$strata_root/tools/configure_memory_lifecycle.py"
fi
printf '%s\n' 'Built and installed the engine and CPU vision encoder.' \
    'Start the existing run-coder-uncen-iq1_m.sh launcher after stopping the old server.'
