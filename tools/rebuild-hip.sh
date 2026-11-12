#!/usr/bin/env bash
# Build the HIP engine (and, with --vision, the CPU vision encoder) and install it into engine/.
# Replaces the old rebuild-{memory,ownership,long-context,swap-prefetch,prompt-stream}-fix.sh scripts.
# Installs only after a successful build that contains every feature marker; the previous engine is
# kept as engine/strata.prev (one copy, overwritten each time). The run config is not touched.
set -euo pipefail
strata_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
strata_build="$strata_root/build-memory-fix-hip"
strata_vision_build="$strata_root/build-memory-fix-vision"
strata_arch="${STRATA_HIP_ARCH:-gfx1030}"
strata_jobs="${STRATA_BUILD_JOBS:-2}"
strata_with_vision=0
[[ "${1:-}" == "--vision" ]] && strata_with_vision=1

if ! command -v cmake >/dev/null; then
    printf '%s\n' 'cmake is required. Activate the existing Strata .venv, or install cmake, then retry.' >&2
    exit 1
fi
if [[ ! -f "$strata_root/third_party/llama.cpp/ggml/CMakeLists.txt" ]]; then
    printf '%s\n' 'The bundled third_party/llama.cpp checkout is missing.' >&2
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
for strata_marker in 'idle GPU shadows: live ownership v2' 'RAM working set v1' \
                     'resident swap prefetch v1' 'prompt file reader v1'; do
    if ! LC_ALL=C grep -aFq "$strata_marker" "$strata_build/strata"; then
        printf 'The built engine lacks "%s"; the installed engine was not changed.\n' "$strata_marker" >&2
        exit 1
    fi
done

if (( strata_with_vision )); then
    # Keep the old native compiler where available instead of reusing an uploaded absolute build cache.
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
fi

# Both builds succeeded: stage, keep one backup of each, then swap in.
mkdir -p "$strata_root/engine"
strata_new="$(mktemp "$strata_root/engine/.strata-new.XXXXXX")"
strata_vision_new="$(mktemp "$strata_root/engine/.strata-vision-new.XXXXXX")"
trap 'rm -f -- "$strata_new" "$strata_vision_new"' EXIT
cp -- "$strata_build/strata" "$strata_new"
chmod 755 "$strata_new"
strata_installs=(strata)
if (( strata_with_vision )); then
    cp -- "$strata_vision_build/bin/strata-vision" "$strata_vision_new"
    chmod 755 "$strata_vision_new"
    strata_installs+=(strata-vision)
fi
for strata_binary in "${strata_installs[@]}"; do
    if [[ -f "$strata_root/engine/$strata_binary" ]]; then
        cp -p -- "$strata_root/engine/$strata_binary" "$strata_root/engine/$strata_binary.prev"
        printf 'Previous %s saved as engine/%s.prev\n' "$strata_binary" "$strata_binary"
    fi
done
mv -f -- "$strata_new" "$strata_root/engine/strata"
if (( strata_with_vision )); then mv -f -- "$strata_vision_new" "$strata_root/engine/strata-vision"; fi
printf '%s\n' "Installed: ${strata_installs[*]}. Restart the Strata server to use it."
