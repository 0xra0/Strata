#!/usr/bin/env bash
# Build the HIP engine (and, with --vision, the CPU picture reader) and install the binaries into engine/.
# Replaces the old rebuild-{memory,ownership,long-context,swap-prefetch,prompt-stream}-fix.sh scripts.
# Installs only after a successful build that contains every feature marker; the previous engine is
# kept as engine/strata.prev (one copy, overwritten each time). The run config is not touched.
set -euo pipefail
strata_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# the source moved to engine/ and terminal/tools/vision/ in 0.2.22: new build folders (CMake refuses a changed source dir)
strata_build="$strata_root/build-local-hip"
strata_vision_build="$strata_root/build-local-vision"
strata_arch="${STRATA_HIP_ARCH:-gfx1030}"
strata_jobs="${STRATA_BUILD_JOBS:-2}"
strata_with_vision=0
[[ "${1:-}" == "--vision" ]] && strata_with_vision=1

if ! command -v cmake >/dev/null; then
    printf '%s\n' 'cmake is required. Activate the existing Strata .venv, or install cmake, then retry.' >&2
    exit 1
fi
# ggml is vendored in engine/third_party/strata-ggml (no llama.cpp checkout needed). The ROCm toolchain is named
# explicitly: a fresh build folder has no cache to inherit it from (STRATA_ROCM overrides /opt/rocm).
strata_rocm="${STRATA_ROCM:-/opt/rocm}"
strata_gcc_dir="$(dirname "$(gcc -print-libgcc-file-name 2>/dev/null || echo /nonexistent/x)")"
strata_hip_flags="--rocm-path=$strata_rocm --rocm-device-lib-path=$strata_rocm/amdgcn/bitcode"
[[ -d "$strata_gcc_dir" ]] && strata_hip_flags+=" --gcc-install-dir=$strata_gcc_dir"
cmake -S "$strata_root/engine" -B "$strata_build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
    -DCMAKE_C_COMPILER="$strata_rocm/llvm/bin/clang" -DCMAKE_CXX_COMPILER="$strata_rocm/llvm/bin/clang++" \
    -DCMAKE_HIP_COMPILER="$strata_rocm/llvm/bin/clang++" -DCMAKE_HIP_COMPILER_ROCM_ROOT="$strata_rocm" \
    -DCMAKE_PREFIX_PATH="$strata_rocm" -DCMAKE_HIP_FLAGS="$strata_hip_flags" \
    -DCMAKE_HIP_ARCHITECTURES="$strata_arch" \
    -DSTRATA_PORTABLE=OFF \
    -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_CONVERSATION_TESTS=OFF
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
    cmake -S "$strata_root/terminal/tools/vision" -B "$strata_vision_build" \
        -DCMAKE_BUILD_TYPE=Release -DSTRATA_PORTABLE=OFF -DSTRATA_VISION_CUDA=OFF \
        "${strata_vision_compilers[@]}"
    cmake --build "$strata_vision_build" --target strata-vision -j "$strata_jobs"
    "$strata_vision_build/bin/strata-vision" --version >/dev/null 2>&1
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
