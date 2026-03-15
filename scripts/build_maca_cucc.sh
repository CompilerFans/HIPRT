#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build_maca_make}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_TARGET="${BUILD_TARGET:-unittest}"
BUILD_JOBS="${BUILD_JOBS:-4}"
HIPRT_RUNTIME_KERNEL_CACHE="${HIPRT_RUNTIME_KERNEL_CACHE:-OFF}"

: "${MACA_PATH:=/opt/maca}"
: "${CUCC_PATH:=$MACA_PATH/tools/cu-bridge}"
: "${CUDA_PATH:=$CUCC_PATH}"

export MACA_PATH
export CUCC_PATH
export CUDA_PATH
export PATH="${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}"
export CUCC_CMAKE_ENTRY="${CUCC_CMAKE_ENTRY:-2}"
export LIBRARY_PATH="${CUCC_PATH}/lib:/opt/mxdriver/lib:${LIBRARY_PATH:-}"

GENERATOR="${CMAKE_GENERATOR:-}"
if [[ "$GENERATOR" == "Ninja" ]]; then
  echo "For maca_dev, Ninja/ninja_maca is currently unstable for runtime JIT." >&2
  echo "Use Unix Makefiles + make_maca, or unset CMAKE_GENERATOR." >&2
  exit 1
fi

if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  expected_generator="${GENERATOR:-Unix Makefiles}"
  cache_generator="$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "$BUILD_DIR/CMakeCache.txt" | head -n 1)"
  if [[ -n "$cache_generator" && "$cache_generator" != "$expected_generator" ]]; then
    echo "Existing build directory '$BUILD_DIR' uses generator '$cache_generator', expected '$expected_generator'." >&2
    echo "Use a fresh BUILD_DIR or remove the old CMake cache before reconfiguring." >&2
    exit 1
  fi
fi

cmake_args=(
  -S "$ROOT_DIR"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DHIPRT_ENABLE_RUNTIME_KERNEL_CACHE="$HIPRT_RUNTIME_KERNEL_CACHE"
)

if [[ -n "$GENERATOR" && "$GENERATOR" != "Unix Makefiles" ]]; then
  cmake_args+=(-G "$GENERATOR")
fi

cmake_maca "${cmake_args[@]}"
pushd "$BUILD_DIR" >/dev/null
make_maca -j"$BUILD_JOBS" "$BUILD_TARGET"
popd >/dev/null
