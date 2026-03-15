#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build_maca_ninja}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_TARGET="${BUILD_TARGET:-unittest}"
BUILD_JOBS="${BUILD_JOBS:-4}"

: "${MACA_PATH:=/opt/maca}"
: "${CUCC_PATH:=$MACA_PATH/tools/cu-bridge}"
: "${CUDA_PATH:=$CUCC_PATH}"

export MACA_PATH
export CUCC_PATH
export CUDA_PATH
export PATH="${CUCC_PATH}/tools:${CUCC_PATH}/bin:${PATH}"
export CUCC_CMAKE_ENTRY="${CUCC_CMAKE_ENTRY:-2}"
export LIBRARY_PATH="${CUCC_PATH}/lib:/opt/mxdriver/lib:${LIBRARY_PATH:-}"
export CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"

if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  cache_generator="$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "$BUILD_DIR/CMakeCache.txt" | head -n 1)"
  if [[ -n "$cache_generator" && "$cache_generator" != "$CMAKE_GENERATOR" ]]; then
    echo "Existing build directory '$BUILD_DIR' uses generator '$cache_generator', expected '$CMAKE_GENERATOR'." >&2
    echo "Use a fresh BUILD_DIR or remove the old CMake cache before reconfiguring." >&2
    exit 1
  fi
fi

cmake_maca -S "$ROOT_DIR" -B "$BUILD_DIR" -G "$CMAKE_GENERATOR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
ninja_maca -C "$BUILD_DIR" -j"$BUILD_JOBS" "$BUILD_TARGET"
