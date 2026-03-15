#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-}"
BUILD_TESTS="${BUILD_TESTS:-ON}"
CCACHE_PROGRAM="${CCACHE_PROGRAM:-$(command -v ccache || true)}"
MOLD_PROGRAM="${MOLD_PROGRAM:-$(command -v mold || true)}"
if [[ -z "${CMAKE_GENERATOR:-}" ]] && command -v ninja >/dev/null 2>&1; then
  CMAKE_GENERATOR="Ninja"
fi

cmake_args=(
  -S "$ROOT_DIR"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
)

if [[ -n "${CMAKE_GENERATOR:-}" ]]; then
  cmake_args+=(-G "$CMAKE_GENERATOR")
fi

if [[ -n "$CUDA_ARCHITECTURES" ]]; then
  cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES")
fi

if [[ "$BUILD_TESTS" == "OFF" ]]; then
  cmake_args+=(-DNO_UNITTEST=ON)
fi

if [[ -n "$CCACHE_PROGRAM" ]]; then
  cmake_args+=(
    -DCMAKE_C_COMPILER_LAUNCHER="$CCACHE_PROGRAM"
    -DCMAKE_CXX_COMPILER_LAUNCHER="$CCACHE_PROGRAM"
    -DCMAKE_CUDA_COMPILER_LAUNCHER="$CCACHE_PROGRAM"
  )
fi

if [[ -n "$MOLD_PROGRAM" ]]; then
  MOLD_DIR="$(dirname "$MOLD_PROGRAM")"
  cmake_args+=(
    -DCMAKE_EXE_LINKER_FLAGS="-B$MOLD_DIR"
    -DCMAKE_SHARED_LINKER_FLAGS="-B$MOLD_DIR"
  )
fi

cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j
