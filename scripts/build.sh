#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-}"
BUILD_TESTS="${BUILD_TESTS:-ON}"
GENERATOR="${GENERATOR:-Ninja}"
USE_CCACHE="${USE_CCACHE:-ON}"
USE_MOLD="${USE_MOLD:-ON}"

cmake_args=(
  -G "$GENERATOR"
  -S "$ROOT_DIR"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
)

if [[ -n "$CUDA_ARCHITECTURES" ]]; then
  cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES")
fi

if [[ "$BUILD_TESTS" == "OFF" ]]; then
  cmake_args+=(-DNO_UNITTEST=ON)
fi

if [[ "$USE_CCACHE" == "ON" ]] && command -v ccache >/dev/null 2>&1; then
  cmake_args+=(
    -DCMAKE_C_COMPILER_LAUNCHER=ccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
  )
fi

if [[ "$USE_MOLD" == "ON" ]] && command -v mold >/dev/null 2>&1; then
  mold_dir="$(dirname "$(command -v mold)")"
  cmake_args+=(
    "-DCMAKE_EXE_LINKER_FLAGS=-B${mold_dir}"
    "-DCMAKE_SHARED_LINKER_FLAGS=-B${mold_dir}"
    "-DCMAKE_MODULE_LINKER_FLAGS=-B${mold_dir}"
  )
fi

cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j
