#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-}"
BUILD_TESTS="${BUILD_TESTS:-ON}"

cmake_args=(
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

cmake "${cmake_args[@]}"
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j
