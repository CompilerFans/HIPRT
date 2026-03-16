#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build_maca2}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_TARGET="${BUILD_TARGET:-unittest}"
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

cmake_maca -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DHIPRT_ENABLE_RUNTIME_KERNEL_CACHE="$HIPRT_RUNTIME_KERNEL_CACHE"
pushd "$BUILD_DIR" >/dev/null
make_maca -j"${BUILD_JOBS:-4}" "$BUILD_TARGET"
popd >/dev/null
