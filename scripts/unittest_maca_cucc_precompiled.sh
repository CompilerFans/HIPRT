#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export HIPRT_ENABLE_BAKE_KERNEL="${HIPRT_ENABLE_BAKE_KERNEL:-ON}"
export HIPRT_ENABLE_BITCODE="${HIPRT_ENABLE_BITCODE:-ON}"
export HIPRT_ENABLE_PRECOMPILED_TRACE_KERNEL="${HIPRT_ENABLE_PRECOMPILED_TRACE_KERNEL:-ON}"

"$SCRIPT_DIR/build_maca_cucc.sh"

export HIPRT_PATH="${HIPRT_PATH:-$ROOT_DIR}"
export HIPRT_DISABLE_RUNTIME_KERNEL_CACHE="${HIPRT_DISABLE_RUNTIME_KERNEL_CACHE:-1}"
export LD_LIBRARY_PATH="$ROOT_DIR/contrib/embree/linux:${LD_LIBRARY_PATH:-}"

"$ROOT_DIR/dist/bin/Release/unittest64" \
	--width=512 \
	--height=512 \
	--referencePath="$ROOT_DIR/test/references" \
	--gtest_filter="${GTEST_FILTER:-hiprtTest.LoadPrecompiledTraceKernel:hiprtTest.LoadPrecompiledTraceKernelWithCustomFuncTable:hiprtTest.LaunchPrecompiledTraceKernel:hiprtTest.LaunchPrecompiledTraceKernelWithCustomFuncTable}"
