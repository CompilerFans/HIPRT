#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data/HIPRT}"
MACA_PATH="${MACA_PATH:-/opt/maca}"
CUDA_PATH="${CUDA_PATH:-/root/cu-bridge/CUDA_DIR}"
MXCC="${MXCC:-$MACA_PATH/mxgpu_llvm/bin/mxcc}"
OFFLOAD_ARCH="${MXCC_OFFLOAD_ARCH:-xcore1000}"

if [[ ! -d "$CUDA_PATH/lib64" ]]; then
  for candidate in /root/cu-bridge/CUDA_DIR /opt/maca/tools/cu-bridge/CUDA_DIR /opt/maca/tools/cu-bridge; do
    if [[ -d "$candidate/lib64" ]]; then
      CUDA_PATH="$candidate"
      break
    fi
  done
fi

WORK_DIR="$(mktemp -d /tmp/hiprt_mxcc_probe_XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

cat >"$WORK_DIR/probe.cu" <<'EOF'
#define HIPRT_BITCODE_LINKING
#include <cuda_runtime.h>
#include <hiprt/hiprt_device.h>
extern "C" __global__ void K() {}
EOF

"$MXCC" -O3 -std=c++17 -x maca -fgpu-rdc -c \
  --include cuda_runtime.h -D__CUDACC__ \
  -I"$ROOT_DIR" \
  -I"$ROOT_DIR/contrib/Orochi" \
  -I"$CUDA_PATH/include" \
  -I"$MACA_PATH/include" \
  --offload-arch="$OFFLOAD_ARCH" \
  "$WORK_DIR/probe.cu" -o "$WORK_DIR/probe.o"

"$MXCC" -fgpu-rdc --maca-link "$WORK_DIR/probe.o" -fatbin -o "$WORK_DIR/probe_linked.mcfb"

cat >"$WORK_DIR/load_probe.cpp" <<'EOF'
#include <cuda.h>
#include <cstdio>
#include <fstream>
#include <string>
int main(int argc, char** argv){
  if (argc != 2) return 2;
  if(cuInit(0)!=CUDA_SUCCESS){ puts("cuInit fail"); return 3; }
  std::ifstream f(argv[1], std::ios::binary | std::ios::ate);
  if(!f.is_open()){ puts("open fail"); return 4; }
  size_t sz = (size_t)f.tellg();
  f.seekg(0, std::ios::beg);
  std::string buf(sz, '\0');
  f.read(buf.data(), sz);
  CUmodule m = nullptr;
  CUresult r = cuModuleLoadData(&m, buf.data());
  if(r != CUDA_SUCCESS){
    const char* s = nullptr; cuGetErrorString(r, &s);
    std::printf("load fail: %d %s\n", (int)r, s ? s : "?");
    return 5;
  }
  CUfunction k = nullptr;
  r = cuModuleGetFunction(&k, m, "K");
  if(r != CUDA_SUCCESS){
    const char* s = nullptr; cuGetErrorString(r, &s);
    std::printf("get func fail: %d %s\n", (int)r, s ? s : "?");
    return 6;
  }
  puts("ok");
  cuModuleUnload(m);
  return 0;
}
EOF

c++ \
  -I"$MACA_PATH/tools/cu-bridge/include" \
  -I"$MACA_PATH/include" \
  -I"$MACA_PATH/include/mcr" \
  "$WORK_DIR/load_probe.cpp" \
  -L"$CUDA_PATH/lib64" \
  -L"$CUDA_PATH/lib64/stubs" \
  -L"$MACA_PATH/lib" \
  -lcudart -lcuda -lnvrtc -lruntime_cu \
  -o "$WORK_DIR/load_probe"

"$WORK_DIR/load_probe" "$WORK_DIR/probe_linked.mcfb"

echo "mxcc maca-link probe succeeded: $WORK_DIR/probe_linked.mcfb"
