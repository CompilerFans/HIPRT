# bitcode / precompile / bake_kernel 当前状态

## 1. 当前已经恢复的能力

当前 `main` 上，CUDA/cu-bridge 路径已经重新接回 3 条链：

1. `bake_kernel`
   - 通过 `-DHIPRT_ENABLE_BAKE_KERNEL=ON`
   - 生成：
     - `hiprt/cache/Kernels.h`
     - `hiprt/cache/KernelArgs.h`

2. HIPRT 预编译 fatbin
   - 通过 `-DHIPRT_ENABLE_BITCODE=ON`
   - 生成：
     - `dist/bin/<Config>/hiprt<ver>_nv_lib.fatbin`
     - `dist/bin/<Config>/hiprt<ver>_nv.fatbin`
     - 同步拷贝到 `hiprt/bitcodes/`

3. 预编译 trace-kernel fatbin
   - 通过 `-DHIPRT_ENABLE_PRECOMPILED_TRACE_KERNEL=ON`
   - 生成：
     - `dist/bin/<Config>/hiprt<ver>_nv_precompiled_bitcode.fatbin`
     - 同步拷贝到 `hiprt/bitcodes/`

此外：

- `hiprtBuildTraceKernelsFromBitcode(...)` 不再是桩函数
- 当前实现已经支持把用户侧 PTX/CUBIN 与 `hiprt*_nv_lib.fatbin` 做链接

## 2. 本次恢复的实现方式

### 2.1 bake_kernel

- 继续沿用 `tools/bakeKernel.sh` / `tools/bakeKernel.bat`
- 但去掉了历史加密输出，改为直接生成当前主线可消费的明文字符串头
- 同时恢复了：
  - `GET_ARGS`
  - `GET_INC`
  - `GET_ARG_LIST`

这样 `HIPRT_LOAD_FROM_STRING` 路径重新可用，不需要再依赖历史 HIP 工具链。

### 2.2 bitcode / precompile 脚本

历史脚本原本偏向：

- HIP / `hipcc`
- AMD `hipfb`
- 旧 Orochi 并行原语脚本

当前已经改成：

- 面向当前 CUDA / cu-bridge 的 `nvcc` 兼容命令行
- 由 `CMakeLists.txt` 通过：
  - `HIPRT_ENABLE_BITCODE`
  - `HIPRT_ENABLE_PRECOMPILED_TRACE_KERNEL`
  统一接入
- 默认使用当前配置下的 `CMAKE_CUDA_COMPILER`
  - 原生 CUDA 下对应 `nvcc`
  - MACA 下对应 `cucc` / `cmake_maca`

### 2.3 runtime bitcode API

`hiprtBuildTraceKernelsFromBitcode(...)` 当前实现：

- 接收用户侧 PTX/CUBIN 二进制
- 读取预编译好的 `hiprt*_nv_lib.fatbin`
- 额外生成 custom function table 的 device 二进制
- 通过 CUDA driver linker 做链接

也就是说：

- 这条 API 已经重新“有功能”
- 不再是单纯 `hiprtErrorNotImplemented`

## 3. 当前已验证结果

在本地 `MACA + cu-bridge` 环境下，已经实际验证：

1. `HIPRT_ENABLE_BAKE_KERNEL=ON`
   - `hiprt/cache/Kernels.h`
   - `hiprt/cache/KernelArgs.h`
   已成功生成

2. `HIPRT_ENABLE_BITCODE=ON`
   成功生成：
   - `hiprt03001_nv_lib.fatbin`
   - `hiprt03001_nv.fatbin`

3. `HIPRT_ENABLE_PRECOMPILED_TRACE_KERNEL=ON`
   成功生成：
   - `hiprt03001_nv_precompiled_bitcode.fatbin`

4. 新增的 bitcode 相关 UT
   - `BuildTraceKernelFromBitcode`
   - `BuildTraceKernelFromBitcodeWithCustomFuncTable`
   在 cu-bridge 环境下不会再挂红

5. 新增的 precompiled 消费侧 UT
   - `LoadPrecompiledTraceKernel`
   - `LoadPrecompiledTraceKernelWithCustomFuncTable`
   已在 `MACA + cu-bridge` 下实际通过
   - 这两条测试直接从生成好的 `hiprt*_nv_precompiled_bitcode.fatbin` 中加载 kernel
   - 用来守护“预编译产物可被当前运行时消费”这条主路径

## 4. 当前还存在的限制

当前最重要的限制只有一条：

- **在 `MACA + cu-bridge` 下，runtime `nvrtc --device-c` 直接为“用户 trace-kernel 源码”产出可消费二进制这一步仍不稳定**

这意味着：

- `hiprtBuildTraceKernelsFromBitcode(...)` 的 API 本身已经恢复
- 但在 cu-bridge 环境里，如果上游想“纯 runtime 临时编一段用户源码，再立即喂给这个 API”，这条链目前还不稳

所以当前建议是：

- 对原生 CUDA：
  - 可以继续把 runtime bitcode API 当作可用路径
- 对 `MACA + cu-bridge`：
  - **优先走 precompile 工作流**
  - 即：
    1. 先用 `HIPRT_ENABLE_BITCODE=ON`
    2. 再用 `HIPRT_ENABLE_PRECOMPILED_TRACE_KERNEL=ON`
    3. 使用生成好的 fatbin 做验证与部署

这也是为什么当前新增的两个 runtime-bitcode UT 在 cu-bridge 环境下会主动跳过：

- 不是因为 bitcode 工作流整体不可用
- 而是为了避免把“cu-bridge 下 runtime 用户源码产物不稳定”误判成 HIPRT 主链恢复失败

与之对应，当前真正作为 cu-bridge 主验证路径的是：

- 先生成 `hiprt*_nv_precompiled_bitcode.fatbin`
- 再通过 `LoadPrecompiledTraceKernel*` UT 直接加载验证

## 5. 当前推荐用法

### 5.1 原生 CUDA

```bash
./scripts/build.sh
```

如果要生成 bitcode / precompiled 产物：

```bash
HIPRT_ENABLE_BAKE_KERNEL=ON \
HIPRT_ENABLE_BITCODE=ON \
HIPRT_ENABLE_PRECOMPILED_TRACE_KERNEL=ON \
./scripts/build.sh
```

### 5.2 MACA / cu-bridge

```bash
HIPRT_ENABLE_BAKE_KERNEL=ON \
HIPRT_ENABLE_BITCODE=ON \
HIPRT_ENABLE_PRECOMPILED_TRACE_KERNEL=ON \
./scripts/build_maca_cucc.sh
```

当前在 MACA 上，推荐把：

- `hiprt*_nv_lib.fatbin`
- `hiprt*_nv.fatbin`
- `hiprt*_nv_precompiled_bitcode.fatbin`

视为主验证产物。

## 6. 结论

当前结论可以明确写成一句话：

- **bitcode / precompile / bake_kernel 工作流已经在当前主线重新恢复到“可构建、可产生产物、可作为 MACA 首次运行降时延方案”的状态；但 `MACA + cu-bridge` 下 runtime 用户源码即时 bitcode 生成仍有限制，因此该环境当前以 precompile 路径为主。**
