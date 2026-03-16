# 用 mxcc 增强 HIPRT 离线编译链的方案

## 1. 目标

不是把整个 HIPRT 主构建入口从 `cucc/cmake_maca` 改写成纯 `mxcc`，而是：

- **在离线编译、预编译、外部 cubin fallback 这几条链上，增强 `mxcc` 的参与度和稳定性**

原因：

- 当前主构建入口仍然需要 `cucc/cmake_maca` 解决 CUDA 工程兼容问题
- 但离线产物生成和 wrapper fallback 更适合走更底层、更直接的编译器能力

## 2. 当前现状

当前已经工作的链：

1. `cmake_maca + make_maca`
2. `scripts/bitcodes/compile.py`
3. `scripts/bitcodes/precompile_bitcode.py`
4. `Compiler.cpp` 中的外部 cubin fallback

当前问题：

- `cu-bridge` 下 runtime `nvrtc --device-c` 对用户 trace kernel 源码即时产物仍不稳定
- 因此更多场景依赖：
  - precompiled fatbin
  - 外部 cubin fallback

## 3. 为什么不能直接用 mxcc 替代 cucc

### 3.1 `cucc/cmake_maca` 是工程入口兼容层

它解决的是：

- `.cu` 源码入口
- CUDA 语言模式
- 现有 CMake 对 CUDA compiler 的预期
- CUDA 头与 runtime/driver API 的桥接

### 3.2 `mxcc` 更像底层编译后端

它擅长的是：

- 单个或少量 translation unit 的静态编译
- 已知 include / define / arch 参数下的离线产物生成

结论：

- **不能直接拿 `mxcc` 替掉 `cucc` 作为主工程入口**
- **但可以让 `mxcc` 成为离线链的稳定后端**

## 4. 建议优先做的 3 个方向

### 4.1 方向 A：增强 `scripts/bitcodes/compile.py`

目标：

- 当前脚本主要通过 `CMAKE_CUDA_COMPILER`
- 实际底层仍可能走 `cucc -> mxcc`

建议增强：

- 新增显式开关，例如：
  - `--toolchain=cucc`
  - `--toolchain=mxcc`
- 当选 `mxcc` 时：
  - 直接生成 `hiprt*_nv_lib.fatbin`
  - 直接生成 `hiprt*_nv.fatbin`

价值：

- 降低对 `nvcc wrapper` 路径的黑盒依赖
- 更容易定位离线编译问题

### 4.2 方向 B：增强 `scripts/bitcodes/precompile_bitcode.py`

目标：

- 让 precompiled trace-kernel fatbin 生成不只依赖 `cucc/nvcc` 风格入口

建议增强：

- 增加 `mxcc` 后端分支
- 仍然保留当前 `cucc` 默认路径

价值：

- 当前 precompile 是 MACA 已验证主路径
- 这是最值得先做稳定化的地方

### 4.3 方向 C：增强 `Compiler.cpp` 外部 cubin fallback

目标：

- 当前 `compileSourceToCubin(...)` 已经是 runtime 失败后的兜底

建议增强：

- 优先级改为：
  1. `HIPRT_CUDA_COMPILER`
  2. `CUDACXX`
  3. `mxcc`
  4. `cucc`
  5. `nvcc`

或至少提供：

- `HIPRT_EXTERNAL_DEVICE_COMPILER=mxcc`

价值：

- 可以把当前很多 tutorial/SDK 场景里的“外部 cubin 兜底”显式固定到 `mxcc`
- 减少环境漂移

## 5. 不建议优先做的方向

### 5.1 不建议直接把主 CMake 从 cucc 改成 mxcc

原因：

- 影响面太大
- 会重新打开：
  - CUDA 语言识别
  - CMake compiler detection
  - nvrtc/runtime include 兼容

### 5.2 不建议先改 unit test 主路径

原因：

- 当前主测试链已经稳定
- 更应该先在：
  - bitcode/precompile 脚本
  - compiler fallback
 这两层验证 `mxcc`

## 6. 建议实施顺序

### 阶段 1

- 只改 `scripts/bitcodes/compile.py`
- 增加可选 `mxcc` 后端
- 目标：
  - 能生成 `hiprt*_nv_lib.fatbin`
  - 能生成 `hiprt*_nv.fatbin`

### 阶段 2

- 改 `scripts/bitcodes/precompile_bitcode.py`
- 增加可选 `mxcc` 后端
- 目标：
  - 能生成 `hiprt*_nv_precompiled_bitcode.fatbin`

### 阶段 3

- 改 `Compiler.cpp`
- 让 runtime fallback 优先支持 `mxcc`

当前已完成到：

- `findCudaCompiler()` 已支持 `HIPRT_EXTERNAL_DEVICE_COMPILER`
- 可显式指定：
  - `mxcc`
  - `cucc`
  - `nvcc`

也就是说，runtime fallback 现在已经具备“优先走 mxcc”的入口条件。

### 阶段 4

- 在 `test/` 或 SDK 场景上补验证
- 验证：
  - 外部 cubin fallback
  - precompiled fatbin
  - custom func table 场景

## 7. 当前最现实的判断

如果现在只问一句：

- **离线编译剩余问题是否适合优先用 `mxcc` 解决？**

答案是：

- **是，适合**

但更准确的说法是：

- **适合把 `mxcc` 用在离线产物和 fallback 路径上**
- **不适合现在就把整个 HIPRT 工程入口层从 `cucc/cmake_maca` 全量替换掉**

## 8. 一句话结论

- **`mxcc` 不是当前 HIPRT 主构建入口的替代品，但非常适合作为 bitcode / precompile / cubin fallback 的增强后端；如果后续要继续优化离线编译稳定性，优先应该沿这条线推进。**
