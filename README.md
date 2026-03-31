# HIPRT

## 项目概览

当前仓库保留 `HIPRT` 项目名称以及 `hiprt*` 对外 API 命名，但主实现已经收敛到 **CUDA-only** 后端。

- 保留：
  - `hiprt.h`
  - `hiprtew.h`
  - `hiprtCreateContext`
  - `hiprtBuildTraceKernels`
  - `hiprtBuildTraceKernelsFromLinkedBundle`
- 已移出当前主构建链：
  - AMD HIP runtime 主路径
  - ROCm toolchain 依赖
  - `hipcc`
  - 历史 HIP loader 主路径
- 当前迁移阶段要求：
  - 先确保纯 CUDA 基线可编译、可运行、可复现
  - 再进入后续 `MACA + cu-bridge` 适配与扩展

`hiprt/hiprtew.h` 仍保留为兼容入口头，但当前实现已不再承担运行时动态加载器角色。

## 快速开始

### 1. 拉取代码

```bash
git clone https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT.git
cd HIPRT
git submodule update --init --recursive
```

说明：

- 当前默认优先使用 `CMake + Ninja`。

### 2. 推荐构建方式

优先直接使用仓库脚本：

```bash
./scripts/build.sh
```

脚本会在环境可用时自动接入：

- `ccache`
- `mold`
- `Ninja`

如果需要手动指定常见选项，可直接设置环境变量：

```bash
CUDA_ARCHITECTURES=89 BUILD_TYPE=Release ./scripts/build.sh
```

等价的原生 CMake 调用为：

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

### 3. 推荐功能回归

根据当前仓库约定，功能正确性验证不要依赖历史 JIT cache 结果。回归前建议先清理测试过程中生成的 `scripts/cache/`：

```bash
cd scripts
rm -rf cache
./unittest.sh
```

默认先跑非性能功能测试；性能测试单独执行。

## 主要 API 中文指南

`README` 首页只保留入口信息。主要 public API 的主机侧用法、对象生命周期和 trace-kernel 三条构建路径，已经整理到单独中文指南：

- [主要 API 使用指南（中文）](docs/api-guide-zh.md)

这份指南重点覆盖：

- `hiprtCreateContext` / `hiprtDestroyContext`
- `hiprtCreateGeometry` / `hiprtBuildGeometry`
- `hiprtCreateScene` / `hiprtBuildScene`
- `hiprtCreateFuncTable` / `hiprtSetFuncTable`
- `hiprtBuildTraceKernels`
- `hiprtBuildTraceKernelsFromBitcode`
- `hiprtBuildTraceKernelsFromLinkedBundle`

## 当前推荐使用路径

- 纯 CUDA 基线：
  - 优先保证 `./scripts/build.sh` + `cd scripts && ./unittest.sh` 可稳定通过
  - source-based trace kernel 走 `hiprtBuildTraceKernels(...)`
  - 已有可重定位 PTX/CUBIN 时可走 `hiprtBuildTraceKernelsFromBitcode(...)`
- `MACA + cu-bridge`：
  - 当前更适合走 precompiled / linked-bundle 路径
  - 显式 public API 为 `hiprtBuildTraceKernelsFromLinkedBundle(...)`
  - 详细限制与状态见下方文档

如果在验证 trace kernel / runtime JIT 行为，优先避免复用旧缓存；必要时请显式关闭 cache，或者切换到新的临时 cache 目录。

## 文档导航

- [主要 API 使用指南（中文）](docs/api-guide-zh.md)
- [CUDA-only 编译与改造说明](docs/cuda-only-build-zh.md)
- [bitcode / precompile / bake_kernel 当前状态](docs/bitcode-status-zh.md)
- [当前 main 分支 MACA 修复说明](docs/main-maca-fix-notes-zh.md)
- [main 与 upstream 差异分类](docs/main-vs-upstream-classification-zh.md)
- [mxcc offline 进展](docs/mxcc-offline-progress-zh.md)
- [mxcc offline 计划](docs/mxcc-offline-plan-zh.md)

## 示例效果

以下图片来自当前仓库联动 `HIPRTSDK` 跑通后的真实结果：

### Geometry Intersection

![Geometry Intersection](docs/images/sdk_showcase/01_geom_intersection.png)

### Custom BVH Import

![Custom BVH Import](docs/images/sdk_showcase/07_custom_bvh_import.png)

### Primary Ray

![Primary Ray Normal](docs/images/sdk_showcase/19_primary_ray_normal.png)

## 构建补充说明

- `CUDAToolkit` 是当前主构建前提。
- `CMAKE_CUDA_ARCHITECTURES` 可通过 cache 或脚本环境变量指定。
- 单测默认参与构建；如需关闭可使用 `BUILD_TESTS=OFF ./scripts/build.sh` 或 `-DNO_UNITTEST=ON`。
- 构建产物默认输出到 `dist/bin/<Config>/`。
- 可选 bitcode / precompile 开关：
  - `HIPRT_ENABLE_BAKE_KERNEL=ON`
  - `HIPRT_ENABLE_BITCODE=ON`
  - `HIPRT_ENABLE_PRECOMPILED_TRACE_KERNEL=ON`

## 单元测试

当前测试主要分为三类：

1. `HiprtTests`：基础功能覆盖
2. `ObjTestCases`：网格与场景相关功能
3. `PerformanceTestCases`：性能相关测试

常用入口：

- `cd scripts && ./unittest.sh`
- `cd scripts && ./unittest_perf.sh`

## 开发约定

### Coding Guidelines

- Resolve compiler warnings.
- Use lower camel case for variable names (e.g., `nodeCount`) and upper camel case for constants (e.g., `LogSize`).
- Separate functions by one line.
- Use prefix `m_` for non-static member variables.
- Do not use static local variables.
- Do not use `void` for functions without arguments (leave it blank).
- Do not use blocks without any reason.
- Use references instead of pointers if possible.
- Use bit-fields instead of explicit bit masking if possible.
- Use `nullptr` instead of `NULL` or zero.
- Use `using` instead of `typedef`.
- Use C++-style casts (e.g., `static_cast`) instead of C-style cast.
- Add `const` for references and pointers if they are not being changed.
- Add `constexpr` for variables and functions if they can be constant in compile time (do not use `#define` if possible).
- Use `if constexpr` instead of `#ifdef` if possible.
- Throw `std::runtime_error` with an appropriate message in case of failure in the core and catch it in `hiprt.cpp`.

#### String

- Use `std::string` instead of C strings (i.e., `char*`) and avoid C string functions as much as possible.
- Use `std::cout` and `std::cerr` instead of `printf`.
- Do not assign `char8_t` (or `std::u8string`) to `char` (or `std::string`). They will not be compatible in C++20.

#### File

- Use `std::ifstream` and `std::ofstream` instead of `FILE`.
- Use `std::filesystem::path` for files and paths instead of `std::string`.

#### Class

- Use the in-class initializer instead of the default constructor.
- Use the keyword `override` instead of `virtual` (or nothing) when overriding a virtual function from the base class.
- Use `std::optional` instead of pointers for optional parameters.
- A base class destructor should be either public and virtual, or protected and non-virtual.
- Implement the customized {copy/move} {constructor/assignment operator} if an user-defined destructor of a class is needed, or remove them using `= delete`.

### Versioning

- When we update the master branch, we need to update the version number of hiprt in `version.txt`.
- If there is a change in the API, you need to update minor version.
- If the major and minor versions matches, the binaries are compatible.
- Each commit in the master should have a unique patch version.
