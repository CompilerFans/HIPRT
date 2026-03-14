# HIPRT 当前编译与改造说明

## 目标边界

当前仓库保留 `HIPRT` 项目名以及 `hiprt*` 公开 API 命名，但底层实现已经收敛为 **CUDA-only**。

这次改造的目标不是重命名项目，而是去除以下依赖：

- AMD GPU HIP runtime
- ROCm toolchain
- `hipcc`
- 运行时 HIP 动态加载路径
- 旧的 bitcode / 预编译构建链
- 历史兼容层中的 HIP 运行时绑定逻辑

保留的内容：

- `hiprt.h`
- `hiprtew.h`
- `hiprtCreateContext` 等公开 API
- 现有测试名称和大部分外部调用方式

## 当前编译方式

项目现在仅支持 **CMake + CUDA Toolkit**。

### Linux / Windows 通用

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

常用可选项：

- 指定架构：`-DCMAKE_CUDA_ARCHITECTURES=89`
- 不编译单测：`-DNO_UNITTEST=ON`

构建产物默认输出到：

- `dist/bin/Release/`
- `dist/bin/Debug/`

## 当前运行时方式

### 主库

主库直接链接并调用：

- `CUDA::cudart`
- `CUDA::cuda_driver`
- `CUDA::nvrtc`

### hiprtew 兼容头

`hiprt/hiprtew.h` 仍然保留，但它不再执行 `dlopen/dlsym` 或 `LoadLibrary/GetProcAddress`。

现在的行为是：

- `hiprtewInit()` 仅返回成功状态
- 后续调用直接落到已链接的 `hiprt` 导出 API

也就是说，`hiprtew.h` 现在是一个 **兼容入口头**，不是独立的动态绑定层。

## 已移除路径

以下路径已经不再参与当前仓库的编译链：

- `premake5.lua`
- `tools/functions.lua`
- `scripts/bitcodes/*`
- `test/bitcodes/*`
- 历史 `HIPRTEW` 测试目标
- HIP runtime loader 路径表
- `contrib/Orochi/contrib/hipew`
- `contrib/Orochi/Test`
- `contrib/Orochi/UnitTest`
- `contrib/Orochi/scripts`
- `contrib/Orochi/tools`

`contrib/Orochi` 当前仅保留对主仓库仍有价值的核心源码，例如：

- `contrib/Orochi/Orochi`
- `contrib/Orochi/ParallelPrimitives`

如果后续新增代码，请不要再恢复这些旧路径，也不要重新引入：

- `HIP_PATH`
- `hipcc`
- ROCm 版本判断
- 运行时 HIP 动态库查找

## 测试建议

基本回归可先跑：

```bash
cd scripts
./unittest.sh
```

如果只做快速验证，推荐直接筛选：

```bash
../dist/bin/Release/unittest64 --width=512 --height=512 --referencePath=../test/references/ --gtest_filter=hiprtTest.CudaEnabled:hiprtTest.MinimumCornellBox:ObjTestCases.PrimaryRayCornellBox
```

## 后续改动建议

后续如果继续演进本仓库，建议遵守两条原则：

1. 保持 `HIPRT/hiprt*` 公开命名稳定，避免无必要 ABI 破坏。
2. 所有新增实现都默认按 CUDA-only 路径设计，不再为 AMD HIP runtime 保留分支。
