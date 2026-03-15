# HIPRT 当前主线相对官方基线的差异分类

## 1. 基线说明

- 当前审计对象：`/data/HIPRT` 的 `main`
- 当前可确认的官方共同基线：`upstream_main_dev = 16d7899`
- 当前本地主线额外提交主要集中在：
  - `4c86eed` 到 `75307b6`

说明：

- 这里不是逐行重新评审所有差异，而是按“改动本质影响”分类。
- 目标是回答：
  1. 哪些修改是当前 MACA 主路径真正必要的
  2. 哪些修改只是当前私有环境适配
  3. 哪些修改后续可以继续收敛

## 2. 总体结论

当前 `main` 相对官方共同基线的差异，本质上分成 4 组：

1. **CUDA-only / 仓库瘦身**
   - 清理 HIP/ROCm、旧 Orochi 子树、测试和工具残留

2. **MACA/cu-bridge 运行时与构建兼容修复**
   - runtime JIT 兼容
   - scene transform 修复
   - 构建脚本与 cache 控制

3. **bitcode / precompile / bake_kernel 恢复**
   - 离线产物生成
   - 预编译 fatbin 消费
   - 相关测试和脚本

4. **文档与测试补强**
   - 记录当前支持边界
   - 增加 MACA 主路径的红绿灯

## 3. 建议保留的必要修改

这些修改已经被编译、单测或实际运行验证过，当前不建议回退。

### 3.1 runtime/JIT 兼容修复

关键文件：

- `CMakeLists.txt`
- `hiprt/impl/Compiler.cpp`
- `hiprt/impl/Compiler.h`
- `hiprt/impl/Context.cpp`
- `hiprt/impl/Context.h`
- `hiprt/impl/hiprt.cpp`

必要原因：

- `HIPRT_CU_BRIDGE_RUNTIME_JIT_WORKAROUND`
  - 解决 cu-bridge 下 `nvrtcAddNameExpression` 兼容问题
- runtime JIT 中 device helper / filter / custom func 的符号可见性问题
- `hiprtBuildTraceKernelsFromBitcode(...)` 从 stub 恢复为真实工作实现
- `nvrtcGetPTX` 不稳定时，增加 `nvrtcGetCUBIN` / 外部 cubin fallback

判断：

- 这些是“当前功能是否能工作”的主路径修复，不是风格调整。

### 3.2 scene transform 修复

关键文件：

- `hiprt/impl/BvhNode.h`
- `hiprt/impl/BvhBuilderKernels.h`
- `hiprt/impl/hiprt_device_impl.h`

必要原因：

- scene build/update 路径会覆盖 `transform header`
- 会直接影响 scene traversal / instancing / motion blur 结果

判断：

- 已做过真实实验验证，属于必要修复。

### 3.3 bitcode / precompile / bake_kernel 工作流恢复

关键文件：

- `CMakeLists.txt`
- `hiprt/hiprt_common.h`
- `scripts/bitcodes/compile.py`
- `scripts/bitcodes/precompile_bitcode.py`
- `scripts/build.sh`
- `scripts/build_maca_cucc.sh`
- `scripts/unittest_maca_cucc_precompiled.sh`
- `tools/bakeKernel.sh`
- `tools/bakeKernel.bat`
- `test/bitcodes/custom_func_table.cpp`
- `test/bitcodes/unit_test.cpp`
- `test/hiprtTest.cpp`
- `test/hiprtTest.h`
- `test/main.cpp`

必要原因：

- 当前 MACA 主路径首次运行成本高
- precompiled workflow 是当前已验证可行的降时延方案
- `bake_kernel`、`precompiled fatbin`、`trace kernel launch` 都已形成闭环

判断：

- 这是当前主线新增能力，不建议回退。

### 3.4 构建与验证脚本

关键文件：

- `scripts/build_maca_cucc.sh`
- `scripts/unittest_maca_cucc.sh`
- `scripts/unittest_maca_cucc_precompiled.sh`

必要原因：

- 当前 `cmake_maca + make_maca` 是已验证稳定路径
- 单独把 bitcode/precompiled 验证固定成脚本，有助于避免回归漂移

判断：

- 保留。

## 4. 当前有效，但建议后续继续收敛的修改

### 4.1 文档中的环境路径硬编码

关键文件：

- `README.md`
- `docs/bitcode-status-zh.md`
- `docs/main-maca-fix-notes-zh.md`

问题：

- 当前内容对本地环境很友好
- 但对外部环境复用性弱

建议：

- 保留结论
- 把路径和命令改成更参数化的写法

### 4.2 Compiler.cpp 里的外部编译器 fallback

关键文件：

- `hiprt/impl/Compiler.cpp`

问题：

- 当前 fallback 是必要兜底
- 但实现更像“工程侧 workaround”

建议：

- 短期保留
- 长期应优先推动：
  - cu-bridge runtime JIT 进一步稳定
  - 或离线编译脚本更标准化

### 4.3 测试中的 cu-bridge 条件跳过

关键文件：

- `test/main.cpp`

问题：

- `BuildTraceKernelFromBitcode*` 在 cu-bridge 下被跳过
- 当前是合理的

建议：

- 保留
- 但后续一旦 native CUDA runtime bitcode 路径补验通过，应把“只在 cu-bridge 跳过”写得更明确

## 5. 私有环境适配，不建议原样上游

这些不是错误修改，但更适合留在私有分支或做参数化，而不是原样上游。

### 5.1 本地路径和工具假设

- `/data/HIPRT`
- `/opt/maca`
- `/root/cu-bridge`
- `cmake_maca`
- `make_maca`

### 5.2 仅为当前环境服务的文档说明

- 当前机器无原生 `nvcc`
- 当前主要验证路径是 precompiled
- 当前 cache 策略如何配置

这类内容对私有分支有价值，但不适合作为官方仓库默认结论。

## 6. 目前没有发现应立即回退的核心逻辑改动

当前结论不是“所有改动都完美”，而是：

- **没有发现需要立刻回退的主路径逻辑修改**

原因：

- 当前关键改动都至少被以下之一覆盖：
  - 编译通过
  - 单元测试通过
  - 预编译产物生成与执行通过

## 7. 后续精简建议

如果要为后续和官方同步做更清晰的 patch 组织，建议重排成 4 组：

1. `build-system`
   - `CMakeLists.txt`
   - `scripts/*`

2. `runtime-adapter`
   - `Compiler.*`
   - `Context.*`
   - `hiprt.cpp`

3. `scene-fixes`
   - `BvhNode.h`
   - `BvhBuilderKernels.h`
   - `hiprt_device_impl.h`

4. `bitcode-and-tests`
   - `bitcodes/*`
   - `test/bitcodes/*`
   - `test/*`
   - `tools/bakeKernel*`

## 8. 一句话结论

- **当前 HIPRT 主线相对官方基线的关键 MACA 改动总体是必要的；核心保留项是 runtime/JIT 兼容、scene transform 修复、bitcode/precompile/bake_kernel 恢复，当前没有明显应立即回退的关键逻辑改动。**
