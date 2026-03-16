# main 分支 MACA 修复结论（2026-03-15）

## 本轮结论

本轮在 `main` 上没有继续整体回放 `maca_dev`，而是只按已经在 `maca_dev` 上通过实验确认的三类真实问题做最小修复：

1. cu-bridge/MACA runtime JIT 的 `nvrtcAddNameExpression` 兼容问题
2. runtime JIT 中 device helper / filter 的符号可见性问题
3. scene instance 在 build/update 路径覆盖 `transform header` 的问题

当前判断是：

- 这三类都属于真实问题
- 不属于临时绕行
- 回放到 `main` 后也都直接影响当前主线测试结果

## main 上采用的最小修复

### 1. runtime JIT 兼容修复

- 在 `CMakeLists.txt` 中，通过 `HIPRT_USING_CU_BRIDGE` 向主库和 `unittest` 注入 `HIPRT_CU_BRIDGE_RUNTIME_JIT_WORKAROUND`
- 在 `hiprt/impl/Compiler.cpp` 中：
  - cu-bridge 路径下跳过 `nvrtcAddNameExpression`
  - 直接按 `extern "C"` 入口名取函数

### 2. device helper / filter 可见性修复

- 在 `hiprt/impl/hiprt_device_impl.h` 中，把 `hiprtPointWorldToObject` / `hiprtPointObjectToWorld` / `hiprtVectorWorldToObject` / `hiprtVectorObjectToWorld` 以及对应的多层 instance 版本改成 `HIPRT_INLINE`
- 在 `test/kernels/HiprtTestKernel.h` 中，把：
  - `duplicityFilter`
  - `cutoutFilter`
  - `intersectCircle`
  - `intersectSphere`
  改成 `HIPRT_INLINE`

### 3. scene instance transform header 保留

- 在 `hiprt/impl/BvhNode.h` 中：
  - scene instance 初始化统一保留 `m_transform`
  - 不再把单帧 instance 直接写成 static matrix 语义
- 在 `hiprt/impl/BvhBuilderKernels.h` 的 update-leaf 路径中：
  - 同样保留 `transform header`
  - 不再把单帧 instance 写回 `m_identity + m_matrix`

## 当前本地验证结果

在 `main` 上使用：

- `cmake_maca`
- `make_maca`
- `HIPRT_ENABLE_RUNTIME_KERNEL_CACHE=OFF`
- `HIPRT_DISABLE_RUNTIME_KERNEL_CACHE=1`

做本地串行回归，当前可见的非性能测试结果为：

- `39 / 39 passed`

包含已验证通过的关键路径：

- 基础 JIT：
  - `hiprtTest.MinimumCornellBox`
  - `hiprtTest.Compaction`
  - `hiprtTest.PairTriangles`
  - `hiprtTest.TraceKernel`
- scene / transform / update 用户可见路径：
  - `ObjTestCases.TranslateCornellBox`
  - `ObjTestCases.ScaleCornellBox`
  - `ObjTestCases.RotateCornellBox`
  - `ObjTestCases.BvhUpdateCornellBox`
- 当前 `main` 可见的完整非性能测试集：
  - `39 / 39` 通过

## 备注

这份结论文件只记录 `main` 当前已经完成的最小修复与验证结果。

如果后续继续把 `maca_dev` 的其余诊断测试或文档回放到 `main`，应继续遵守：

- 只引入已经通过实验确认必要的修复
- 不把尚未证实必要的历史绕行策略机械回放
