# `main` 相对 `upstream_main_dev` 的实现逻辑差异复盘

## 范围

- 对比基线：`upstream_main_dev`
- 当前分支：`main`
- 关注点：**实现逻辑差异**
- 不把纯文档、纯测试补充、纯脚本入口视为“实现逻辑差异”

## 当前结论

当前 `main` 相对 `upstream_main_dev` 的核心行为差异，可以分成 4 类：

1. 纯 CUDA 收敛
2. cu-bridge / MACA 构建与 runtime JIT 适配
3. wave64 / lane-mask 语义修复
4. scene instance transform header 保留

到目前为止，**没有发现可以直接安全回退到官方实现的核心行为修复**。

也就是说，当前这批真正影响运行结果的主线差异，暂时都属于“必要差异”，而不是“可随手改回官方”的历史残留。

## 1. 纯 CUDA 收敛

涉及：

- `CMakeLists.txt`
- `hiprt/hiprtew.h.in`
- `hiprt/hiprt.h.in`
- 一部分 `hiprt/impl/*`

当前判断：

- **必要**

原因：

- 当前项目主线目标本来就是纯 CUDA 收敛
- 这部分差异不是 MACA 专属 patch，而是整个仓库分叉方向本身
- 直接回退到 `upstream_main_dev` 的 HIP / 多后端逻辑，不符合当前主线目标

## 2. cu-bridge / MACA runtime JIT 适配

涉及：

- `CMakeLists.txt`
- `hiprt/impl/Compiler.cpp`
- `hiprt/impl/hiprt_device_impl.h`
- `test/kernels/HiprtTestKernel.h`

关键差异：

1. cu-bridge 路径下注入 `HIPRT_CU_BRIDGE_RUNTIME_JIT_WORKAROUND`
2. runtime JIT 下跳过 `nvrtcAddNameExpression`
3. runtime compile 用到的 device helper / filter 改成 `HIPRT_INLINE`

当前判断：

- **必要**

理由：

- 这是已经在 `maca_dev` 和 `main` 上都通过真实实验确认的根因修复
- 未修复前：
  - `CornellBoxKernel` 会在 `__mcrtc_*` 包装中不可见
  - `duplicityFilter(...)`、`hiprtPointWorldToObject(...)` 会在 runtime JIT 链接阶段未定义
- 修复后：
  - `MinimumCornellBox`
  - `Compaction`
  - `PairTriangles`
  - `TraceKernel`
  都恢复通过

结论：

- 这一组不能回退
- 它们已经不是“平台特殊优化”，而是当前 cu-bridge/MACA 下的必要兼容实现

## 3. wave64 / lane-mask 语义修复

涉及：

- `hiprt/hiprt_common.h`
- `hiprt/impl/BvhBuilderUtil.h`
- `hiprt/impl/BvhBuilderKernels.h`
- `hiprt/impl/PlocBuilderKernels.h`

关键差异：

- `LaneMask`
- 64-bit ballot / popcount / first-set / lower-count
- `WarpSize` 对应的 lane-mask 语义调整

当前判断：

- **必要**

原因：

- 这是 MACA wave64 语义下的真实兼容修复
- 之前在 `maca_dev` 的基础诊断和回归中已经证明：
  - `MinimumCornellBox`
  - `Compaction`
  - `PairTriangles`
  - `PlocFallback`
  等基础路径都直接依赖这组修复

结论：

- 当前没有证据支持把这部分改回 `upstream_main_dev`
- 相反，它属于跨文件但高度必要的公共语义层修复

## 4. scene instance transform header 保留

涉及：

- `hiprt/impl/BvhNode.h`
- `hiprt/impl/BvhBuilderKernels.h`

关键差异：

- 不再把单帧 instance 自动写回 `static + matrix`
- 统一保留 `m_transform`
- update-leaf 路径也统一保留 `transform header`

当前判断：

- **必要**

原因：

- 这组修复已经在 `maca_dev` 上做过针对性 A/B 验证
- 在 `main` 上回放后，以下路径恢复通过：
  - `SceneSingletonSrtNodeUsesTransformHeader`
  - `SceneInternalTransformRaySrt`
  - `SceneTransformDebugSrt`
  - `SceneInterpolatedFrameDebugSrt`
  - `SceneInverseMatrixDebugSrt`
  - `SceneClosestHitSingletonSrt`
  - `SceneClosestHitSingletonSrtRecreate`
  - `SceneTraceKernelSingletonSrt`
  - `SceneIntersectionSingleton`
  - `SceneIntersection`
  - `SceneIntersectionMlas`
  - `Shear`
  - `TranslateCornellBox`
  - `ScaleCornellBox`
  - `RotateCornellBox`
  - `BvhUpdateCornellBox`

结论：

- 这组修复同样不能回退
- 它修正的是 scene instance 在 build / update 路径上的真实错误语义

## 当前没有回退的原因

对照 `upstream_main_dev` 后，当前还没做“尽量改回官方实现”的代码回退，原因不是没做，而是：

- 已经分析到的行为差异里，真正影响主线功能结果的部分都已经被实验证实为必要
- 当前没有哪一项核心逻辑差异可以在不丢失测试正确性的前提下直接回退

## 后续仍可继续收敛的方向

虽然核心行为修复当前都必要，但以下部分仍可继续做“更接近官方主线”的收敛：

1. `CMakeLists.txt` 中与构建体验相关但不影响功能正确性的分支
2. MACA 专用脚本的默认参数和入口组织
3. 文档与测试层中的平台说明
4. 某些 helper 的写法是否能更贴近官方风格，同时保持当前行为不变

这些属于“工程形态收敛”，不是“当前核心逻辑回退”。

## 当前主线验证状态

截至本轮：

- `main` 当前可见非性能测试集：`62 / 62 passed`
- 已验证两种口径：
  1. `HIPRT_ENABLE_RUNTIME_KERNEL_CACHE=OFF` / `HIPRT_DISABLE_RUNTIME_KERNEL_CACHE=1`
  2. `HIPRT_ENABLE_RUNTIME_KERNEL_CACHE=ON` / `HIPRT_DISABLE_RUNTIME_KERNEL_CACHE=0`

因此，当前主线上的核心行为差异，至少在现有测试覆盖下，都是“有必要且有效”的。
