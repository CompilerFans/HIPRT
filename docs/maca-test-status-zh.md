# MACA 测试状态

本文档记录 HIPRT 在 **MACA + cu-bridge** 环境下的当前测试状态，并作为后续迁移与收敛工作的基线。

调试层面的阶段性结论见：`docs/maca-debug-findings-zh.md`

## 测试来源

- 本次状态基线来自本地串行回归结果：
  `reports/case-report-20260314-172722-current/summary_current.tsv`
- 统计时间基于 **2026-03-14** 的当前工作区代码状态。
- 结果口径以**串行 targeted rerun** 为准，不以并发跑批结果为准。

## 更新原则

- 只有在一轮代码修复后完成新的串行回归，才更新本文档。
- 允许使用多 session / 多 worker 做初始采样，但不能直接把并发结果当作最终状态。
- 如果某个 case 的结果依赖临时绕行或兼容策略，需要在本文档中明确写出。
- 如果 case 从 `failure` 变为 `success`，需要同时说明本轮修正的原因与限制。

## 观测原则

- `success`：
  进程退出成功，且 gtest 未失败。
- `failure`：
  包括 gtest 断言失败、非法内存访问、设备 trap、运行时编译失败等。
- 图像类 case 以现有测试 harness 的阈值为准：
  只要未触发 `validateAndWriteImage()` 的失败条件，就记为通过。
- 允许“带观测通过”：
  例如存在少量 `Pixel difference` 输出但仍在阈值内，或 `TraceKernel` 存在运行时编译 warning 但测试仍通过。
- 文档中的“支持 / 不支持”是**当前测试结论**，不是长期 API 保证。

## 当前结果概览

- 总通过率：`59 / 59`，即 `100%`
- `ObjTestCases`：`21 / 21` 通过
- `hiprtTest`：`38 / 38` 通过

## 当前支持的 case

### ObjTestCases

- `BvhFastCornellBox`
- `BvhHighQCornellBox`
- `ShadowRayCornellBox`
- `AoRayCornellBox`
- `AoRayEmbreeCornellBox`
- `UvsCornellBox`
- `PrimIdsCornellBox`
- `HitDistCornellBox`
- `NormalsCornellBox`
- `BvhBalancedCornellBox`
- `PrimaryRayCornellBox`

### hiprtTest

- `CudaEnabled`
- `MinimumCornellBox`
- `Compaction`
- `BatchCornellBox`
- `BoundingBox`
- `CustomBvhImport`
- `BvhIoApi`
- `MeshIntersection`
- `MeshIntersectionNonIndexed`
- `PairTriangles`
- `Cutout`
- `CustomIntersection`
- `MotionBlur`
- `MotionBlurMatrix`
- `MotionBlurSlerp`
- `Rebuild`
- `Update`
- `PlocFallback`
- `TraceKernel`
- `BatchConstruction`
- `SceneAabbSingletonSrt`
- `SceneAabbSingletonMatrixShear`
- `SceneSingletonSrtNodeUsesTransformHeader`
- `SceneWorldToObjectRaySrt`
- `SceneInternalTransformRaySrt`
- `SceneInverseMatrixDebugSrt`
- `SceneWorldToObjectRayMatrixShear`
- `SceneTransformDebugSrt`
- `SceneInterpolatedFrameDebugSrt`
- `SceneClosestHitSingletonSrt`
- `SceneManualClosestHitSingletonSrt`
- `GeomClosestHitScaledRay`
- `RotateCornellBoxSmallAngleNoRef`
- `RecreateCornellBoxTwiceBuildOnly`
- `RenderCornellBoxTwiceSameSceneNoRef`
- `RecreateCornellBoxTwiceNoRef`
- `RecreateCornellBoxTwiceSameTransformNoRef`
- `PrimaryRayKernelRecreateStableRegs`
- `SceneTraceKernelSingletonSrt`
- `SceneIntersectionSingleton`
- `SceneIntersection`
- `SceneIntersectionMlas`
- `Shear`

## 当前未支持 / 未收敛的 case

- 当前 MACA 基线下，没有失败的非性能 unit test。
- 之前的红灯诊断 case 仍保留在测试集中，用作回归守卫。

## 已知“通过但需关注”的现象

- `hiprtTest.MinimumCornellBox`
  运行时间较长，当前回归约 `117.9s`
- `hiprtTest.TraceKernel`
  测试通过，但仍会打印运行时编译 warning
- `MeshIntersection` / `MeshIntersectionNonIndexed` / `PairTriangles` / `Cutout`
  有少量像素差异输出，但仍在测试阈值内
- `BatchCornellBox`
  当前是通过**单对象 API 不走 batch kernel** 的兼容策略打通的；多对象 batch 构建仍以 `BatchConstruction` 为主验证
- `TranslateCornellBox` / `ScaleCornellBox` / `RotateCornellBox` / `Shear` / `SceneIntersection*`
  已恢复通过，但当前恢复依赖 scene build 后的 instance node host-side patch
- `SceneInternalTransformRaySrt` / `SceneInverseMatrixDebugSrt`
  当前工作树下已恢复通过，继续作为 transform 内部路径的回归基线
- `RecreateCornellBoxTwiceBuildOnly` / `RenderCornellBoxTwiceSameSceneNoRef`
  已恢复通过，说明当前问题不在“重复 build”本身，也不在“同一 scene 连续 render 两次”本身
- `RecreateCornellBoxTwiceNoRef` / `RecreateCornellBoxTwiceSameTransformNoRef` / `PrimaryRayKernelRecreateStableRegs` / `SceneTraceKernelSingletonSrt`
  当前工作树下也已恢复通过，继续作为 recreate / stack / rebuild 稳定性回归基线

## 本轮修复后观察到的主要变化

- wave64 相关 bitmask 问题得到一轮系统性修正后，基础几何 / BVH 构建路径从设备非法访存收敛到可运行状态。
- `MinimumCornellBox`、`Compaction`、`BoundingBox`、`CustomBvhImport`、`BvhIoApi`、`MeshIntersection*`、`PairTriangles`、`Cutout`、`CustomIntersection` 等基础 case 已恢复。
- `BatchCornellBox` 通过对单对象 `*Geometries/*Scenes` API 的兼容回退恢复。
- 新增的 3 条诊断单测已经证明：
  1. 单实例 SRT scene AABB 构建正确
  2. 单实例 matrix shear scene AABB 构建正确
  3. 单实例 SRT scene 的 frame / transform header 已正确写入 scene header 与 instance node
- 新增的 ray / traversal 诊断已经进一步证明：
  1. 单实例 SRT scene AABB 构建正确
  2. `hiprtPointWorldToObject()` 的 SRT 路径正确
  3. `hiprtPointWorldToObject()` 的 matrix shear 路径正确
  4. `hiprtGeomTraversalClosest` 在对应 local ray 上正确
  5. `SceneManualClosestHitSingletonSrt` 已通过，说明 helper 形式的 scene world-to-object + geometry traversal 组合正确
  6. `SceneClosestHitSingletonSrt` 已恢复通过，说明当前 scene traversal 功能路径已经恢复
  7. `SceneInternalTransformRaySrt` 与 `SceneInverseMatrixDebugSrt` 当前工作树下已恢复通过
  8. 当前功能性剩余问题更集中在 recreate 后 trace kernel rebuild / custom stack 路径
- 当前功能性问题已经全部收敛，保留的诊断测试主要用于防止：
  1. recreate 后再次 render 的生命周期回归
  2. custom/global stack 路径回归
  3. scene / geometry update 路径回归

## 后续收敛顺序

1. 继续保持 `recreate` / `stack` / `transform` 诊断测试为回归门禁
2. 如后续再次出现 regressions，优先回放：
   - `SceneTraceKernelSingletonSrt`
   - `PrimaryRayKernelRecreateStableRegs`
   - `RecreateCornellBoxTwiceSameTransformNoRef`
   - `RecreateCornellBoxTwiceNoRef`
   - `BvhUpdateCornellBox`
