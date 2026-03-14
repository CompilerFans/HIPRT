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

- 总通过率：`55 / 60`，约 `91.7%`
- `ObjTestCases`：`17 / 20` 通过
- `hiprtTest`：`38 / 40` 通过

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
- `SceneWorldToObjectRayMatrixShear`
- `SceneTransformDebugSrt`
- `SceneInterpolatedFrameDebugSrt`
- `SceneClosestHitSingletonSrt`
- `SceneManualClosestHitSingletonSrt`
- `GeomClosestHitScaledRay`
- `RotateCornellBoxSmallAngleNoRef`
- `RecreateCornellBoxTwiceBuildOnly`
- `RenderCornellBoxTwiceSameSceneNoRef`
- `SceneIntersectionSingleton`
- `SceneIntersection`
- `SceneIntersectionMlas`
- `Shear`

## 当前未支持 / 未收敛的 case

### 变换相关

- `hiprtTest.SceneInternalTransformRaySrt`
  现象：device 侧 `Transform::transformRay()` 仍返回未缩放的 ray
- `hiprtTest.SceneInverseMatrixDebugSrt`
  现象：device 侧 inverse matrix 对角线仍为 `1`，预期为 `2`

### Recreate / lifecycle 诊断

- `ObjTestCases.RecreateCornellBoxTwiceNoRef`
  现象：第二次 recreate + render 会 device trap
- `ObjTestCases.RecreateCornellBoxTwiceSameTransformNoRef`
  现象：即使 transform 不变，第二次 recreate + render 仍会 device trap

### Update 路径

- `ObjTestCases.BvhUpdateCornellBox`
  现象：像素差异约 `60%`

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
- `RecreateCornellBoxTwiceBuildOnly` / `RenderCornellBoxTwiceSameSceneNoRef`
  已恢复通过，说明当前问题不在“重复 build”本身，也不在“同一 scene 连续 render 两次”本身

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
  7. 仍保留的失败诊断集中在 `Transform::transformRay()` / inverse matrix 的 device 内部实现
- 当前剩余问题已经集中到：
  1. transform 内部实现诊断
  2. recreate 后再次 render 的生命周期问题
  3. scene / geometry update 正确性

## 后续收敛顺序

1. `RecreateCornellBoxTwiceSameTransformNoRef`
2. `RecreateCornellBoxTwiceNoRef`
3. `BvhUpdateCornellBox`
4. `SceneInternalTransformRaySrt`
5. `SceneInverseMatrixDebugSrt`
