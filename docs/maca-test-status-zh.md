# MACA 测试状态

本文档记录 HIPRT 在 **MACA + cu-bridge** 环境下的当前测试状态，并作为后续迁移与收敛工作的基线。

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

- 总通过率：`38 / 48`，约 `79.2%`
- `ObjTestCases`：`11 / 15` 通过
- `hiprtTest`：`27 / 33` 通过

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
- `GeomClosestHitScaledRay`

## 当前未支持 / 未收敛的 case

### 变换相关

- `ObjTestCases.TranslateCornellBox`
  现象：像素差异约 `40%`
- `ObjTestCases.ScaleCornellBox`
  现象：像素差异约 `100%`
- `ObjTestCases.RotateCornellBox`
  现象：像素差异约 `60%`
- `hiprtTest.Shear`
  现象：像素差异约 `20%`

### Scene intersection 正确性

- `hiprtTest.SceneIntersectionSingleton`
  现象：像素差异约 `20%`
- `hiprtTest.SceneIntersection`
  现象：像素差异约 `10%`
- `hiprtTest.SceneIntersectionMlas`
  现象：像素差异约 `30%`
- `hiprtTest.SceneClosestHitSingletonSrt`
  现象：单实例 SRT scene 下中心射线仍 miss，geometry 对照组可 hit

### Transform 内部路径诊断

- `hiprtTest.SceneInternalTransformRaySrt`
  现象：device 侧 `Transform::transformRay()` 仍返回未缩放的 ray

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
  5. 剩余问题集中在 scene traversal / internal transformRay 路径，而不是 scene AABB 构建
- 当前剩余问题已经集中到：
  1. instance transform / scene traversal 正确性
  2. scene update 正确性

## 后续收敛顺序

1. `SceneIntersectionSingleton`
2. `SceneIntersection`
3. `SceneIntersectionMlas`
4. `TranslateCornellBox`
5. `ScaleCornellBox`
6. `RotateCornellBox`
7. `Shear`
8. `BvhUpdateCornellBox`
9. `SceneClosestHitSingletonSrt`
10. `SceneInternalTransformRaySrt`
