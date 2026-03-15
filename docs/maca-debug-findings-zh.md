# MACA 调试结论

本文档记录 **MACA + cu-bridge** 迁移过程中，已经通过新增诊断单测或最小复现实验得到的阶段性结论。

它不替代 `docs/maca-test-status-zh.md`。

- `maca-test-status-zh.md` 关注“哪些 case 过了、哪些没过”
- 本文档关注“当前已经被证实的事实、已排除的假设、还待验证的方向”

## 当前结论基线

- 时间：`2026-03-14`
- 代码基线：当前 `main` 上已 push 的迁移修复与诊断单测
- 结论来源：
  - 现有功能 case 的串行回归
  - 新增诊断单测
  - 最小 host 侧复算 / AABB 导出验证

## 已证实的事实

### 1. runtime kernel disk cache 在常规测试中应开启，调试阶段才建议关闭

- 根级 `cache/` 会缓存 JIT 结果。
- 只清 `scripts/cache/` 不足以让运行时内核重新编译。
- 在常规 `Release` 测试中，开启 cache 可以显著减少重复 runtime compile 时间。
- 在调试阶段，旧 cache 会掩盖头文件修改，导致“源码已改、运行行为未变”的假象。
- 当前仓库已经恢复 runtime kernel disk cache 的原始默认策略：`Release` 为 `ON`，非 `Release` 为 `OFF`。
- 当前测试脚本默认保留 cache 开启，并保留：
  - CMake 开关：`HIPRT_ENABLE_RUNTIME_KERNEL_CACHE`
  - 运行时环境变量：`HIPRT_DISABLE_RUNTIME_KERNEL_CACHE=1`

### 2. wave64 相关 bitmask 问题是第一层公共阻塞点

- `PairTriangles`、`Collapse`、`subwarpMask`、`packetMask` 等路径中，原先存在典型的 32-bit mask 用法。
- 修正这些 64-bit mask 之后：
  - `hiprtTest.MinimumCornellBox` 恢复通过
  - `hiprtTest.Compaction` 恢复通过
- 说明这类问题确实是核心构建路径的首要 blocker。

### 3. `BatchCornellBox` 的问题不应再走单对象 batch kernel 路径

- `hiprtTest.BatchConstruction` 能过，说明多对象 batch 路径本身并非完全不可用。
- `hiprtTest.BatchCornellBox` 失败时，实际是“单对象通过 `*Geometries/*Scenes` API 进入了 batch kernel 路径”。
- 当前已对单对象 `*Geometries/*Scenes` 调用做兼容回退，不再走 batch kernel。
- 这条策略已经让 `BatchCornellBox` 恢复通过。

### 4. geometry batch 路径更像 latent OOB，而不是普遍的 batch 不可用

新增诊断测试后，当前已经得到更具体的事实：

- `BatchGeometryCornellSweepBuildOnly`
  - `triangleCount=2/4/8/16` 通过
  - `triangleCount=32` 在 `BatchBuild_hiprtGeometryBuildInput` 上触发 memory violation
- `BatchGeometryIndexedQuadStripSweepBuildOnly`
  - 即使换成**索引 mesh**，边界仍然是 `2/4/8/16` 通过，`32` 失败
- `BatchGeometryIndexedQuadStripPrePairedSweepBuildOnly`
  - 在**预先提供 `trianglePairIndices`** 后，`2/4/8/16/32` 全部通过

这说明：

- 问题不像“batch 几何内核整体不可用”
- 更像是“未配对 triangle mesh 的 batch 几何路径”在规模到 `32` 左右时触发 latent OOB
- 这类问题在 CUDA 上有可能因为 padding / UB / 对齐余量未立即崩溃，但在当前 MACA 环境下被更早暴露

因此当前最优先方向是：

- 检查 batch geometry kernel 是否缺少与普通 builder 等价的 triangle pairing / packet 预处理
- 继续对比 host 侧分配大小与 device 侧实际写入大小，排查真实越界来源

## 变换 / scene 方向的已证实事实

### 5. scene AABB 构建本身是正确的

以下诊断单测已通过：

- `SceneAabbSingletonSrt`
- `SceneAabbSingletonMatrixShear`

这说明：

- 单实例 `SRT` scene 的 world-space AABB 构建正确
- 单实例 `Matrix + shear` scene 的 world-space AABB 构建正确

因此，剩余问题**不在 scene build 的 AABB 生成本身**。

### 6. scene header / instance node 的 frame 元数据写入是正确的

以下诊断单测已通过：

- `SceneSingletonSrtNodeUsesTransformHeader`
- `SceneTransformDebugSrt`
- `SceneInterpolatedFrameDebugSrt`

这说明：

- `sceneHeader->m_frames[0]` 中保存的缩放值正确
- `instanceNode.m_transform.frameIndex/frameCount` 正确
- `instanceNode.m_static == 0`
- `instanceNode.m_identity == 0`
- `Transform::interpolateFrames(0.0f)` 取出的 `Frame` 仍然是正确的

因此，剩余问题**不在 frame/header 元数据写入，也不在 `Transform` 的 frame 选择逻辑**。

### 7. helper 形式的 world-to-object 变换是正确的

以下诊断单测已通过：

- `SceneWorldToObjectRaySrt`
- `SceneWorldToObjectRayMatrixShear`

这说明通过：

- `hiprtGetWorldToObjectFrameMatrix(...)`
- `hiprtPointWorldToObject(...)`

拿到的单层 instance world-to-object 结果，在当前测试覆盖下是正确的。

因此，剩余问题**不在这组 helper API 的单层结果**。

### 8. geometry traversal 本身是正确的

以下诊断单测已通过：

- `GeomClosestHitScaledRay`

它证明：

- 在 geometry local space 下，给定正确的 local ray，`hiprtGeomTraversalClosest` 命中行为正确。

因此，剩余问题**不在 geometry traversal / triangle hit 这一层**。

### 9. helper 变换 + geometry traversal 组合也是正确的

以下诊断单测已通过：

- `SceneManualClosestHitSingletonSrt`

它证明：

- 从 scene 中取出 instance geometry
- 先通过 helper 路径把 world ray 变到 object/local space
- 再直接执行 `hiprtGeomTraversalClosest`

这一整条组合路径是正确的。

因此，剩余问题进一步收敛为：

- `hiprtSceneTraversalClosest` 自身的 instance traversal / control flow
  或
- 它内部使用的 `Transform::transformRay()`

### 10. 当前 scene traversal 功能路径已恢复

以下 case 现已恢复通过：

- `SceneClosestHitSingletonSrt`
- `SceneIntersectionSingleton`
- `SceneIntersection`
- `SceneIntersectionMlas`
- `TranslateCornellBox`
- `ScaleCornellBox`
- `RotateCornellBox`
- `Shear`

当前恢复方式：

- scene build 完成后，在 host 侧对 `InstanceNode` 做一次补丁
- 强制把单帧 instance 节点切回动态 transform 路径

因此，当前功能性失败已经不再集中在 scene traversal 主路径，而是缩小为：

- recreate 后再次 render 的生命周期组合路径
- scene / geometry update
- transform 内部实现诊断

## 当前最关键的失败诊断

### 11. `Transform::transformRay()` 诊断在当前工作树下已恢复

以下诊断单测当前已通过：

- `SceneInternalTransformRaySrt`

这说明：

- 至少在当前工作树下，`Transform::transformRay()` 不再是最直接的功能 blocker
- 它仍然值得保留为回归测试，因为之前确实出现过错误结果

### 12. `MatrixFrame::getMatrixFrameInv(frame)` 诊断在当前工作树下已恢复

以下诊断单测当前已通过：

- `SceneInverseMatrixDebugSrt`

这进一步说明：

- 这条诊断目前不再是主 blocker
- 但它仍应保留为红灯回归基线，防止后续 scene transform 修正时倒退

### 13. recreate 后再次 render 问题在当前工作树下已恢复

以下新增诊断单测当前已通过：

- `RecreateCornellBoxTwiceBuildOnly`：通过
- `RenderCornellBoxTwiceSameSceneNoRef`：通过
- `RecreateCornellBoxTwiceNoRef`：通过
- `RecreateCornellBoxTwiceSameTransformNoRef`：通过

当前结论：

- recreate + render 路径在当前工作树下已恢复
- 这组测试应继续保留，防止后续再次出现生命周期回归

### 14. trace kernel rebuild / custom stack 问题在当前工作树下已恢复

以下诊断单测当前已通过：

- `SceneTraceKernelSingletonSrt`
- `PrimaryRayKernelRecreateStableRegs`

当前结论：

- custom/global stack 路径在有效 launch 配置下已恢复
- recreate 后同一个 `PrimaryRayKernel` 的 runtime compile / load 已恢复稳定
- 这两条测试应继续保留，作为回归守卫

## 已排除的错误方向

截至当前，以下方向已不再是优先怀疑对象：

- scene AABB 构建
- instance frame/header 写入
- helper 形式的 world-to-object 变换
- geometry local-space traversal
- 多数基础 BVH build / intersection / motion blur 基础路径

## 当前最优先分析顺序

1. 保持当前诊断测试为回归门禁
2. 如后续出现 regressions，优先重放：
   - `SceneTraceKernelSingletonSrt`
   - `PrimaryRayKernelRecreateStableRegs`
   - `RecreateCornellBoxTwiceSameTransformNoRef`
   - `RecreateCornellBoxTwiceNoRef`
   - `BvhUpdateCornellBox`

## 与当前状态文档的关系

- 当前 case 通过 / 失败快照见：`docs/maca-test-status-zh.md`
- 当前文档只沉淀“已经证实的技术结论”和“下一步最该查什么”
