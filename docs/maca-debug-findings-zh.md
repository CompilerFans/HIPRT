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

### 1. runtime kernel disk cache 不应在调试阶段默认开启

- 根级 `cache/` 会缓存 JIT 结果。
- 只清 `scripts/cache/` 不足以让运行时内核重新编译。
- 在调试阶段，旧 cache 会掩盖头文件修改，导致“源码已改、运行行为未变”的假象。
- 当前仓库已经把 runtime kernel disk cache 默认值改为 `OFF`，并保留：
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

## 变换 / scene 方向的已证实事实

### 4. scene AABB 构建本身是正确的

以下诊断单测已通过：

- `SceneAabbSingletonSrt`
- `SceneAabbSingletonMatrixShear`

这说明：

- 单实例 `SRT` scene 的 world-space AABB 构建正确
- 单实例 `Matrix + shear` scene 的 world-space AABB 构建正确

因此，剩余问题**不在 scene build 的 AABB 生成本身**。

### 5. scene header / instance node 的 frame 元数据写入是正确的

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

### 6. helper 形式的 world-to-object 变换是正确的

以下诊断单测已通过：

- `SceneWorldToObjectRaySrt`
- `SceneWorldToObjectRayMatrixShear`

这说明通过：

- `hiprtGetWorldToObjectFrameMatrix(...)`
- `hiprtPointWorldToObject(...)`

拿到的单层 instance world-to-object 结果，在当前测试覆盖下是正确的。

因此，剩余问题**不在这组 helper API 的单层结果**。

### 7. geometry traversal 本身是正确的

以下诊断单测已通过：

- `GeomClosestHitScaledRay`

它证明：

- 在 geometry local space 下，给定正确的 local ray，`hiprtGeomTraversalClosest` 命中行为正确。

因此，剩余问题**不在 geometry traversal / triangle hit 这一层**。

### 8. helper 变换 + geometry traversal 组合也是正确的

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

### 9. 当前 scene traversal 功能路径已恢复

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

### 10. `Transform::transformRay()` 的 device 结果与期望不一致

以下诊断单测仍失败：

- `SceneInternalTransformRaySrt`

现象：

- device 侧 `Transform tr(...).transformRay(ray, 0.0f)` 返回的 ray 仍保持原始尺度
- 按当前测试预期，它应当返回被 world-to-object 缩放后的 ray

这说明：

- 即便 `interpolateFrames()` 结果正确
- `transformRay()` 的 device 路径仍然是当前高度可疑点

### 11. `MatrixFrame::getMatrixFrameInv(frame)` 的 device 结果也可疑

以下诊断单测仍失败：

- `SceneInverseMatrixDebugSrt`

现象：

- 对 scale=`0.5`
- device 侧 `MatrixFrame::getMatrixFrameInv(frame)` 的对角线仍返回 `1`
- 预期应为 `2`

这进一步说明：

- 问题很可能就在 `Frame -> inverse matrix` 这一层
- 或它在 device 编译路径下与 host 侧结果不一致

### 12. recreate 后再次 render 仍然会触发 device trap

以下新增诊断单测已经形成了一个很小的切片：

- `RecreateCornellBoxTwiceBuildOnly`：通过
- `RenderCornellBoxTwiceSameSceneNoRef`：通过
- `RecreateCornellBoxTwiceNoRef`：失败
- `RecreateCornellBoxTwiceSameTransformNoRef`：失败

这说明当前剩余的功能性生命周期问题不是：

- “重复 build” 本身
- “同一 scene 连续 render 两次” 本身

而是：

- “一个 scene destroy 之后，再 create 新 scene，再 render” 这条组合路径

这条现象与 `BvhUpdateCornellBox` 当前的失败形态一致，因此二者很可能共享根因。

## 已排除的错误方向

截至当前，以下方向已不再是优先怀疑对象：

- scene AABB 构建
- instance frame/header 写入
- helper 形式的 world-to-object 变换
- geometry local-space traversal
- 多数基础 BVH build / intersection / motion blur 基础路径

## 当前最优先分析顺序

1. `MatrixFrame::getMatrixFrameInv(frame)` 的 device 结果为什么与 host 预期不同
2. `Transform::transformRay()` 为什么在 device 侧没有体现缩放
3. recreate 后再次 render 时，JIT module / context / scene buffer 生命周期是否仍有未清理状态
4. `BvhUpdateCornellBox` 为什么在当前 scene node host-side patch 之后仍然失败
5. 在上述问题修正后，再回归：
   - `RecreateCornellBoxTwiceSameTransformNoRef`
   - `RecreateCornellBoxTwiceNoRef`
   - `BvhUpdateCornellBox`
   - `SceneInternalTransformRaySrt`
   - `SceneInverseMatrixDebugSrt`

## 与当前状态文档的关系

- 当前 case 通过 / 失败快照见：`docs/maca-test-status-zh.md`
- 当前文档只沉淀“已经证实的技术结论”和“下一步最该查什么”
