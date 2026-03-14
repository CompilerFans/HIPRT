# `maca_init` 之后逻辑改动复盘

本文档复盘 `maca_init` 到当前 `HEAD` 之间的**行为级**代码改动，重点回答：

1. 哪些改动属于真实 bug 修复，不应恢复。
2. 哪些改动只是调试期或兼容期策略，后续应恢复或改成显式开关。
3. 哪些结论已经通过当前代码和 targeted rerun 验证。

## 1. 范围与基线

- 基线 tag：`maca_init`
- 基线提交：`6caaf4a37506f1e0c482358446e835ad82c3d659`
- 当前代码：当前工作树
- 复盘范围：
  - 只看 `maca_init..HEAD` 期间对行为有影响的代码
  - 不把纯文档、纯测试新增、构建产物纳入“是否恢复”的判断

## 2. 复盘方法

本次判断分两层：

1. **源码对照**
   重点查看：
   - `hiprt/impl/Context.cpp`
   - `hiprt/impl/BvhNode.h`
   - `hiprt/impl/hiprt_device_impl.h`
   - `hiprt/impl/Transform.h`
   - `hiprt/impl/Compiler.cpp`
   - `hiprt/hiprt_common.h`
   - `hiprt/impl/BvhBuilderKernels.h`
   - `hiprt/impl/BvhBuilderUtil.h`
   - `hiprt/impl/PlocBuilderKernels.h`

2. **当前代码 targeted rerun**
   所有命令都在当前 `HEAD` 上执行，统一使用：

```bash
cd /data/HIPRT/scripts
../dist/bin/Release/unittest64 --width=512 --height=512 --referencePath=../test/references ...
```

## 3. 总结结论

### 3.1 明确不建议恢复

以下改动当前仍属于真实修复，不建议恢复：

1. wave32/wave64 与 maskbit 相关修复
2. geometry 单对象 `*Geometries` 不再走 batch path

### 3.2 不应误用为常规测试默认值，但调试期仍合理

以下改动更适合作为开发策略，而不是最终默认行为：

1. runtime kernel disk cache 调试时显式关闭

### 3.3 已经验证可回归原始逻辑

以下改动当前已经通过 targeted rerun 验证，可回归 `maca_init` 语义：

1. `PreferHighQualityBuild` 在非 NVIDIA 上的静默降级
2. scene 单对象 `*Scenes` 不再走 batch path
3. scene build 后 host-side `patchSceneInstanceNodes()`
4. `BvhNode::init()` 中单帧 instance 的原始静态 matrix / 动态 transform 分支
5. `hiprt_device_impl` 中 scene dynamic transform 分支使用 `Transform tr(...).transformRay(...)`
6. `Transform::transformRay()` 恢复为 `frame.invTransform(...)` 的原始实现

### 3.4 当前最值得恢复或改成显式开关

以下改动不应继续以“静默兼容”的形式保留：

1. `PreferHighQualityBuild` 在非 NVIDIA 上被静默降级为 `PreferFastBuild`

### 3.5 仍待继续分析

1. geometry 单对象 batch 路径的根因仍需继续分析

## 4. 分项结论

## 4.1 wave32/wave64 与 maskbit 修复

### 当前判断

- 结论：**必须保留**
- 原因：这是公共语义修复，不是绕行策略

### 关键源码点

- `hiprt/hiprt_common.h`
  - `HIPRT_WARP_SIZE` / `WarpSize`
  - `LaneMask`
  - `laneMaskFirstSet()` / `laneMaskPopCount()` / `laneMaskLowerCount()` / `subLaneMask()`
- `hiprt/impl/BvhBuilderKernels.h`
- `hiprt/impl/BvhBuilderUtil.h`
- `hiprt/impl/PlocBuilderKernels.h`
- `hiprt/impl/hiprt_device_impl.h`

### 本质影响

- 这是对 wave mask 宽度和位操作的一致性修复
- 影响的是：
  - `ballot()` 返回值宽度
  - `__ffsll` / `__popcll` 这一类 mask 处理
  - `subwarpMask` / `packetMask` / `warpBallot` 的位数语义

恢复这类改动，会重新把 MACA 上的 wave64 路径带回 32-bit 假设。

### 本轮验证

执行：

```bash
--gtest_filter=hiprtTest.MinimumCornellBox:hiprtTest.Compaction:hiprtTest.PairTriangles:hiprtTest.PlocFallback
```

结果：

- `MinimumCornellBox`：通过
- `Compaction`：通过
- `PairTriangles`：通过
- `PlocFallback`：通过

结论：

- 当前这些直接依赖公共 mask 语义的基础路径都已通过
- 因此 wave/mask 修复不是可恢复项

## 4.2 单对象 batch 回退

### 当前判断

- 结论：**需要保留 geometry 单对象回退；scene 单对象回退已可恢复原始逻辑**
- 原因：两条路径的当前结论不同，不能再合并看

### 关键源码点

- geometry：
  - `Context::createGeometries`
  - `Context::buildGeometries`
  - `Context::getGeometriesBuildTempBufferSize`
- scene：
  - `Context::createScenes`
  - `Context::buildScenes`
  - `Context::getScenesBuildTempBufferSize`

这使得单对象 `*Geometries/*Scenes` 不再因为 API 入口形式而误入 batch kernel。

### 本质影响

- 这改变的是**内部调度策略**
- 不改变公开 API 输入输出语义
- 但 geometry 与 scene 两条路径的稳定性目前已经分化

### 本轮验证

#### geometry 单对象 batch 路径

在仅恢复 geometry 单对象 batch 原始行为后，完整重编再执行：

```bash
--gtest_filter=hiprtTest.BatchConstruction:hiprtTest.BatchCornellBox
```

结果：

- `BatchCornellBox` 在 `BatchBuild_hiprtGeometryBuildInput` 上触发设备非法访存
- 后续还出现过 `EmitTopologyAndFitBounds_TriangleMesh` / `LbvhBuilder` 一侧的非法访问级联错误

结论：

- geometry 单对象 batch 回退**仍然必要**

#### scene 单对象 batch 路径

执行：

```bash
--gtest_filter=hiprtTest.SceneAabbSingletonSrt:hiprtTest.SceneAabbSingletonMatrixShear:hiprtTest.SceneSingletonSrtNodeUsesTransformHeader:hiprtTest.SceneWorldToObjectRaySrt:hiprtTest.SceneInternalTransformRaySrt:hiprtTest.SceneTransformDebugSrt:hiprtTest.SceneInterpolatedFrameDebugSrt:hiprtTest.SceneInverseMatrixDebugSrt:hiprtTest.SceneWorldToObjectRayMatrixShear:hiprtTest.SceneClosestHitSingletonSrt:hiprtTest.SceneClosestHitSingletonSrtRecreate:hiprtTest.SceneManualClosestHitSingletonSrt:hiprtTest.SceneTraceKernelSingletonSrt:hiprtTest.SceneIntersectionSingleton:hiprtTest.SceneIntersection:hiprtTest.SceneIntersectionMlas:ObjTestCases.TranslateCornellBox:ObjTestCases.ScaleCornellBox:ObjTestCases.RotateCornellBox:hiprtTest.Shear
```

结果：

- `20 / 20` 通过

结论：

- scene 单对象 batch 回退当前**已非必要**
- scene 这半边可以恢复到 `maca_init` 的原始逻辑

## 4.3 scene host patch 与 instance 节点语义

### 当前判断

- 结论：**当前已验证可恢复原始逻辑**
- 原因：去掉 host patch，并恢复 `BvhNode::init()` 的原始 instance 节点语义后，scene / transform / motion blur 覆盖仍通过

### 关键源码点

- `hiprt/impl/BvhNode.h`
- `hiprt/impl/Context.cpp`
- `hiprt/impl/hiprt_device_impl.h`

### 本质影响

这一组改动在前一阶段被作为 scene 主路径的兜底策略引入，但当前逐步回归后，已经能把下列逻辑一起恢复：

1. 删除 host-side `patchSceneInstanceNodes()`
2. 恢复 `BvhNode::init()` 中单帧 instance 的原始静态 matrix 分支
3. 恢复 `hiprt_device_impl` 中 dynamic transform 分支直接调用 `Transform tr(...).transformRay(...)`

### 本轮验证

执行：

```bash
--gtest_filter=hiprtTest.SceneAabbSingletonSrt:hiprtTest.SceneAabbSingletonMatrixShear:hiprtTest.SceneSingletonSrtNodeUsesTransformHeader:hiprtTest.SceneWorldToObjectRaySrt:hiprtTest.SceneInternalTransformRaySrt:hiprtTest.SceneTransformDebugSrt:hiprtTest.SceneInterpolatedFrameDebugSrt:hiprtTest.SceneInverseMatrixDebugSrt:hiprtTest.SceneWorldToObjectRayMatrixShear:hiprtTest.SceneClosestHitSingletonSrt:hiprtTest.SceneClosestHitSingletonSrtRecreate:hiprtTest.SceneManualClosestHitSingletonSrt:hiprtTest.SceneTraceKernelSingletonSrt:hiprtTest.SceneIntersectionSingleton:hiprtTest.SceneIntersection:hiprtTest.SceneIntersectionMlas:ObjTestCases.TranslateCornellBox:ObjTestCases.ScaleCornellBox:ObjTestCases.RotateCornellBox:hiprtTest.Shear
```

结果：

- `20 / 20` 通过

再执行：

```bash
--gtest_filter=hiprtTest.MotionBlur:hiprtTest.MotionBlurMatrix:hiprtTest.MotionBlurSlerp
```

结果：

- `3 / 3` 通过

结论：

- host-side `patchSceneInstanceNodes()` 当前已非必要
- `BvhNode::init()` 的原始语义当前已恢复可用
- `hiprt_device_impl` 的原始 dynamic transform 路径当前已恢复可用

## 4.4 `Transform::transformRay()` 改动

### 当前判断

- 结论：**当前已验证可恢复原始逻辑**
- 原因：把 `Transform.h` 回归到 `maca_init` 的 `frame.invTransform(...)` 实现后，scene / transform / motion blur 覆盖仍然通过

### 关键源码点

- `hiprt/impl/Transform.h:345`
- `hiprt/impl/hiprt_device_impl.h:983`

### 当前状态

- `maca_init` 时，`Transform::transformRay()` 直接使用 `frame.invTransform(...)`
- 当前已直接恢复到这条实现

### 本轮验证

执行：

```bash
--gtest_filter=hiprtTest.SceneAabbSingletonSrt:hiprtTest.SceneAabbSingletonMatrixShear:hiprtTest.SceneSingletonSrtNodeUsesTransformHeader:hiprtTest.SceneWorldToObjectRaySrt:hiprtTest.SceneInternalTransformRaySrt:hiprtTest.SceneTransformDebugSrt:hiprtTest.SceneInterpolatedFrameDebugSrt:hiprtTest.SceneInverseMatrixDebugSrt:hiprtTest.SceneWorldToObjectRayMatrixShear:hiprtTest.SceneClosestHitSingletonSrt:hiprtTest.SceneClosestHitSingletonSrtRecreate:hiprtTest.SceneManualClosestHitSingletonSrt:hiprtTest.SceneTraceKernelSingletonSrt:hiprtTest.SceneIntersectionSingleton:hiprtTest.SceneIntersection:hiprtTest.SceneIntersectionMlas:ObjTestCases.TranslateCornellBox:ObjTestCases.ScaleCornellBox:ObjTestCases.RotateCornellBox:hiprtTest.Shear
```

结果：

- `20 / 20` 通过

再执行：

```bash
--gtest_filter=hiprtTest.MotionBlur:hiprtTest.MotionBlurMatrix:hiprtTest.MotionBlurSlerp
```

结果：

- `3 / 3` 通过

结论：

- `Transform::transformRay()` 当前已可恢复到 `maca_init` 原始实现

## 4.5 runtime kernel disk cache 调试时显式关闭

### 当前判断

- 结论：**调试期合理，常规测试不应默认关闭**
- 原因：这是调试策略，不是正确性修复

### 关键源码点

- `CMakeLists.txt:13`
- `hiprt/impl/Compiler.cpp:37`
- `hiprt/impl/Compiler.cpp:161`

### 本质影响

- 当前默认策略：
  - `Release`：`HIPRT_ENABLE_RUNTIME_KERNEL_CACHE=ON`
  - 非 `Release`：`HIPRT_ENABLE_RUNTIME_KERNEL_CACHE=OFF`
- 同时支持运行时强制关闭：
  - `HIPRT_DISABLE_RUNTIME_KERNEL_CACHE=1`

这能避免旧 cache 掩盖 JIT/header 修改，但它属于调试控制，不应再作为常规 `Release` 测试的默认运行方式。

### 本轮验证

- 调试时仍可使用：

```bash
HIPRT_DISABLE_RUNTIME_KERNEL_CACHE=1
```

结论：

- 这项控制对开发调试仍有价值
- 常规 `Release` 测试应保持 cache 开启，以减少重复 runtime compile 时间

## 4.6 `PreferHighQualityBuild` 静默降级

### 当前判断

- 结论：**已验证可恢复原语义；不应继续静默保留**
- 原因：这不是 bug fix，而是语义改写，而且当前工作树下移除后相关 non-performance case 已通过

### 关键源码点

- `hiprt/impl/Context.cpp:39`
- `hiprt/impl/Context.cpp:42`

当前逻辑：

- 当请求 `hiprtBuildFlagBitPreferHighQualityBuild`
- 且设备名不包含 `NVIDIA`
- 就直接改写为 `hiprtBuildFlagBitPreferFastBuild`

### 本质影响

- 这会让调用方请求 HQ builder，但实际执行 fast builder
- 当前测试仍然在明确请求 HQ：
  - `test/main.cpp:454`
  - `test/main.cpp:690`
  - `test/main.cpp:771`

同时文档又把这些 case 记为“已支持”，例如：

- `README.md:61`
- `docs/maca-test-status-zh.md:44`

因此当前问题不是“功能是否能跑”，而是：

- **测试和文档口径把“请求 HQ 且最终通过”误写成了“当前支持 HQ builder”**

### 本轮验证

先做源码语义确认：

- 代码中 flag 在进入 builder 分发前已被改写，因此之前的通过结果不能证明真正执行了 HQ builder

随后在当前工作树中临时移除静默降级逻辑，再执行：

```bash
--gtest_filter=ObjTestCases.BvhHighQCornellBox:hiprtTest.MinimumCornellBox:hiprtTest.Compaction
```

结果：

- `ObjTestCases.BvhHighQCornellBox`：通过
- `hiprtTest.MinimumCornellBox`：通过
- `hiprtTest.Compaction`：通过

额外观察：

- `BvhHighQCornellBox` 在移除降级后，`Bvh build time` 明显升高到约 `26.6s`
- 这说明当前运行的已不再是先前的 fast fallback，而是真正的 HQ builder 路径

### 建议

当前更推荐：

1. 直接恢复原语义：
   - 请求 HQ 就真的走 HQ
   - 不支持再显式失败
2. 如果未来还要保留 fallback，必须改成显式控制：
   - 例如单独的 MACA fallback 开关
   - 同时修正文档和测试口径

当前不建议继续保留“静默降级”。

## 4.7 `BvhNode::init()` 中的死赋值

### 当前判断

- 结论：**建议清理**
- 原因：这是代码噪声，不是行为策略

### 关键源码点

- `hiprt/impl/BvhNode.h:1163`
- `hiprt/impl/BvhNode.h:1171`
- `hiprt/impl/BvhNode.h:1210`
- `hiprt/impl/BvhNode.h:1218`

### 本质影响

- `m_static` 先按 `transform.frameCount == 1` 赋值
- 随后又立刻被覆盖为 `0`

这不影响当前行为判断，但会误导阅读者，以为 static path 仍然部分保留在构造逻辑中。

建议后续单独清理，不要把它和“恢复 static path”混为一谈。

## 5. 当前建议顺序

如果按“最值得先处理”的顺序排：

1. 优先处理 `PreferHighQualityBuild` 静默降级
   - 恢复原语义，或改成显式 fallback 开关
2. 保持当前 batch 单对象回退与 scene transform/host patch 不动
3. 保持 wave/mask 修复不动
4. 调试时按需显式关闭 runtime cache；常规测试保持开启
5. 单独清理 `BvhNode::init()` 的死赋值

## 6. 本轮验证命令汇总

### 6.1 wave/mask 基础路径

```bash
cd /data/HIPRT/scripts
../dist/bin/Release/unittest64 \
  --width=512 --height=512 --referencePath=../test/references \
  --gtest_filter=hiprtTest.MinimumCornellBox:hiprtTest.Compaction:hiprtTest.PairTriangles:hiprtTest.PlocFallback
```

结果：`4 / 4` 通过

### 6.2 batch 路径

```bash
cd /data/HIPRT/scripts
../dist/bin/Release/unittest64 \
  --width=512 --height=512 --referencePath=../test/references \
  --gtest_filter=hiprtTest.BatchConstruction:hiprtTest.BatchCornellBox
```

结果：`2 / 2` 通过

### 6.3 scene transform / traversal 诊断

```bash
cd /data/HIPRT/scripts
../dist/bin/Release/unittest64 \
  --width=512 --height=512 --referencePath=../test/references \
  --gtest_filter=hiprtTest.SceneSingletonSrtNodeUsesTransformHeader:hiprtTest.SceneWorldToObjectRaySrt:hiprtTest.SceneClosestHitSingletonSrt:hiprtTest.SceneInternalTransformRaySrt:hiprtTest.SceneInverseMatrixDebugSrt
```

结果：`5 / 5` 通过
