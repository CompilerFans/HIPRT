# HIPRT 主要 API 使用指南

## 目标与范围

这份文档只讲当前仓库最常用的 **host-side public API** 用法，帮助你快速把下面这条主链串起来：

1. 创建 `hiprtContext`
2. 构建 `hiprtGeometry`
3. 构建 `hiprtScene`
4. 按需创建 `hiprtFuncTable`
5. 构建或加载 trace kernel
6. 启动 kernel 并销毁资源

本文不展开 device-side traversal 细节，只在最后给出入口位置。

相关背景文档：

- [CUDA-only 编译与改造说明](cuda-only-build-zh.md)
- [bitcode / precompile / bake_kernel 当前状态](bitcode-status-zh.md)

## 一张图看主流程

```text
hiprtCreateContext
  -> hiprtCreateGeometry
  -> hiprtBuildGeometry
  -> hiprtCreateScene
  -> hiprtBuildScene
  -> hiprtCreateFuncTable / hiprtSetFuncTable   (可选)
  -> hiprtBuildTraceKernels*                    (三选一)
  -> launch kernel
  -> hiprtDestroy*
```

最常见的对象生命周期是：

- `hiprtContext` 持有全局运行时状态
- `hiprtGeometry` 是底层几何加速结构
- `hiprtScene` 是由 geometry/scene instance 组成的顶层结构
- `hiprtFuncTable` 为自定义 intersection/filter 提供运行时数据入口
- `hiprtBuildTraceKernels*` 负责把你的 device 代码变成可 launch 的函数句柄

## 1. 创建上下文

当前主线后端是 CUDA-only。最小上下文创建代码可以写成：

```cpp
#include <cuda_runtime_api.h>
#include <hiprt/hiprt.h>

int device = 0;
cudaGetDevice(&device);

hiprtContextCreationInput ctxtInput{};
ctxtInput.device = device;
ctxtInput.deviceType = hiprtDeviceNVIDIA;

hiprtContext ctxt = nullptr;
hiprtCreateContext(HIPRT_API_VERSION, ctxtInput, ctxt);
hiprtSetLogLevel(ctxt, hiprtLogLevelError | hiprtLogLevelWarn);
```

这里有几个要点：

- 当前测试主路径显式设置的是 `device` 和 `deviceType`。
- `hiprtApiVersion` 必须传 `HIPRT_API_VERSION`，否则会得到 `hiprtErrorInvalidApiVersion`。
- 同一个 `hiprtContext` 上的并发调用需要由调用方自己做外部同步。

收尾时调用：

```cpp
hiprtDestroyContext(ctxt);
```

## 2. 构建 geometry

### 2.1 准备几何输入

当前最常见的是三角形网格：

```cpp
hiprtTriangleMeshPrimitive mesh{};
mesh.vertices = d_vertices;
mesh.vertexCount = vertexCount;
mesh.vertexStride = sizeof(float3);
mesh.triangleIndices = d_indices;
mesh.triangleCount = triangleCount;
mesh.triangleStride = sizeof(uint3);

constexpr uint32_t GeomType = 2;

hiprtGeometryBuildInput geomInput{};
geomInput.type = hiprtPrimitiveTypeTriangleMesh;
geomInput.primitive.triangleMesh = mesh;
geomInput.geomType = GeomType;
```

其中：

- `vertices`、`triangleIndices` 都是 device 指针。
- `geomType` 不是几何类型枚举，而是给 custom func table 用的槽位编号。
- 如果后续完全不用 custom intersection/filter，`geomType` 可以保持默认值。

### 2.2 查询临时空间并 build

`create` 和 `build` 是两步：

```cpp
hiprtBuildOptions buildOptions{};
buildOptions.buildFlags = hiprtBuildFlagBitPreferFastBuild;

size_t geomTempSize = 0;
hiprtGetGeometryBuildTemporaryBufferSize(ctxt, geomInput, buildOptions, geomTempSize);

void* d_geomTemp = nullptr;
cudaMalloc(&d_geomTemp, geomTempSize);

hiprtGeometry geom = nullptr;
hiprtCreateGeometry(ctxt, geomInput, buildOptions, geom);
hiprtBuildGeometry(
    ctxt,
    hiprtBuildOperationBuild,
    geomInput,
    buildOptions,
    d_geomTemp,
    nullptr,
    geom);
```

使用习惯上要注意：

- `hiprtCreateGeometry(...)` 分配 geometry 本体。
- `hiprtBuildGeometry(...)` 真正填充 BVH/加速结构内容。
- `hiprtBuildOperationUpdate` 只适用于拓扑不变的更新场景。
- `temporaryBuffer` 由调用方自己分配和释放。

收尾：

```cpp
hiprtDestroyGeometry(ctxt, geom);
cudaFree(d_geomTemp);
```

## 3. 构建 scene

`scene` 是 geometry 或子 scene 的实例集合。最小单实例流程如下：

```cpp
hiprtInstance instance{};
instance.type = hiprtInstanceTypeGeometry;
instance.geometry = geom;

hiprtInstance* d_instances = nullptr;
cudaMalloc(&d_instances, sizeof(hiprtInstance));
cudaMemcpy(d_instances, &instance, sizeof(instance), cudaMemcpyHostToDevice);

hiprtFrameSRT frame{};
frame.translation = {0.0f, 0.0f, 0.0f};
frame.scale = {1.0f, 1.0f, 1.0f};
frame.rotation = {0.0f, 0.0f, 1.0f, 0.0f};

hiprtFrameSRT* d_frames = nullptr;
cudaMalloc(&d_frames, sizeof(hiprtFrameSRT));
cudaMemcpy(d_frames, &frame, sizeof(frame), cudaMemcpyHostToDevice);

hiprtSceneBuildInput sceneInput{};
sceneInput.instances = d_instances;
sceneInput.instanceFrames = d_frames;
sceneInput.instanceCount = 1;
sceneInput.frameCount = 1;
sceneInput.frameType = hiprtFrameTypeSRT;
```

然后查询 scene build 临时空间并构建：

```cpp
size_t sceneTempSize = 0;
hiprtGetSceneBuildTemporaryBufferSize(ctxt, sceneInput, buildOptions, sceneTempSize);

void* d_sceneTemp = nullptr;
cudaMalloc(&d_sceneTemp, sceneTempSize);

hiprtScene scene = nullptr;
hiprtCreateScene(ctxt, sceneInput, buildOptions, scene);
hiprtBuildScene(
    ctxt,
    hiprtBuildOperationBuild,
    sceneInput,
    buildOptions,
    d_sceneTemp,
    nullptr,
    scene);
```

几个关键点：

- `instances`、`instanceFrames`、`instanceMasks` 都是 device 指针。
- `instanceTransformHeaders == nullptr` 时，默认按每个 instance 一个 frame 解释。
- 如果要做 motion blur 或复杂 frame 布局，就要同时正确填写 `instanceTransformHeaders` 和 `frameCount`。

收尾：

```cpp
hiprtDestroyScene(ctxt, scene);
cudaFree(d_sceneTemp);
cudaFree(d_frames);
cudaFree(d_instances);
```

## 4. 使用 func table

`func table` 解决的是两件事：

1. build trace kernel 时，告诉 HIPRT 某个 `geomType/rayType` 使用哪个自定义 symbol
2. runtime 时，为这些自定义函数提供数据指针

这两层不要混淆：

- `hiprtFuncNameSet`：编译期的“函数名映射”
- `hiprtFuncDataSet`：运行期的“函数数据指针”

### 4.1 传入函数名映射

如果你的 device 代码里定义了自定义 filter/intersection，例如：

```cpp
HIPRT_DEVICE bool cutoutFilter(...);
HIPRT_DEVICE bool intersectSphere(...);
```

那么构建 trace kernel 前，需要先按 `geomType/rayType` 排好名字表：

```cpp
hiprtFuncNameSet funcNameSets[4]{};
funcNameSets[3].filterFuncName = "cutoutFilter";
```

这里的下标必须和前面 `geomInput.geomType = 3` 对齐。

### 4.2 传入运行时数据

如果你的自定义函数需要额外 device 数据，可通过 `hiprtFuncDataSet` 传进去：

```cpp
hiprtFuncTable funcTable = nullptr;
hiprtCreateFuncTable(ctxt, 4, 1, funcTable);

hiprtFuncDataSet funcData{};
funcData.intersectFuncData = d_spheres;
funcData.filterFuncData = d_filterData;

hiprtSetFuncTable(ctxt, funcTable, 3, 0, funcData);
```

如果自定义函数不需要额外数据，像很多测试那样传空的 `hiprtFuncDataSet{}` 即可。

收尾：

```cpp
hiprtDestroyFuncTable(ctxt, funcTable);
```

## 5. 构建 trace kernel 的三条路径

当前主线有三条 public API。

### 5.1 `hiprtBuildTraceKernels(...)`

这是最直接的 source-based 路径。你传入源代码字符串、头文件内容和编译选项，HIPRT 返回可 launch 的函数句柄。

最简调用形式：

```cpp
const char* funcName = "TraceKernel";
hiprtApiFunction function = nullptr;

hiprtBuildTraceKernels(
    ctxt,
    1,
    &funcName,
    source.c_str(),
    "runtime_bitcode_test.cu",
    numHeaders,
    headers,
    includeNames,
    compileOptionCount,
    compileOptions,
    numGeomTypes,
    numRayTypes,
    funcNameSets,
    &function,
    nullptr,
    false);
```

建议：

- 调试 runtime JIT 或 trace-kernel 行为时，优先把最后一个 `cache` 参数设成 `false`。
- 如果必须开 cache，至少结合 `hiprtSetCacheDirPath(...)` 切到新的临时目录，避免旧缓存干扰结论。

### 5.2 `hiprtBuildTraceKernelsFromBitcode(...)`

这条路径适合“你已经有可重定位 PTX/CUBIN 二进制”的场景。当前实现会把用户二进制与 `hiprt*_nv_lib.fatbin` 链接起来。

```cpp
hiprtBuildTraceKernelsFromBitcode(
    ctxt,
    1,
    &funcName,
    "runtime_bitcode_test.cu",
    bitcodeBinary.data(),
    bitcodeBinary.size(),
    numGeomTypes,
    numRayTypes,
    funcNameSets,
    &function,
    false);
```

适用场景：

- 原生 CUDA 环境下已经能稳定拿到 PTX/CUBIN
- 想把用户代码编译阶段和 HIPRT 链接阶段拆开

当前 `MACA + cu-bridge` 下，这条 API 本身已恢复，但 runtime 用户源码即时编译链仍不是首选验证路径；更稳妥的方式仍然是 precompiled / linked-bundle。

### 5.3 `hiprtBuildTraceKernelsFromLinkedBundle(...)`

这条路径直接接收已经预链接好的可加载模块镜像，例如 `mxcc --maca-link -fatbin` 产物。

```cpp
hiprtBuildTraceKernelsFromLinkedBundle(
    ctxt,
    1,
    &funcName,
    "runtime_bitcode_test.cu",
    bundleBinary.data(),
    bundleBinary.size(),
    &function,
    nullptr,
    false);
```

这条 API 的定位很明确：

- 绕过 runtime `cuLinkAddData`
- 直接消费已经准备好的 bundle
- 是当前 `MACA + cu-bridge` 路径更值得优先考虑的 public API

## 6. 一个最小的主机侧组合示例

把前面的核心步骤串起来，主机侧代码通常会长这样：

```cpp
hiprtContext ctxt = nullptr;
hiprtGeometry geom = nullptr;
hiprtScene scene = nullptr;
hiprtFuncTable funcTable = nullptr;
hiprtApiFunction function = nullptr;

hiprtCreateContext(HIPRT_API_VERSION, ctxtInput, ctxt);
hiprtSetLogLevel(ctxt, hiprtLogLevelError | hiprtLogLevelWarn);

hiprtCreateGeometry(ctxt, geomInput, buildOptions, geom);
hiprtBuildGeometry(ctxt, hiprtBuildOperationBuild, geomInput, buildOptions, d_geomTemp, nullptr, geom);

hiprtCreateScene(ctxt, sceneInput, buildOptions, scene);
hiprtBuildScene(ctxt, hiprtBuildOperationBuild, sceneInput, buildOptions, d_sceneTemp, nullptr, scene);

hiprtCreateFuncTable(ctxt, numGeomTypes, numRayTypes, funcTable);
hiprtSetFuncTable(ctxt, funcTable, geomType, rayType, funcDataSet);

hiprtBuildTraceKernels(
    ctxt,
    1,
    &funcName,
    source.c_str(),
    moduleName,
    numHeaders,
    headers,
    includeNames,
    compileOptionCount,
    compileOptions,
    numGeomTypes,
    numRayTypes,
    funcNameSets,
    &function,
    nullptr,
    false);

// 把 function 转成 CUDA driver function 后 launch
// kernel 参数里通常会传 scene/geom、输出 buffer、funcTable 等

hiprtDestroyFuncTable(ctxt, funcTable);
hiprtDestroyScene(ctxt, scene);
hiprtDestroyGeometry(ctxt, geom);
hiprtDestroyContext(ctxt);
```

## 7. 常见易错点

- `hiprtCreate*` 不等于已经完成 build，`hiprtBuild*` 仍然必须调用。
- `temporaryBuffer` 由调用方负责分配和释放，HIPRT 不接管这块内存。
- `geomType`、`numGeomTypes`、`funcNameSets`、`hiprtSetFuncTable(...)` 的槽位编号必须一致。
- `instanceFrames`、`instanceTransformHeaders`、`frameCount` 三者关系很容易配错，尤其是 motion blur 或多 frame 场景。
- 验证 trace-kernel / JIT 问题时，不要复用历史 `scripts/cache/`。
- 当前阶段要先确保纯 CUDA 基线跑通，再去看 `MACA + cu-bridge` 的兼容或离线路径。

## 8. 还可以继续看的 API

如果你已经把主流程跑通，可以继续看这些扩展 API：

- `hiprtCompactGeometry(...)` / `hiprtCompactScene(...)`
- `hiprtExportGeometryAabb(...)` / `hiprtExportSceneAabb(...)`
- `hiprtSaveGeometry(...)` / `hiprtLoadGeometry(...)`
- `hiprtSaveScene(...)` / `hiprtLoadScene(...)`
- `hiprtCreateGlobalStackBuffer(...)`
- 批量版本：
  - `hiprtCreateGeometries(...)`
  - `hiprtBuildGeometries(...)`
  - `hiprtCreateScenes(...)`
  - `hiprtBuildScenes(...)`

## 9. device-side 入口位置

如果你接下来要看 kernel 内部怎么做 traversal，可以直接从这些文件入手：

- `test/kernels/HiprtTestKernel.h`
- `test/bitcodes/custom_func_table.cpp`

里面已经覆盖了这些典型入口：

- `hiprtGeomTraversalClosest`
- `hiprtGeomTraversalAnyHit`
- `hiprtSceneTraversalAnyHit`
- `hiprtSceneTraversalClosestCustomStack`
