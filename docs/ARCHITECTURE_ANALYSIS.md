# HIPRT架构深度分析

## 整体架构概览

HIPRT采用分层架构设计，清晰地分离了不同层次的功能职责：

```
┌─────────────────────────────────────────────┐
│          应用层 (User Application)           │
├─────────────────────────────────────────────┤
│         HIPRT API层 (hiprt.h)               │
├─────────────────────────────────────────────┤
│          核心实现层 (hiprt/impl/)            │
├─────────────────────────────────────────────┤
│      硬件抽象层 (Orochi)                    │
├─────────────────────────────────────────────┤
│     GPU驱动层 (HIP/CUDA)                    │
└─────────────────────────────────────────────┘
```

## 核心组件详解

### 1. Context管理系统

Context是HIPRT的核心管理组件，负责整个系统的生命周期管理：

```cpp
class Context {
    hiprtDevice m_device;           // GPU设备管理
    hiprtCompiler m_compiler;       // 内核编译器
    MemoryArena m_arena;           // 内存分配器
    FuncTableManager m_funcTable;   // 函数表管理
    KernelCache m_cache;           // 内核缓存
};
```

#### 设计特点：
- **单例模式变体**: 每个hiprtContext对应一个Context实例
- **资源集中管理**: 所有GPU资源通过Context统一分配和释放
- **延迟初始化**: 编译器等组件按需创建

### 2. 内核编译系统

HIPRT的内核编译系统是其独特之处，支持运行时编译优化：

#### 编译流程：
```
源代码(.cpp) → [预处理] → [HIP编译] → Bitcode/Fatbinary → [缓存] → GPU执行
```

#### 关键组件：
- **Compiler类**: 封装hipcc编译过程
- **KernelCache**: 基于文件系统的编译结果缓存
- **Bitcode支持**: 预编译内核以减少运行时开销

#### 编译选项：
```cpp
struct CompilerOptions {
    bool useBitcode;        // 使用预编译bitcode
    bool enableEncryption;  // 内核加密
    bool enableCache;       // 启用缓存
    std::string cachePath;  // 缓存路径
};
```

### 3. BVH构建系统

HIPRT提供三种BVH构建算法，适应不同的性能需求：

#### LBVH (Linear BVH)
- **特点**: 构建速度最快
- **算法**: 基于Morton码的并行构建
- **适用场景**: 动态场景、实时应用

```cpp
class LbvhBuilder : public BvhBuilder {
    void build() {
        computeMortonCodes();
        sortPrimitives();
        buildHierarchy();
        computeAABBs();
    }
};
```

#### PLOC (Parallel Locally-Ordered Clustering)
- **特点**: 平衡构建速度和质量
- **算法**: 自底向上的并行聚类
- **适用场景**: 一般渲染应用

```cpp
class PlocBuilder : public BvhBuilder {
    void build() {
        createClusters();
        mergeClustersPairwise();
        optimizeTopology();
    }
};
```

#### SBVH (Spatial BVH)
- **特点**: 最高质量，支持空间分割
- **算法**: SAH驱动的空间分割
- **适用场景**: 离线渲染、高质量需求

```cpp
class SbvhBuilder : public BvhBuilder {
    void build() {
        spatialSplit();
        optimizeSAH();
        compactNodes();
    }
};
```

### 4. 内存管理策略

#### MemoryArena
线性内存分配器，用于临时数据：
```cpp
class MemoryArena {
    uint8_t* m_data;
    size_t m_size;
    size_t m_offset;
    
    void* allocate(size_t size) {
        void* ptr = m_data + m_offset;
        m_offset += align(size);
        return ptr;
    }
    
    void reset() { m_offset = 0; }
};
```

#### 设备内存管理
通过Orochi统一接口管理GPU内存：
```cpp
// 分配设备内存
oroMalloc(&devicePtr, size);

// 数据传输
oroMemcpyHtoD(devicePtr, hostPtr, size);

// 释放内存
oroFree(devicePtr);
```

### 5. GPU设备抽象

通过Orochi实现的硬件抽象层，支持AMD和NVIDIA GPU：

```cpp
// Orochi自动检测并加载正确的GPU API
oroInitialize(oroApi::ORO_API_HIP);  // AMD GPU
oroInitialize(oroApi::ORO_API_CUDA); // NVIDIA GPU
```

## 数据流分析

### 构建阶段数据流
```
输入几何数据
    ↓
[顶点/索引缓冲区上传到GPU]
    ↓
[BVH构建器处理]
    ↓
[生成BVH节点数据]
    ↓
[压缩和优化]
    ↓
输出：hiprtGeometry对象
```

### 遍历阶段数据流
```
光线数据
    ↓
[上传到GPU]
    ↓
[内核执行光线遍历]
    ↓
[BVH节点遍历]
    ↓
[三角形相交测试]
    ↓
输出：相交结果
```

## 线程模型

### CPU端
- **主线程**: API调用、资源管理
- **编译线程**: 异步内核编译（可选）
- **工作线程**: BVH构建的CPU部分（如果有）

### GPU端
- **Warp/Wavefront**: 32/64线程的SIMD执行
- **Block/Workgroup**: 多个warp的协作单元
- **Grid**: 整个内核的执行网格

## 扩展机制

### 自定义几何体
```cpp
// 1. 定义几何类型
enum MyGeometryType {
    MyCustomType = hiprtGeometryTypeCount
};

// 2. 实现相交函数
__device__ bool intersectCustom(
    const Ray& ray,
    const CustomData& data,
    Hit& hit
);

// 3. 注册到函数表
hiprtSetFuncTable(context, 
    MyCustomType, 
    intersectCustom);
```

### 过滤函数
```cpp
// 自定义过滤逻辑
__device__ bool filterFunc(
    const Ray& ray,
    const Hit& hit,
    void* payload
) {
    // 实现自定义过滤
    return shouldAccept;
}
```

## 性能优化策略

### 1. 数据布局优化
- **AoS vs SoA**: 根据访问模式选择
- **对齐**: 确保数据结构对齐以优化内存访问
- **压缩**: BVH节点压缩减少内存带宽

### 2. 并行化策略
- **任务并行**: BVH构建的不同阶段
- **数据并行**: 光线遍历的SIMD执行
- **异步执行**: CPU/GPU重叠执行

### 3. 缓存优化
- **内核缓存**: 避免重复编译
- **BVH缓存**: 静态场景重用BVH
- **纹理缓存**: 利用GPU纹理缓存

## 错误处理机制

```cpp
// API层错误处理
hiprtError handleError(std::function<void()> func) {
    try {
        func();
        return hiprtSuccess;
    } catch (std::runtime_error& e) {
        logError(e.what());
        return hiprtErrorInternal;
    }
}

// 设备端错误传播
__device__ void reportError(ErrorCode code) {
    atomicExch(&g_errorCode, code);
}
```

## 调试支持

### 日志系统
```cpp
hiprtContextSetLogLevel(context, hiprtLogLevelDebug);
// 输出详细的构建和执行信息
```

### 性能分析
- GPU计时器集成
- 内存使用统计
- BVH质量指标

## 架构优势与局限

### 优势
1. **简洁性**: API简单，易于理解和使用
2. **灵活性**: 支持多种BVH算法和自定义扩展
3. **跨平台**: AMD/NVIDIA GPU统一支持
4. **开源**: 完全开放的实现

### 局限
1. **软件实现**: 目前不支持硬件RT加速
2. **功能范围**: 相比OptiX等功能较为基础
3. **生态系统**: 第三方集成和工具支持有限

## 未来架构演进方向

1. **硬件加速支持**: 集成AMD RDNA3的RT加速
2. **异构计算**: CPU+GPU协同光线追踪
3. **分布式渲染**: 多GPU扩展支持
4. **AI集成**: 深度学习加速的降噪和采样