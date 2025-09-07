# HIPRT技术特性详解

## BVH构建技术

### 1. 线性BVH (LBVH) - 极速构建

#### 核心算法
基于Morton码的并行构建算法，适合GPU大规模并行处理。

**Morton码计算**：
```cpp
// 将3D坐标映射到1D Morton码
uint32_t mortonCode(float3 v) {
    // 量化到[0, 1023]
    uint32_t x = clamp(v.x * 1024.0f, 0.0f, 1023.0f);
    uint32_t y = clamp(v.y * 1024.0f, 0.0f, 1023.0f);
    uint32_t z = clamp(v.z * 1024.0f, 0.0f, 1023.0f);
    
    // 交织位产生Morton码
    return expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2);
}
```

**构建步骤**：
1. 计算所有图元的Morton码
2. 基数排序(RadixSort)
3. 自底向上构建层次结构
4. 计算节点包围盒

**性能特征**：
- 构建时间：O(n log n)
- GPU并行度：极高
- 内存占用：2n-1个节点

### 2. PLOC构建器 - 平衡选择

#### 核心思想
Parallel Locally-Ordered Clustering，并行局部有序聚类。

**算法流程**：
```
初始化：每个图元为一个簇
重复：
  1. 寻找最近邻簇对
  2. 并行合并簇对
  3. 更新簇包围盒
直到：只剩一个根簇
```

**优化策略**：
- 使用最近邻表加速查找
- 批量并行合并
- 自适应阈值控制

**质量vs速度权衡**：
- 构建速度：中等
- 遍历性能：良好
- SAH成本：接近最优

### 3. 空间分割BVH (SBVH) - 最高质量

#### SAH驱动的空间分割
表面积启发式(Surface Area Heuristic)指导分割决策。

```cpp
float sahCost(const AABB& left, const AABB& right, 
              int leftCount, int rightCount) {
    float leftArea = left.surfaceArea();
    float rightArea = right.surfaceArea();
    float parentArea = merge(left, right).surfaceArea();
    
    return (leftArea * leftCount + rightArea * rightCount) / parentArea;
}
```

#### 空间分割策略
当物体重叠严重时，进行空间分割：
```cpp
struct SplitCandidate {
    float position;    // 分割位置
    int axis;         // 分割轴
    float cost;       // SAH成本
};
```

**特点**：
- 支持重复引用
- 最优的遍历性能
- 构建时间较长

## 光线遍历优化

### 1. 压缩BVH节点

```cpp
// 标准节点：64字节
struct BVHNode {
    float3 min;
    float3 max;
    uint32_t leftChild;
    uint32_t rightChild;
};

// 压缩节点：32字节
struct CompressedBVHNode {
    // 量化的包围盒
    uint16_t min[3];
    uint16_t max[3];
    // 子节点偏移
    uint32_t children;
};
```

### 2. 遍历算法优化

#### 无栈遍历
使用持久化遍历状态，避免栈内存开销：
```cpp
__device__ void traverse(Ray ray, BVH bvh) {
    uint32_t node = 0;
    uint32_t bitstack = 0;
    
    while (node != INVALID_NODE) {
        if (isLeaf(node)) {
            testPrimitives(node, ray);
            node = popStack(bitstack);
        } else {
            uint2 children = getChildren(node);
            uint2 hits = intersectChildren(ray, children);
            
            if (hits.x && hits.y) {
                pushStack(bitstack, children.y);
                node = children.x;
            } else if (hits.x) {
                node = children.x;
            } else if (hits.y) {
                node = children.y;
            } else {
                node = popStack(bitstack);
            }
        }
    }
}
```

#### SIMD优化
利用GPU的向量指令同时测试多个包围盒：
```cpp
__device__ bool4 intersectAABB4(
    float4 rayOrig[3], 
    float4 rayInvDir[3],
    float4 bboxMin[3], 
    float4 bboxMax[3]
) {
    // 同时测试4个包围盒
    float4 tMin = make_float4(0.0f);
    float4 tMax = make_float4(FLT_MAX);
    
    for (int i = 0; i < 3; ++i) {
        float4 t1 = (bboxMin[i] - rayOrig[i]) * rayInvDir[i];
        float4 t2 = (bboxMax[i] - rayOrig[i]) * rayInvDir[i];
        
        tMin = fmax(tMin, fmin(t1, t2));
        tMax = fmin(tMax, fmax(t1, t2));
    }
    
    return tMin <= tMax;
}
```

### 3. 光线包遍历
同时遍历多条相干光线：
```cpp
struct RayPacket {
    float4 origin[3];     // 4条光线的起点
    float4 direction[3];  // 4条光线的方向
    float4 tMin, tMax;    // 光线范围
};
```

## 实例化技术

### 多级实例化
支持实例的嵌套，实现复杂场景的高效表示：

```cpp
struct Instance {
    float4x4 transform;      // 实例变换矩阵
    float4x4 invTransform;   // 逆变换矩阵
    uint32_t geometryID;     // 几何体ID
    uint32_t mask;          // 可见性掩码
};

// 光线变换到实例空间
__device__ Ray transformRay(Ray worldRay, Instance instance) {
    Ray localRay;
    localRay.origin = transformPoint(worldRay.origin, instance.invTransform);
    localRay.direction = transformVector(worldRay.direction, instance.invTransform);
    return localRay;
}
```

### 实例化优化
- **变换缓存**: 缓存常用变换矩阵
- **批处理**: 相同几何体的实例批量处理
- **LOD支持**: 不同细节级别的实例

## 自定义几何体系统

### 函数表机制
允许用户注册自定义的相交函数：

```cpp
// 用户定义的相交函数
__device__ bool intersectSphere(
    const Ray& ray,
    const SphereData& sphere,
    float& t,
    float2& uv,
    float3& normal
) {
    float3 oc = ray.origin - sphere.center;
    float b = dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - c;
    
    if (discriminant < 0) return false;
    
    float sqrtD = sqrt(discriminant);
    t = -b - sqrtD;
    
    if (t < ray.tMin || t > ray.tMax) {
        t = -b + sqrtD;
        if (t < ray.tMin || t > ray.tMax) return false;
    }
    
    float3 hitPoint = ray.origin + t * ray.direction;
    normal = normalize(hitPoint - sphere.center);
    
    // 计算UV坐标
    float theta = acos(-normal.y);
    float phi = atan2(-normal.z, normal.x) + M_PI;
    uv.x = phi / (2 * M_PI);
    uv.y = theta / M_PI;
    
    return true;
}
```

### AABB列表支持
用于程序化几何体和体积渲染：

```cpp
struct AABBList {
    AABB* boxes;
    uint32_t* primitiveIDs;
    uint32_t count;
    
    // 自定义AABB内容的相交测试
    hiprtFuncTable customIntersect;
};
```

## 内存管理技术

### MemoryArena实现
高效的线性内存分配器：

```cpp
class MemoryArena {
private:
    struct Block {
        uint8_t* data;
        size_t size;
        size_t used;
    };
    
    std::vector<Block> m_blocks;
    size_t m_defaultBlockSize;
    
public:
    void* allocate(size_t size, size_t alignment = 16) {
        size_t alignedSize = align(size, alignment);
        
        // 从当前块分配
        if (m_blocks.back().used + alignedSize <= m_blocks.back().size) {
            void* ptr = m_blocks.back().data + m_blocks.back().used;
            m_blocks.back().used += alignedSize;
            return ptr;
        }
        
        // 分配新块
        allocateNewBlock(std::max(alignedSize, m_defaultBlockSize));
        return allocate(size, alignment);
    }
    
    void reset() {
        for (auto& block : m_blocks) {
            block.used = 0;
        }
    }
};
```

### GPU内存池
减少内存分配开销：

```cpp
class DeviceMemoryPool {
    struct Allocation {
        void* ptr;
        size_t size;
        bool inUse;
    };
    
    std::vector<Allocation> m_allocations;
    
public:
    void* allocate(size_t size) {
        // 查找可重用的分配
        for (auto& alloc : m_allocations) {
            if (!alloc.inUse && alloc.size >= size) {
                alloc.inUse = true;
                return alloc.ptr;
            }
        }
        
        // 新分配
        void* ptr;
        oroMalloc(&ptr, size);
        m_allocations.push_back({ptr, size, true});
        return ptr;
    }
};
```

## 编译器优化

### 内核特化
根据场景特征生成优化的内核：

```cpp
struct KernelConfig {
    bool hasTriangles;
    bool hasCustomGeometry;
    bool hasInstances;
    bool hasMotionBlur;
    int maxTraceDepth;
};

std::string generateKernelCode(const KernelConfig& config) {
    std::string code = baseKernelCode;
    
    if (!config.hasCustomGeometry) {
        code += "#define SKIP_CUSTOM_GEOMETRY\n";
    }
    
    if (config.maxTraceDepth == 1) {
        code += "#define SINGLE_RAY_TYPE\n";
    }
    
    return code;
}
```

### 编译缓存
避免重复编译：

```cpp
class CompilerCache {
    std::string getCacheKey(const std::string& source, 
                           const CompilerOptions& options) {
        // 基于源码和选项生成唯一键
        size_t hash = std::hash<std::string>{}(source);
        hash ^= std::hash<std::string>{}(options.toString());
        return std::to_string(hash) + ".cache";
    }
    
    bool loadFromCache(const std::string& key, CompiledKernel& kernel) {
        std::filesystem::path cachePath = m_cacheDir / key;
        if (std::filesystem::exists(cachePath)) {
            // 加载缓存的二进制
            return kernel.load(cachePath);
        }
        return false;
    }
};
```

## 调试和性能分析

### GPU性能计时
```cpp
class GpuTimer {
    oroEvent_t m_start, m_stop;
    
public:
    void start(oroStream_t stream) {
        oroEventCreate(&m_start);
        oroEventRecord(m_start, stream);
    }
    
    float stop(oroStream_t stream) {
        oroEventCreate(&m_stop);
        oroEventRecord(m_stop, stream);
        oroEventSynchronize(m_stop);
        
        float milliseconds = 0;
        oroEventElapsedTime(&milliseconds, m_start, m_stop);
        return milliseconds;
    }
};
```

### BVH质量分析
```cpp
struct BVHStats {
    float sahCost;           // SAH成本
    int maxDepth;           // 最大深度
    float avgLeafPrims;     // 平均叶节点图元数
    float emptySpaceRatio;  // 空白空间比例
    
    void analyze(const BVH& bvh) {
        // 递归分析BVH结构
        analyzeNode(bvh.getRoot(), 0);
    }
};
```