# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

HIPRT是AMD开发的高性能光线追踪库，支持AMD和NVIDIA GPU。采用MIT许可，由Advanced Rendering Research Group维护。

## 常用构建命令

### CMake构建（推荐）
```bash
# Linux常规构建
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DHIP_PATH=/opt/rocm -S .. -B .
cmake --build . --config Release -j$(nproc)

# Windows构建  
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DHIP_PATH="C:\Program Files\AMD\ROCm\6.2" -S .. -B .
# 用Visual Studio打开build\hiprt.sln

# 启用Bitcode预编译（提升运行时性能）
cmake -DCMAKE_BUILD_TYPE=Release -DBITCODE=ON -DPRECOMPILE=ON -S .. -B .
```

### 运行测试
```bash
# Linux - 功能测试（从scripts目录运行）
cd scripts && ./unittest.sh

# Linux - 性能测试
cd scripts && ./unittest_perf.sh

# 运行特定测试
../dist/bin/Release/unittest64 --gtest_filter="hiprt.BvhImport" --width=512 --height=512

# 列出所有可用测试
../dist/bin/Release/unittest64 --gtest_list_tests
```

### 内核编译（开发者）
```bash
# 编译内核到bitcode和fatbinary
cd scripts/bitcodes
python compile.py

# 预编译bitcode（需要在CMake中启用-DPRECOMPILE=ON）
python precompile_bitcode.py
```

## 项目架构

### 目录结构
```
hiprt/
├── hiprt/              # 公共API头文件
│   ├── hiprt.h         # 主机端C++ API
│   ├── hiprt_device.h  # 设备端(GPU)API
│   ├── hiprt_types.h   # 数据类型定义
│   └── hiprtew.h       # 动态加载包装器
├── hiprt/impl/         # 核心实现
│   ├── Context.*       # 上下文管理（核心组件）
│   ├── Compiler.*      # 运行时内核编译
│   ├── Kernel.*        # 内核管理
│   ├── LbvhBuilder.*   # Linear BVH构建器（快速）
│   ├── PlocBuilder.*   # PLOC BVH构建器（平衡）
│   ├── SbvhBuilder.*   # Spatial BVH构建器（高质量）
│   ├── TriangleMesh.*  # 三角形网格处理
│   └── MemoryArena.*   # 内存分配器
├── contrib/Orochi/     # HIP/CUDA统一抽象层
└── test/              # 测试套件
```

### 核心架构设计

#### 1. 分层API设计
- **主机端API** (`hiprt.h`): 创建上下文、构建BVH、管理几何体
- **设备端API** (`hiprt_device.h`): GPU内核中的光线遍历和相交测试
- **统一硬件抽象**: 通过Orochi支持AMD(HIP)和NVIDIA(CUDA) GPU

#### 2. Context核心模式
```cpp
hiprtContext          // 全局上下文，管理所有资源
├── hiprtDevice       // GPU设备抽象
├── hiprtCompiler     // 内核编译器
├── hiprtFuncTable    // 函数表管理
└── MemoryArena       // 临时内存分配
```

#### 3. BVH构建流程
1. **算法选择**: 根据`hiprtBuildOptions::buildFlags`选择构建器
   - `hiprtBuildFlagBitPreferFastBuild`: 使用LBVH（线性构建）
   - `hiprtBuildFlagBitPreferHighQualityBuild`: 使用SBVH（空间分割）
   - 默认: 使用PLOC（平衡性能）

2. **构建管线**:
   ```
   几何数据 → BvhBuilder → BVH节点 → 压缩存储
   ```

#### 4. 内核编译系统
- **运行时编译**: 根据场景配置动态编译优化的内核
- **缓存机制**: 编译后的内核缓存到文件系统
- **Bitcode支持**: 预编译内核以减少运行时开销
- **加密保护**: 支持内核源码加密（可选）

#### 5. 内存管理策略
- **MemoryArena**: 线性分配器用于临时数据，避免频繁malloc/free
- **缓冲区复用**: BVH构建过程中的临时缓冲区可复用
- **设备内存管理**: 通过Orochi统一管理HIP/CUDA内存

### 关键设计模式

#### 1. 统一GPU抽象
```cpp
// Orochi统一了HIP和CUDA API
oroMalloc() → hipMalloc() 或 cudaMalloc()
oroMemcpy() → hipMemcpy() 或 cudaMemcpy()
```

#### 2. 模板化BVH构建
```cpp
template<typename BvhBuilder>
void buildBvhImpl(Context* context, ...) {
    BvhBuilder builder;
    builder.build(...);
}
```

#### 3. 函数表机制
- 自定义相交函数通过函数表注册
- 支持用户扩展几何体类型
- 设备端通过索引调用相应函数

### 开发要点

#### 编码规范（严格遵守）
- 驼峰命名：`nodeCount`（变量）、`LogSize`（常量）
- 成员变量前缀：`m_memberVar`
- 优先使用引用而非指针
- 使用`std::string`而非`char*`
- 使用`std::filesystem::path`处理路径
- 使用`override`标记虚函数重写
- 遵循Rule of Five

#### 扩展几何体类型
1. 在`hiprt_types.h`添加新的`hiprtGeometryType`
2. 实现相应的构建器（参考`TriangleMesh`）
3. 编写设备端相交函数
4. 注册到函数表

#### 调试技巧
- 设置`HIPRT_CACHE_PATH`环境变量指定内核缓存位置
- 使用`hiprtContextSetLogLevel`启用详细日志
- GPU调试：使用`rocgdb`(AMD)或`cuda-gdb`(NVIDIA)

### 版本管理
- 版本号存储在`version.txt`：主版本.次版本.补丁版本
- API变更必须更新次版本号
- 主版本和次版本相同则二进制兼容
- 每个master提交必须有唯一补丁版本

### 性能优化建议
1. 使用Bitcode预编译避免运行时编译开销
2. 根据场景特征选择合适的BVH构建算法
3. 复用`hiprtContext`避免重复初始化
4. 批量构建多个几何体以提高吞吐量