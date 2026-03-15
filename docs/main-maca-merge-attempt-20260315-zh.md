# main 分支 MACA 合并尝试记录（2026-03-15）

## 本轮目标

基于当前 `main`：

- 保留已经完成的 CUDA-only 主线收敛
- 将历史 `0002~0005` 合并后的 MACA 适配 patch 直接回放到当前主线
- 不再保留“分阶段小 patch”作为执行单元

## 已完成内容

### 1. patch 形态重组

已生成新的合并 patch 目录：

- `patches/maca-sync-merged-20260315/0001-cuda-only.patch`
- `patches/maca-sync-merged-20260315/0002-maca-adaptation.patch`

含义改成两层：

1. `0001`：CUDA-only 基线
2. `0002`：完整 MACA 适配

### 2. merged MACA patch 已落到当前源码树

当前源码树已经带入了 merged patch 的主体内容，包括：

- wave64 / lane-mask 语义收敛
- runtime kernel cache 控制
- scene / traversal 诊断测试
- batch geometry 诊断测试
- MACA 状态与分析文档

### 3. MACA 构建链已切到新默认入口

已完成以下脚本和入口调整：

- `scripts/build_maca_cucc.sh`
  - 默认 `cmake_maca -G Ninja`
  - 默认 `ninja_maca`
  - 默认使用新构建目录 `build_maca_ninja`
- `scripts/build_and_test_maca_cucc.sh`
  - 新增一键“编译并执行”入口
- `scripts/unittest_maca_cucc.sh`
  - 改为复用 `build_and_test_maca_cucc.sh`

构建链实测确认：

- generator：`Ninja`
- compiler launcher：`ccache`
- linker：`mold`
- MACA/cu-bridge configure：`cmake_maca`
- MACA/cu-bridge build：`ninja_maca`

### 4. 当前主库与 unittest 可重新编译

在 `build_maca_ninja` 下：

- `libhiprt0300164.so`
- `unittest64`

都已经可以重新编译完成。

## 当前阻塞

当前还没有恢复到“最小功能 case 可运行”，阻塞点在 **runtime/JIT 编译与链接链**，不是常规 host build。

### 阻塞 1：`Transform.h` 与 `hiprt_device_impl.h` 的接口模型不一致

merged patch 中的 `hiprt_device_impl.h` 使用了偏 `maca_dev` 的 transform helper 语义，但当前主线的 `Transform.h` 不是同一版实现。

本轮已经做了最小补齐：

- 为当前 `Transform.h` 补了 `SRTFrame` / `MatrixFrame` 的 helper
- 修正了 `hiprt_device_impl.h` 中 scene transform 对 `hiprtScene` 的调用

结果：

- 主库重新编译已恢复
- 但这还不是最终收敛点，只是把 host 编译错误先压下去

### 阻塞 2：MACA 在线编译器对 trace-kernel 入口发现方式有兼容问题

最小用例 `hiprtTest.MinimumCornellBox` 仍然失败，失败点已经稳定收敛到 runtime 编译/链接阶段。

观察到的问题分两层：

1. `__mcrtc_* = CornellBoxKernel` 的自动包装在 cu-bridge/MACA 下对入口函数发现存在兼容问题
2. 即使绕过 `nvrtcAddNameExpression`，JIT 仍然在链接阶段报未定义符号

当前最小稳定报错包括：

- `duplicityFilter(...)` 未定义
- `hiprtPointWorldToObject(...)` 未定义

这说明当前 MACA JIT 链接问题不只是“入口函数声明顺序”，而是：

- 部分 device helper / filter symbol 没有被正确纳入在线编译模块
- 或者当前 `buildTraceKernels` / `Compiler` 对 cu-bridge/MACA 的 runtime compile 组织方式与 NVIDIA/NVRTC 还不一致

## 本轮尝试过但尚未完全解决的项

### 已尝试

- 在 `HiprtTestKernel.h` 顶部补 kernel 前置声明
- 在 `test/hiprtTest.cpp` 中为目标 trace kernel 预提取声明并 prepend 到 source
- 在 `Compiler.cpp` 中为 cu-bridge/MACA 绕过 `nvrtcAddNameExpression`

### 结果

- 这些改动都推动问题继续前移
- 但仍未把 `MinimumCornellBox` 跑通
- 当前认为真正剩余问题已经进入 **MACA runtime/JIT 模块组织与符号可见性** 层

## 当前结论

这轮工作有明确阶段成果，但还不适合继续在当前 `main` 上深挖：

1. merged patch 在当前主线上已经完成一次真实落地
2. MACA 构建脚本入口已经整理完毕
3. host 编译链已经恢复
4. 剩余问题集中在 runtime/JIT 组织层，继续深挖的性价比不如回到 `maca_dev` 这条已验证更多的旧适配分支

## 建议的后续路线

建议切回 `maca_dev` 后：

1. 先恢复到旧版本上已验证可运行的 MACA 适配基线
2. 重新从 `maca_dev` 验证当前关键测试通过状态
3. 再把已经确认必要的 patch 逐步重新抽回主线
4. 对 runtime/JIT 相关改动单独做最小回放，不把整包 merged patch 一次性压到主线

## 备注

本文件用于记录：

- 本轮 merged patch 合入尝试做到哪里
- 哪些部分已经有效
- 当前真正阻塞在什么层

后续如果重新进入 `main` 做 MACA 合入，可直接以此文件作为回顾入口。
