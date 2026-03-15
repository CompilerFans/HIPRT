# Repository Guidelines

## 沟通与输出
- 与本仓库协作时，说明、提交说明、评审结论、操作记录默认使用中文。
- 做基于历史 patch 的升级时，必须先检查 patch 的意图与当前主线代码是否一致，不能机械套 patch。

## 构建约定
- 默认优先使用 `CMake + Ninja`。
- 默认优先开启 `ccache`，减少反复回归时的编译成本。
- Linux 下默认优先尝试 `mold` 链接；若编译器不支持 `-fuse-ld=mold`，则改用 `-B<mold目录>` 方式接入。
- 推荐直接使用仓库脚本：`./scripts/build.sh`。

## 功能验证
- 功能正确性验证时，不能依赖历史 HIPRT JIT cache 结果。
- 回归前优先清理测试过程中生成的 `scripts/cache/`，避免旧缓存掩盖真实问题。
- 如果在验证 trace kernel/JIT 相关行为，优先避免复用缓存；必要时应将构建调用里的 `cache` 参数设为 `false`，或切换到新的临时 cache 目录后再验证。
- 默认先跑非性能功能测试：`cd scripts && ./unittest.sh`。

## 本次主线迁移阶段要求
- 当前阶段目标是先完成纯 CUDA 适配，并保持当前官方 `main` 上的功能测试可通过。
- 在进入后续 MACA 适配前，必须确保纯 CUDA 基线可编译、可运行、可复现。
