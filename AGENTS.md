# Repository Guidelines

## 项目结构与模块组织
`hiprt/` 是核心库目录，公开头文件位于根下，主要实现位于 `hiprt/impl/`。`test/` 保存 GoogleTest 用例、`test/kernels/` 下的测试内核、`test/references/` 下的参考图，以及 `test/common/` 中的共享模型和辅助代码。`scripts/` 主要提供测试启动脚本。`contrib/` 是三方依赖代码，除非必须同步补丁，否则不要随意改动。构建产物默认输出到 `build/` 和 `dist/bin/`。

## 构建、测试与开发命令
首次拉取后先执行 `git submodule update --init --recursive`。本仓库当前仅支持 CMake。Linux 和 Windows 下都使用 `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`，然后执行 `cmake --build build --config Release`。如需调试构建，可改成 `-DCMAKE_BUILD_TYPE=Debug`。

在 `MACA + cu-bridge` 环境下，不要把当前 runtime compile 故障简单归因到 `Ninja/ninja_maca`。

目前已经完成的对比实验表明：

- `cmake_maca + Ninja + ninja_maca` 下，最小用例 `hiprtTest.MinimumCornellBox` 会稳定失败在 runtime compile 阶段，表现为 `CornellBoxKernel` 在 `__mcrtc_*` 自动包装中不可见。
- `禁用 mold`、`禁用 ccache` 都不会改变这一失败形态。
- `cmake_maca + make_maca` 路线曾出现过一次通过观测，但重复实验并不稳定；后续复验同样回到相同的 runtime compile 失败。

因此当前约束应理解为：

- `make_maca` 仍然值得作为对比路径保留
- 但不能把它当作已确认稳定的 workaround
- 当前真正优先级仍是排查 `buildTraceKernels()` / `Compiler.cpp` 与 cu-bridge/MACA runtime compile 的兼容性

## 代码风格与命名约定
项目使用 C++17，并遵循 `.clang-format`：制表符缩进，宽度 4，列宽上限 128，基于 LLVM 风格，左花括号单独成行。变量使用小驼峰，如 `nodeCount`；常量使用大驼峰，如 `LogSize`；非静态成员统一使用 `m_` 前缀。优先使用 `nullptr`、`override`、`std::optional`、`std::filesystem::path` 和 C++ 风格转换，避免 C 风格写法。

## 测试规范
测试二进制默认生成到 `dist/bin/Release/unittest64`。Linux 下常用脚本为 `cd scripts && ./unittest.sh`，性能场景使用 `cd scripts && ./unittest_perf.sh`。可通过 `--gtest_filter=` 精确筛选，例如 `../dist/bin/Release/unittest64 --gtest_filter=HiprtTests.*`。仓库没有强制覆盖率门槛，但新增功能应补充或扩展对应的 `HiprtTests`、`ObjTestCases` 或 `PerformanceTestCases`。性能测试依赖 Git LFS 资源。

## 提交与 Pull Request 规范
近期提交信息以简短祈使句为主，例如 `fix for test case`，也接受带模块前缀的形式，如 `hiprt: remove encryption support`。单次提交应聚焦单一主题，必要时在标题中标明影响模块。提交 PR 时请说明行为变化、列出已执行的构建与测试命令、注明使用的 CUDA 环境；如果修改了 `test/references/` 或性能路径，请附上图像对比或性能说明。

## 配置与版本说明
若修改公开 API，需要同步更新 `version.txt`：API 变化提升 minor，主线上的 patch 版本应保持唯一。
