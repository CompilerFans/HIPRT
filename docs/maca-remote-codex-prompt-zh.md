# MACA 远端 Codex 执行提示词

本文档用于在远端 MACA 机器上，直接给 Codex 提供足够完整的背景、目标和执行约束，避免重复沟通。

## 最终目标

基于当前已经完成的官方 `main` 纯 CUDA 基线，继续把 `maca_dev` 分支里 `patches/staged-upstream-sync-20260315/0002-maca-core.patch` 所代表的 **MACA core 适配** 收敛到最新官方主线，并在远端 MACA 环境完成构建和功能验证。

要求：

- 不要机械套 `0002-maca-core.patch`
- 必须逐项核对 patch 意图和当前官方 `main` 的真实代码语义
- 每做一批修改就做一轮构建和定向验证
- 优先保证正确性，再考虑恢复旧的快路径或进一步优化

## 当前本地基线

- 本地仓库基于官方最新 `main`
- 已完成第一阶段纯 CUDA 适配
- 当前基线提交：
  - `844cd2a` `docs: 增加MACA远端开发交接说明`
- 其中代码基线提交：
  - `6520392` `cuda: 基于0001阶段补丁完成纯CUDA适配`

本地已验证：

- 构建：`./scripts/build.sh`
- 测试：`cd scripts && ./unittest.sh`
- 结果：`39 / 39` 非性能功能测试通过

## 为什么要转到远端做

本地没有 MACA 工具链和运行环境，无法验证：

- `__MACACC__`
- wave64 / lane mask 语义
- cu-bridge / cucc / runtime compiler
- MACA 下的 scene traversal / transform / update 正确性

所以后续 `0002` 的工作必须放到远端 MACA 机器上做。

## 远端环境已确认的事实

远端机器：

- MACA 版本：`3.3.0.15`
- GPU：`MetaX C500`

远端 `~/.bashrc` 已包含：

- `MACA_PATH=/opt/maca`
- `CUCC_PATH=/opt/maca/tools/cu-bridge`
- `CUDA_PATH=/opt/maca/tools/cu-bridge`
- `PATH += ${CUCC_PATH}/tools:${CUCC_PATH}/bin`

远端工具可用性：

- `cucc`：可用
- `cmake_maca`：可用
- `make_maca`：可用
- `mxcc`：可用
- 原生 `nvcc`：不在 `PATH`

远端还确认了两个关键点：

1. 不能用原生 `cmake` 直接配当前仓库  
   当前官方 `main` 的 `CMakeLists.txt` 仍然是 `project(... LANGUAGES CXX CUDA)`，直接跑原生 `cmake` 会卡在 `CMakeDetermineCUDACompiler`，报找不到 `nvcc/CUDA Toolkit`。

2. 正确入口应是 `cu-bridge` 路线  
   已验证：

```bash
export CUCC_CMAKE_ENTRY=2
export LIBRARY_PATH=${CUCC_PATH}/lib:/opt/mxdriver/lib

cd /data/HIPRT-main-main
cmake_maca -S . -B build_maca_cucc_try -DCMAKE_BUILD_TYPE=Release -DNO_UNITTEST=ON
```

上面的 `cmake_maca` configure 已成功。

注意：

- `~/.bashrc` 里当前有一处明显拼接错误：
  - `export LIBRARY_PATH=${CUCC_PATH}lib:$LIBRARY_PATH`
  - 少了 `/`
- 在当前 shell 内请显式修正为：

```bash
export LIBRARY_PATH=${CUCC_PATH}/lib:/opt/mxdriver/lib
```

## 远端工作目录建议

不要直接在 `/data/HIPRT` 上继续改。

原因：

- `/data/HIPRT` 当前是旧的 `maca_dev`
- 工作树很脏，含大量未跟踪文件和历史构建产物

建议：

- 保留 `/data/HIPRT` 仅作为历史参考和 patch 来源
- 在单独的干净目录上开发，例如：
  - `/data/HIPRT-main-main`
  - 或重新建一个新的 clean clone 目录

如果需要重新建干净目录，优先使用 cu-bridge/LFS 友好的方式，避免被旧分支里的历史 LFS 对象卡住。

## 需要重点参考的文件

官方主线当前工作目录：

- `CMakeLists.txt`
- `scripts/build.sh`
- `hiprt/hiprt_common.h`
- `hiprt/impl/Compiler.cpp`
- `hiprt/impl/Compiler.h`
- `hiprt/impl/Context.cpp`
- `hiprt/impl/BvhNode.h`
- `hiprt/impl/hiprt_device_impl.h`
- `test/hiprtTest.cpp`
- `test/main.cpp`
- `test/kernels/HiprtTestKernel.h`

历史 patch 和参考材料：

- `/data/HIPRT/patches/staged-upstream-sync-20260315/0002-maca-core.patch`
- `/data/HIPRT/docs/maca-adaptation-vs-official-zh.md`
- `/data/HIPRT/docs/maca-test-status-zh.md`

## 对远端 Codex 的明确任务

请按下面顺序执行：

1. 基于远端干净工作树，确认当前 `origin/main` 对应代码已经同步到本地最新提交。
2. 不修改代码，先用 `cmake_maca + make_maca` 跑一轮最小构建验证。
3. 分析 `0002-maca-core.patch`，只抽取当前官方 `main` 真正还缺失的 MACA 适配点，不要整包照搬。
4. 第一批优先落地以下内容：
   - runtime kernel disk cache 开关
   - `HIPRT_DISABLE_RUNTIME_KERNEL_CACHE`
   - `HIPRT_ROOT_DIRECTORY` / JIT include 路径稳定性
   - `__MACACC__` 下的 `WarpSize` 和 mask 类型修正
   - 32-bit mask 到 64-bit mask 的系统性清理
5. 每做完一批改动就编译；能跑测试后，先跑最小定向 case，不要一开始就全量。
6. 如果某个 patch 片段与当前官方主线冲突，必须按“当前主线语义优先”重写，而不是强行贴补丁。

## 当前优先级最高的技术判断

按照已经完成的分析，`0002` 真正需要关注的不是“把所有 MACA 历史修改搬回来”，而是下面几类：

- wave64 下 mask 位宽是否一致
  - `__activemask`
  - `__ballot_sync`
  - `__match_any_sync`
  - `__match_all_sync`
  - 以及对应的承载变量、wrapper、结构体字段、位操作
- 64-bit mask 后处理是否仍残留 32-bit 假设
  - `1u << lane`
  - `__ffs`
  - `__popc`
  - 32-bit bitset / 32-bit temporary
- runtime JIT cache 是否影响调试判断
- scene traversal / transform / update 的正确性问题是否仍然集中存在

## 构建与验证要求

构建时优先：

- `cmake_maca`
- `make_maca`
- `ccache`
- `mold`

但在 MACA + cu-bridge 环境里：

- 不要优先走原生 `ninja`
- 不要优先走原生 `cmake --build`
- 先保证 `cmake_maca` / `make_maca` 路线稳定

功能验证要求：

- 不要依赖旧的 HIPRT JIT cache
- 必要时清理工作区下的 `scripts/cache/`
- 必要时导出：

```bash
export HIPRT_DISABLE_RUNTIME_KERNEL_CACHE=1
```

## 输出要求

请远端 Codex 在每一轮工作后给出：

- 本轮修改了什么
- 哪些内容是从 `0002` 提炼过来的
- 哪些内容因为当前主线不同而被改写
- 当前构建是否通过
- 当前跑了哪些 case
- 哪些 case 通过，哪些仍失败
- 下一轮准备处理什么

## 可直接发送给远端 Codex 的简版提示词

```text
你现在在远端 MACA 机器上工作，目标是把 /data/HIPRT/patches/staged-upstream-sync-20260315/0002-maca-core.patch 所代表的 MACA core 适配，收敛到最新官方 main 的纯 CUDA 基线中。

注意事项：
1. 不要机械套 patch，必须按 patch 意图 + 当前 main 代码 + 实际编译/测试反馈来收敛。
2. 当前 /data/HIPRT 是脏的旧 maca_dev，只把它当 patch 和文档参考，不要直接在上面开发。
3. 使用干净工作树，例如 /data/HIPRT-main-main。
4. 远端不能用原生 cmake 直接配，因为没有原生 nvcc；请走 cu-bridge 路线：
   export MACA_PATH=/opt/maca
   export CUCC_PATH=$MACA_PATH/tools/cu-bridge
   export CUDA_PATH=$MACA_PATH/tools/cu-bridge
   export PATH=$PATH:${CUCC_PATH}/tools:${CUCC_PATH}/bin
   export CUCC_CMAKE_ENTRY=2
   export LIBRARY_PATH=${CUCC_PATH}/lib:/opt/mxdriver/lib
5. 优先使用 cmake_maca + make_maca，不要优先用原生 ninja / cmake --build。
6. 第一批只做最关键的 MACA core 改动：
   - runtime kernel cache 开关
   - HIPRT_DISABLE_RUNTIME_KERNEL_CACHE
   - HIPRT_ROOT_DIRECTORY 和 JIT include 路径
   - __MACACC__ 下 WarpSize / wave64 mask / 64-bit bit operations
7. 每做一批改动就编译并做最小定向验证。
8. 功能验证时必要时禁用 HIPRT JIT cache，避免缓存污染。

请先：
1. 检查当前干净工作树的提交和状态
2. 用 cmake_maca + make_maca 做一轮最小构建
3. 分析 0002-maca-core.patch 的前几类必要改动
4. 开始第一批代码修改并验证
```
