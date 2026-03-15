# MACA 远端开发交接说明

## 当前背景

- 当前仓库本地分支基于官方最新 `main`。
- 已完成第一阶段：基于 `patches/staged-upstream-sync-20260315/0001-cuda-only.patch` 的意图，在官方主线上手工重放纯 CUDA 适配。
- 当前本地基线对应提交：
  - `6520392` `cuda: 基于0001阶段补丁完成纯CUDA适配`

## 本地阶段已完成内容

- 构建系统收敛为 `CMake + CUDA Toolkit`。
- 默认优先使用 `Ninja + ccache + mold` 加速构建。
- 清理旧的 HIP/ROCm、premake、bitcode 和大量不再参与当前主路径的 Orochi 子树。
- 保留 `hiprt*` 公开 API 命名和当前主测试路径。
- 新增中文 `AGENTS.md`，记录了：
  - 默认中文沟通
  - 优先尝试 `ninja + ccache + mold`
  - 功能验证时不要依赖历史 HIPRT JIT cache

## 本地验证结果

- 构建命令：

```bash
./scripts/build.sh
```

- 功能测试命令：

```bash
cd scripts
./unittest.sh
```

- 结果：
  - `39 / 39` 非性能功能测试通过
  - 当前纯 CUDA 基线可作为后续 MACA 适配的稳定起点

## 当前无法在本地继续推进的原因

- 第二阶段 `0002-maca-core.patch` 的核心内容已经分析过，主要集中在：
  - 运行时 kernel cache 控制
  - warp/lane mask 语义抽象
  - MACA 相关 runtime / traversal 行为修正
- 但当前本地环境没有 MACA 工具链和运行环境，无法验证：
  - `__MACACC__` / warp64 相关路径
  - cu-bridge 与 MACA driver/runtime 的交互
  - MACA 环境下的 scene / traversal / transform 正确性

因此，本地继续推进会退化成“只改代码、不验证行为”，风险过高，应该转到远端 MACA 环境完成。

## 远端阶段建议顺序

1. 在远端 `/data/HIPRT` 同步当前 `origin/main`。
2. 先确认远端能复现当前纯 CUDA 基线的构建与基础测试。
3. 再最小化引入 `0002-maca-core.patch` 中真正需要的 MACA compile/runtime 适配。
4. 每引入一批改动就跑一轮定向功能测试，不要直接吞后续 `0003/0004`。
5. 优先关注：
   - runtime kernel cache 开关
   - warp size / lane mask 语义
   - scene traversal / transform / update 路径

## 注意事项

- 不要机械套 `0002`，必须逐项核对当前主线语义。
- 功能验证前优先清理测试过程中生成的 `scripts/cache/`。
- 如果验证 trace kernel/JIT 行为，必要时显式绕开 cache，避免旧缓存影响判断。
