# offline compile 开始前的提交整理建议

## 1. 目标范围

这里讨论的是：

- **开始做 offline compile 之前**
- 当前 `main` 上已经存在的 MACA 适配提交

本次范围不包含：

- `d0a3fa5` 及之后的 offline / bitcode / precompile / maca-link 系列提交

本次范围包含的提交是：

1. `4c86eed` `maca: enable cucc core build path`
2. `94a283c` `scripts: use make_maca in cucc build helper`
3. `e24a1f7` `maca: fix cu-bridge jit and scene transforms on main`
4. `e26b33e` `docs: record cache-enabled main validation`
5. `210a0fc` `test: migrate maca diagnostics to main`
6. `b51b927` `docs: review main against upstream_main_dev`

基线前一个提交：

- `a803f96` `chore: 删除Windows LFS二进制依赖`

它和 MACA 主适配逻辑关系不强，不建议并入本次整理。

## 2. 当前这些提交的本质差异

虽然这 6 个提交是按推进节奏逐步落下来的，但按“改动本质”看，其实可以压缩成更少的特性组。

### 2.1 构建链接入

对应提交：

- `4c86eed`
- `94a283c`

本质内容：

- `cmake_maca + cucc` 路径接入
- `make_maca` 作为稳定构建入口
- 与 MACA 编译入口相关的宏和脚本接线

### 2.2 主路径功能修复

对应提交：

- `e24a1f7`

本质内容：

- cu-bridge runtime JIT 兼容
- scene transform / traversal 修复

这部分是当前主路径真正的功能修复核心。

### 2.3 测试与验证补强

对应提交：

- `210a0fc`

本质内容：

- 从 `maca_dev` 迁移补充测试
- 把主路径问题收敛成更可复现的红绿灯

### 2.4 阶段性说明文档

对应提交：

- `e26b33e`
- `b51b927`

本质内容：

- 记录 cache 开启验证
- 记录相对 `upstream_main_dev` 的差异审查结果

这类提交是“阶段记录”，不属于功能本体。

## 3. 是否建议合并成 1 个提交

**不建议全部压成 1 个提交。**

原因：

- `构建链接入`
- `功能修复`
- `测试迁移`
- `文档审查`

这 4 类东西性质明显不同。

如果硬压成 1 个提交，后续看历史会失去两个关键信息：

1. 哪些是“让代码能编”的改动
2. 哪些是“让功能正确”的改动

## 4. 建议压缩成几个关键特性提交

### 方案 A：压成 3 个提交

这是我认为最平衡、最适合后续维护的方案。

#### 提交 1：MACA 构建链接入

建议合并：

- `4c86eed`
- `94a283c`

建议标题：

- `maca: add cucc/make_maca build path on main`

理由：

- `94a283c` 本质只是 `4c86eed` 的补丁修正
- 单独留一个提交价值不高
- 合并后更符合“一个完整构建特性”

#### 提交 2：主路径运行时与场景修复

保留：

- `e24a1f7`

建议标题可保持原意：

- `maca: fix cu-bridge jit and scene transforms on main`

理由：

- 这是最重要的功能修复提交
- 单独保留最有利于后续审查和回溯

#### 提交 3：测试与阶段文档

建议合并：

- `e26b33e`
- `210a0fc`
- `b51b927`

建议标题：

- `test: migrate maca diagnostics and document main validation`

理由：

- 这三笔都属于“验证与记录层”
- 其中：
  - `210a0fc` 是测试本体
  - `e26b33e` 是验证说明
  - `b51b927` 是差异审查说明
- 作为一个阶段性验证提交比较自然

### 方案 B：压成 4 个提交

如果你希望“审查文档”和“测试迁移”分得更开，可以改成 4 个提交：

1. `4c86eed + 94a283c`
2. `e24a1f7`
3. `210a0fc`
4. `e26b33e + b51b927`

这个方案也成立，但我认为信息密度不如方案 A。

## 5. 哪些提交不建议继续拆

### `e24a1f7`

不建议继续拆。

理由：

- 它虽然同时改了：
  - JIT workaround
  - scene transform
  - 少量测试/脚本同步
- 但这些在当前主线上是同一个“MACA 主路径功能恢复”问题域

如果强拆，反而会让后续 cherry-pick / rebase 更麻烦。

### `210a0fc`

如果不和文档合并，就建议保持独立，不要再往更碎拆。

理由：

- 它本身已经是一组连贯的测试迁移
- 再往下拆文件级提交价值很低

## 6. 哪些提交其实只是修补前一个提交

### `94a283c`

这是最明确的“可以并进前一个提交”的提交。

原因：

- 它只是在 `build_maca_cucc.sh` 中把调用入口从不稳定方式收敛到 `make_maca`
- 本质就是构建接入提交的 follow-up

### `e26b33e`

这类提交独立存在可以，但如果目标是“整理提交”，最适合并进验证/测试组。

## 7. 当前建议的最终整理结构

如果真要做历史整理，我建议你把 offline compile 之前的阶段整理成：

1. `maca: add cucc/make_maca build path on main`
   - 含 `4c86eed + 94a283c`

2. `maca: fix cu-bridge jit and scene transforms on main`
   - 含 `e24a1f7`

3. `test: migrate maca diagnostics and document main validation`
   - 含 `e26b33e + 210a0fc + b51b927`

## 8. 一句话结论

- **offline compile 开始前的这 6 个提交，不建议压成 1 个提交；更合理的整理方式是压成 3 个关键特性提交，其中 `94a283c` 明确应并入 `4c86eed`。**
