# mxcc 离线编译增强第一阶段结果

## 1. 这轮做了什么

本轮没有改主构建入口，而是只改了离线产物脚本：

- `scripts/bitcodes/compile.py`
- `scripts/bitcodes/precompile_bitcode.py`

新增能力：

- 支持显式 `--toolchain mxcc`
- 支持直接指定：
  - `--compiler /opt/maca/mxgpu_llvm/bin/mxcc`

同时修正了一个源头问题：

- 当前仓库已经是 CUDA-only
- 因此 `hiprt_kernels.h` / `hiprt_kernels_bitcode.h` 不应再回落到 `hip/hip_runtime.h`
- 本轮已改成：
  - `__CUDACC__ || __MACACC__` 走 CUDA 分支
  - 否则直接报错

## 2. 这轮验证结果

### 2.1 compile.py 的 mxcc 路径

已验证命令：

```bash
cd /data/HIPRT/scripts/bitcodes
python3 compile.py \
  --root /data/HIPRT \
  --compiler /opt/maca/mxgpu_llvm/bin/mxcc \
  --toolchain mxcc \
  --config Release
```

验证结果：

- 可以成功执行 `mxcc` 生成：
  - `hiprt03001_nv_lib.fatbin`
  - `hiprt03001_nv.fatbin`

其中 `hiprt_kernels_bitcode.h` 这条路径需要额外 wrapper：

- 为 `intersectFunc`
- `filterFunc`

提供默认实现，否则 `mxcc` 离线链接阶段会直接报未定义符号。

这说明一个很关键的结论：

- `mxcc` 的离线链接约束更严格
- 但也正因为更严格，它更早暴露出当前离线编译链真正依赖的运行时约定

### 2.2 precompile_bitcode.py 的 mxcc 路径

已验证命令：

```bash
cd /data/HIPRT/scripts/bitcodes
python3 precompile_bitcode.py \
  --root /data/HIPRT \
  --compiler /opt/maca/mxgpu_llvm/bin/mxcc \
  --toolchain mxcc \
  --config Release
```

验证结果：

- 能成功生成：
  - `hiprt03001_nv_precompiled_bitcode.fatbin`

这说明：

- 对当前 precompiled trace-kernel 路径，`mxcc` 已经能作为可用后端

### 2.3 runtime cubin fallback 的第二阶段验证

已做验证：

- 在主仓 runtime-bitcode 测试中显式设置：
  - `HIPRT_EXTERNAL_DEVICE_COMPILER=mxcc`
- 让测试辅助通过外部编译器生成用户 cubin
- 再把该 cubin 传给：
  - `hiprtBuildTraceKernelsFromBitcode(...)`

验证结果：

- 外部编译器选择器已生效
- fallback 已真正走到 `mxcc`
- 但在 runtime 链接阶段失败于：
  - `cuLinkAddData`
  - 具体表现为 `mcErrorInvalidKernelImage`

这说明一个新的、更精确的结论：

- **`mxcc` 产出的离线 fatbin / precompiled fatbin 可以被 `cuModuleLoadData` 消费**
- **但当前 `mxcc` 产出的“用户 cubin”还不能被现有 runtime bitcode linking 入口稳定接受**

也就是说，第二阶段已经回答了“问题卡在哪”：

- 不再是编译器选择问题
- 而是 **runtime bitcode link 输入格式 / relocatable 属性 / linker 期望的二进制类型** 不匹配

## 3. 这轮确认的“本质缺陷”

当前离线编译的本质问题不是：

- 产物文件格式不支持
- 或 `mxcc` 完全无法参与

真正的本质问题是：

1. **源文件分支仍残留 HIP 路径**
   - 例如 `hiprt_kernels*.h` 里回落到 `hip/hip_runtime.h`
   - 这会直接导致 `mxcc` 走错分支

2. **运行时 custom-func 包装语义没有被显式带进离线编译**
   - `intersectFunc`
   - `filterFunc`
   - 以及相关 helper
   如果不显式补齐，`mxcc` 离线链接阶段会直接失败

也就是说：

- `mxcc` 没有暴露一个全新问题
- 它只是把当前离线链里原本隐藏的依赖关系更早、更明确地暴露出来

3. **runtime bitcode link 对用户二进制的输入格式要求比 `cuModuleLoadData` 更严格**
   - 同样是 `mxcc` 产物：
     - precompiled fatbin 可加载
     - runtime `cuLinkAddData` 却可能拒绝
   - 这说明“能离线生成”与“能作为 runtime bitcode link 输入”不是同一个兼容层级

## 4. 这轮之后的判断

当前可以给出更明确的结论：

- **`mxcc` 已经可以作为 HIPRT 离线产物生成链的后端使用**

但同时也要明确：

- **要让它真正稳定，不是单纯换编译器名字，而是要继续把源码中的 HIP 残留和 custom-func 包装依赖清干净。**
- **同时还需要继续确认：`mxcc` 生成的用户设备二进制，究竟应以 `CUBIN`、`FATBIN`、`device-bc` 还是其它形式喂给 runtime bitcode link。**

## 5. 下一步建议

如果继续推进第二阶段，优先顺序建议是：

1. 继续清理 active path 中的 `hip_runtime.h` / HIP 分支残留
2. 把 `compile.py` 里 wrapper 的逻辑再参数化，避免只服务当前 `hiprt_kernels_bitcode.h`
3. 继续研究 `cuLinkAddData` 对 `mxcc` 产物的真实可接受输入类型：
   - 当前 `CU_JIT_INPUT_CUBIN` 路径失败
   - 后续应试验：
     - `-fatbin`
     - `-device-bin`
     - `-fatbc`
     - 以及是否需要不同的 `CUjitInputType`

## 6. 一句话结论

- **这轮已经验证：`mxcc` 可以承接 HIPRT 的离线产物生成；当前真正需要继续解决的，不是“mxcc 能不能编”，而是“如何把离线编译需要的 CUDA-only 分支和 custom-func 包装语义系统性补齐”。**
