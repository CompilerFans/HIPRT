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

### 2.4 最小输出形态实验矩阵

为了继续判断是不是“产物类型选错”，补做了最小 `mxcc` 输出实验：

- `mxcc -device-bin`
  - 输出文件头：`ELF`
- `mxcc -fatbin`
  - 输出文件头：`__CLANG_OFFLOAD_BUNDLE__`
- `mxcc -fatbc`
  - 输出文件头：`__CLANG_OFFLOAD_BUNDLE__`

然后继续把 runtime fallback 改成：

- `mxcc` 路径产出 `fatbin`
- linker 侧识别 `FATBINARY`

再强制运行主仓 runtime-bitcode 测试，结果仍然失败于：

- `cuLinkAddData`
- `mcErrorInvalidKernelImage`

因此可以把问题进一步压缩成一句话：

- **当前 blocker 已不是“应该产出 cubin 还是 fatbin”，而是“当前 cu-bridge runtime linker 不接受 `mxcc` 生成的用户输入二进制，无论是 ELF 还是 clang offload bundle”。**

### 2.5 `mxcc --maca-link` 最小设备链接实验

为了确认问题是不是必须改走 `mxcc` 自己的设备链接流程，补做了最小实验：

1. 先用：
   - `mxcc -fgpu-rdc -c`
   生成含设备代码的目标文件

2. 再用：
   - `mxcc -fgpu-rdc --maca-link ... -fatbin`
   生成设备 bundle

实验结果：

- `--maca-link -fatbin` 能成功产出 `__CLANG_OFFLOAD_BUNDLE__`
- 并且该 bundle 可以被当前 `cuModuleLoadData` 成功加载，随后可通过 `cuModuleGetFunction` 取到最小 kernel symbol
- 这说明 `mxcc` 自身的设备链接流程是可工作的

但同时也确认了一个现实约束：

- 当前 HIPRT 主链里使用的是：
  - `cuLinkAddData`
  - `cuModuleLoadData`
- 而不是 `mxcc/maca-link` 设备链接流程本身

所以目前可以明确：

- **`maca-link` 是后续值得继续深入的方向**
- **它已经被证明能接到“模块加载”这一步**
- **当前未解决的是：怎样把它和 HIPRT 现有 runtime bitcode API / runtime link 入口拼起来**

### 2.6 `mxcc --maca-link` TraceKernel 级别实验

在最小空 kernel 之外，又补做了更接近 HIPRT 实际使用方式的实验：

- 目标源不是空 `K()`
- 而是：
  - `TraceKernel`
  - `CutoutKernel`
- 同时显式补了默认：
  - `intersectFunc`
  - `filterFunc`

实验链为：

1. `mxcc -fgpu-rdc -c`
2. `mxcc -fgpu-rdc --maca-link -fatbin`
3. `cuModuleLoadData`
4. `cuModuleGetFunction`

验证结果：

- `TraceKernel`
  - 可取到 symbol
- `CutoutKernel`
  - 可取到 symbol

这意味着：

- **`maca-link` 路线不仅能承载空 probe kernel，也已经能承载包含 HIPRT 设备遍历实现的真实 tutorial/test 级别 kernel。**

### 2.7 工程化结果

为了避免第三阶段只停留在一次性命令实验，又新增了仓库内脚本：

- `scripts/bitcodes/build_mxcc_trace_bundle.py`

用途：

- 输入一份用户 kernel 源文件
- 通过：
  1. `mxcc -fgpu-rdc -c`
  2. `mxcc --maca-link -fatbin`
  生成一个可直接加载的 MACA bundle

它的目标不是替代当前主构建入口，而是：

- 为后续“离线先设备链接，运行时只模块加载”的路径提供固定入口

### 2.8 对外 API 接入

当前已经把这条 bundle 路线接到 HIPRT 对外接口中：

- `hiprtBuildTraceKernelsFromLinkedBundle(...)`

这个 API 的定位非常明确：

- 输入的是一个已经离线链接好的 bundle
- 不再要求运行时再走 `cuLinkAddData`
- 运行时只做：
  - 模块加载
  - symbol 获取

这为后续正式把 `maca-link` 路线接入 HIPRT 主工作流提供了 API 落点。

### 2.9 客户侧无感 source API 回退验证

进一步补做了一个更接近真实使用方式的验证：

- 客户仍然调用：
  - `hiprtBuildTraceKernels(...)`
- 不改调用接口
- 仅通过环境变量强制内部走：
  - `mxcc -c`
  - `mxcc --maca-link -fatbin`
  - `cuModuleLoadData`

对应测试：

- `hiprtTest.BuildTraceKernelWithForcedMxccBundleFallback`

验证结果：

- 测试通过

这说明：

- **对典型 source-based 用法，HIPRT 已经具备“客户侧继续沿官方 API 调用，内部自动切到 maca-link bundle fallback”的能力。**

## 2.10 当前第三阶段结论

现在已经可以把第三阶段结论写得更具体：

1. `cuLinkAddData`
   - 当前不接受 `mxcc` 产出的用户输入二进制

2. `mxcc --maca-link -fatbin`
   - 可以产出 bundle
   - 该 bundle 可以被 `cuModuleLoadData` 直接加载
   - 对包含 HIPRT 设备遍历实现的 `TraceKernel/CutoutKernel` 也成立

因此当前最有希望、也已经开始落地的方向是：

- **不要再试图把 `mxcc` 用户产物强塞给当前 `cuLinkAddData`**
- **而是转向“离线先用 `mxcc --maca-link` 完成设备链接，运行时只做 `cuModuleLoadData`”的新路径**
- **并且对 source-based 官方 API，用内部 fallback 保持客户侧尽量无感**

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
   - 目前进一步确认：
     - `device-bin` 不行
     - `fatbin` 也不行
     - 问题仍在 runtime linker 的输入兼容层

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
3. 后续不应再只做“换一种本地输出文件后缀”的试验，而应转向：
   - 调研 cu-bridge runtime linker 对用户输入的真实支持矩阵
   - 或探索是否需要使用 `mxcc` / `maca-link` 自身的设备链接流程，而不是继续强塞到当前 `cuLinkAddData` 路径
   - 当前最新实验已经说明：继续只换 `device-bin/fatbin/fatbc` 的收益很低，下一步应优先研究 `maca-link` 产物如何接到 HIPRT runtime 模块加载路径

## 6. 一句话结论

- **这轮已经验证：`mxcc` 可以承接 HIPRT 的离线产物生成；当前真正需要继续解决的，不是“mxcc 能不能编”，而是“如何把离线编译需要的 CUDA-only 分支和 custom-func 包装语义系统性补齐”。**
