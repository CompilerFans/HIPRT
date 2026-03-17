# HIPRT 宣传级光追效果图方案

## 1. 当前仓库能直接复用的能力

当前 `HIPRT` 主仓里，已经有一条可直接复用的“离线静态渲染”最短路径：

- `ObjTestCases::createScene(...)` 支持从 `OBJ + MTL` 载入网格、材质、面光源并构建 `hiprtScene`
- `ObjTestCases::render(...)` 能编译 trace kernel、发射主光线并输出 `PNG`
- `test/common/meshes/cornellbox/cornellBox.mtl` 已证明 `MTL` 里的 `Ke` 可被当前测试管线识别成 emissive light

对应代码入口：

- `test/hiprtTest.cpp`
- `test/hiprtTest.h`
- `test/shared.h`

这意味着要先产出一张展示图，并不需要先做完整 SDK 应用，直接复用测试侧 `OBJ` 场景装载和图片落盘逻辑即可。

## 2. 当前能力与“非常真实”目标之间的差距

仓库当前常用 kernel 主要是：

- `test/kernels/PrimaryRayKernel.h`
- `test/kernels/ShadowRayKernel.h`
- `test/kernels/AoRayKernel.h`

它们更偏功能验证和诊断，可用于：

- 法线/ID/UV/命中距离可视化
- 一次直接光或 AO
- scene/geometry/traversal 回归

但它们还不等价于“宣传级真实感渲染”。要达到较强真实感，至少要补下面几项：

1. 多反弹 path tracing
2. emissive 面光源的重要性采样
3. 累积采样与降噪
4. 更完整的材质模型
5. tone mapping 与曝光控制

如果只用当前 `PrimaryRayKernel`，画面会更像功能截图，而不是海报级效果图。

## 3. 推荐的技术路线

### 3.1 第一阶段：先做可控静态图

目标是先稳定产出一张带文案的静态图：

- 文案：`沐曦集成电路 MetaX Graphics Ray Tracing`
- 资产：把文本转成可导入的 `OBJ`
- 舞台：深色地台 + 背板 + 顶部面光
- 输出：单张 `PNG`

当前仓库里最合适的入口是：

- 继续复用 `ObjTestCases::setupScene(...)`
- 增加一个无参考图的 showcase 场景用例
- 输出到单独目录，不参与像素金标比较

### 3.2 第二阶段：补 path tracing kernel

为了让金属字、边缘高光、字缝遮蔽、背板反照更自然，建议新增一个专门的 showcase kernel，例如：

- `test/kernels/ShowcasePathTracerKernel.h`

建议最小能力：

1. 2 到 4 次 bounce
2. Lambert + simple specular/rough conductor
3. 面光源直接采样
4. Russian roulette
5. 累积 `N` 帧做降噪前的收敛

### 3.3 第三阶段：输出动态视频

动态视频不建议转动物体，建议只旋转主光或补光：

- 字体和镜头固定
- 主光沿 Y 轴缓慢旋转
- 每帧重新渲染
- 用 `ffmpeg` 合成 `mp4`

这样能明显展示：

- 字体 bevel 边缘高光移动
- 金属表面反射变化
- 背板与地台上的阴影漂移

## 4. 文案资产建议

推荐把宣传字拆成两层：

1. 中文主体：`沐曦集成电路`
2. 英文副标题：`MetaX Graphics Ray Tracing`

造型建议：

- 中文字重厚一些，做较深挤出
- 英文字略细，放在下方或右下角
- 整体不要贴墙，和背板留出距离，方便形成轮廓光

如果只是验证管线，可先用脚本生成像素挤出字体。
如果目标是海报或视频封面，推荐后续改用 DCC 工具生成 bevel 过的高质量文字网格，再导出 `OBJ`。

## 5. 材质建议

当前测试 `Material` 仅直接消费：

- `Kd`
- `Ke`

也就是 diffuse 和 emission 已经能直接走通。若想做“更真实”的金属字，建议后续把测试/示例侧材质扩展为：

- baseColor
- emission
- roughness
- metallic
- specularColor

如果本阶段先不扩材质，可以先这样近似：

- 文字：高亮暖灰 `Kd`
- 地台：低亮深灰 `Kd`
- 主光：高 `Ke`

这能先验证构图、轮廓和阴影关系。

## 6. 灯光与镜头建议

### 6.1 灯光

建议至少三层：

1. 顶部主面光
2. 侧后方轮廓光
3. 很弱的环境填充

如果先复用当前 `OBJ + MTL + emissive triangles` 路线，主面光和轮廓光都可以直接建成 emissive quad。

### 6.2 镜头

静态图推荐：

- 轻微低机位仰拍
- 35 到 50 度视场
- 让文字形成透视纵深

动态视频推荐：

- 镜头固定
- 主光旋转 120 到 180 度
- 视频时长 4 到 8 秒

这比同时动镜头和动光更容易控制结果，也更适合做首版展示。

## 7. 本仓新增工具

### 7.1 文本转 OBJ

新增：

- `tools/showcase/generate_text_obj.py`

作用：

- 从 `TTF/OTF/TTC` 字体把文本栅格化
- 自动挤出成 3D `OBJ`
- 同时写出 `MTL`
- 可选地一并生成地台、背板和顶部面光

注意：

- 当前机器只有 `DejaVu Sans`，默认不覆盖中文字符
- 生成中文场景时，需要显式传入可用的 CJK 字体文件，例如 `Noto Sans CJK` 或 `Source Han Sans`

### 7.2 帧序列转视频

新增：

- `tools/showcase/encode_rotation_video.sh`

作用：

- 把 `frame_0000.png` 这类序列编码成 `mp4`

## 8. 推荐执行方式

### 8.1 先生成文本场景

示例：

```bash
python3 tools/showcase/generate_text_obj.py \
  --text "沐曦集成电路 MetaX Graphics Ray Tracing" \
  --font /path/to/SourceHanSansSC-Bold.otf \
  --output-dir test/common/meshes/metax_showcase \
  --scene-name metax_title \
  --with-stage
```

输出会包含：

- `test/common/meshes/metax_showcase/metax_title.obj`
- `test/common/meshes/metax_showcase/metax_title.mtl`

### 8.2 用 HIPRT 跑静态图

建议直接新增一个 showcase 用例，复用：

- `ObjTestCases::setupScene(...)`
- `ObjTestCases::render(...)`

静态图第一版先用无参考图模式，把图输出到独立目录。

### 8.3 批量出旋转光帧

建议新建一个小脚本或单独示例程序：

1. 固定相机和物体
2. 每帧修改主面光角度
3. 渲染 `frame_%04d.png`

### 8.4 编码视频

```bash
tools/showcase/encode_rotation_video.sh frames metaxtitle_spin.mp4 24
```

## 9. 建议的研发顺序

建议按下面顺序推进，而不是一开始就追求最终海报质量：

1. 先用 `generate_text_obj.py` 把文案转成 `OBJ`
2. 先复用现有 `ObjTestCases` 跑通一张静态图
3. 新增 showcase kernel，补 path tracing
4. 再做旋转主光的逐帧输出
5. 最后再替换成更高质量的 bevel 文本和更完整材质

## 10. 当前结论

结论很明确：

- `HIPRT` 当前仓库已经具备“展示场景资产导入 + 基础光追成图 + PNG 输出”的基础能力
- 要做“非常真实”的宣传图，核心不在 BVH 或场景装载，而在于新增一条专用的 path tracing kernel
- 文本资产可以先通过仓内脚本自动生成，后续再替换成 DCC 导出的精细模型
- 视频输出不需要额外视频框架，直接逐帧渲染后交给 `ffmpeg` 即可
