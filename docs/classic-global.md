# SDF-only classic_global

`classic_global` 将 macrofacet 的 `global_pointwise + generalized_gaussian`
过程接入 Tungsten 的 `Medium` 接口。直接使用场景均值 SDF；不加载、不生成密度场或
majorant 网格，不依赖相邻的 macrofacet 工程。当前接入 CPU `path_tracer`，含体积直接光照和 MIS。

## 运行示例

从 Tungsten 工程根目录执行（PowerShell）：

```powershell
cmake -S . -B build -DTUNGSTEN_BUILD_CLASSIC_GLOBAL_TESTS=ON
cmake --build build --config Release --target tungsten classic_global_test --parallel 4
ctest --test-dir build -C Release -R "^classic_global" --output-on-failure
.\build\Release\tungsten.exe -t 4 --spp 32 -d build/classic-global-image data/example-scenes/classic-global/scene.json
```

示例包含球形解析 SDF、介质包围盒、介质内部的镜面三角形和地面。
包围盒使用 `forward` BSDF，`int_medium` 指向 `classic_global`。
内部三角形不设置 `int_medium`/`ext_medium`，反射后继续留在当前介质。
渲染器先求最近三角形交点，再把 `ray.farT()` 之前的有限段交给介质采样。
没有介质事件时到达三角形；反射后以新方向重新计算方向相关消光。

## shaderball-stationary-27 的对应配置

`gpis-scenes/shaderball/shaderball-stationary-27-classic-global.json` 保留原文件的
程序化 `knob` SDF、变换、`sigma=0.02`、`lengthScale=0.05`、相机、几何和灯光，
默认仍为 512×512、128 spp。使用原来的绿色镜面反射率：

```json
"material": {"type": "mirror", "albedo": [0.15, 0.8, 0.5]}
```

`mirror` 在每次微面反射时乘以固定 RGB `albedo`，与原 `MirrorBsdf` 一致。
Gaussian VNDF、消光和自由程模型保持 classic_global 的定义，直接光照的
`eval` 也乘同一反射率。`conductor` 使用 `eta`、`k` 和 Fresnel；省略
`material.type` 时仍默认 `conductor`，兼容先前的配置。
`force_unit_fresnel` 仅适用于 conductor；白色 mirror 使用 `albedo: 1`。

这份 `knob` 使用单位外部缩放。其锥体项的 Lipschitz 常数为 `sqrt(2)*pi/3`，
底座乘以 `0.7` 后上界约为 `1.037`；其他 SDF 项及平滑布尔运算不会增大该界。
因此配置取 `sdf_lipschitz=1.1`。通用前向差分梯度每个分量至多为该界，
梯度长度至多为 `sqrt(3)*1.1`，配置取 `gradient_bound=2.0` 留出浮点余量。
修改 SDF 或变换缩放后应重新计算这两个界。

```powershell
.\build\Release\tungsten.exe -t 4 -d build/shaderball-classic-global gpis-scenes/shaderball/shaderball-stationary-27-classic-global.json
```

原文件保持不变；新配置的 PNG/EXR 文件名也独立于原场景。

## 使用已有 VDB SDF

保留原场景的相机、灯光、几何和包围盒；将对应 medium 改为下面的结构。
`name` 保持与包围盒的 `int_medium` 一致，SDF 路径相对于场景输入目录。

```json
{
  "name": "gp",
  "type": "classic_global",
  "gaussian_process": {
    "type": "standard",
    "mean": {
      "type": "tabulated",
      "grid": {
        "type": "vdb",
        "file": "scene_sdf.vdb",
        "density_name": "sdf",
        "density_scale": 1,
        "emission_name": "",
        "normalize_size": false,
        "request_sdf": false,
        "interpolate": "linear",
        "integration_method": "raymarching"
      }
    },
    "covariance": {
      "type": "squared_exponential",
      "sigma": 0.001,
      "lengthScale": 0.002828427
    }
  },
  "material": {"eta": [0.2, 0.9, 1.1], "k": [3.9, 2.5, 2.2]},
  "max_bounces": 64
}
```

这里的 `density_name` 是 Tungsten 原有 Grid 接口的通道名参数，读取内容仍是 SDF。
`integration_method` 是底层 Grid 配置；本 medium 使用自己的在线 tracking，
不调用 Grid 的密度积分。旧 medium 的 `density`、`sigma_a`、`sigma_s`、
`phase_function`、`transmittance` 应移除；不适用的这些字段会报错，避免静默改变模型。
`material.force_unit_fresnel: true` 可用于白炉测试。
积分器设为 `path_tracer`，`use_cuda_gpis: false`。

## Lambert 微材质与粗糙几何对照

`classic_global` 和 `global_conditional` 也支持
`"material": {"type": "lambert", "albedo": 0.9}`。此时先按碰撞处的通量加权
梯度分布抽取一次法线，再围绕该法线进行余弦半球散射。NEE 与延续策略共享这次
法线观测，使用条件于该法线的余弦 PDF 做 MIS；conditional 的两个分支还保留
同一个未归一化梯度作为起点观测。消光与距离采样不随材质改变。

粗糙平面与解析圆底 U 槽的复现实验见 `tools/surface_compare/rough_geometry.py`。
默认使用 Lambert 微材质，`--material mirror` 可生成镜面对照。U 槽程序化函数
`rounded_u_groove` 的局部均值为 `1 - sqrt(x*x + min(y,0)^2)`，沿 z 延伸；
缩放到半径 R 时同时设置 `transform.scale=R` 和均值的 `scale=R`，以保留世界空间距离。
圆底与直壁连接处法线连续；空腔中轴仍是距离场的中轴，不是表面棱线。

```powershell
python -B tools/surface_compare/rough_geometry.py render --stage final --size 128 --spp 128
python -B tools/surface_compare/rough_geometry.py validate --stage final
python -B tools/surface_compare/rough_geometry.py report --stage final
```

渲染需要 NumPy，出图还需要 Matplotlib 和 Pillow。输出在
`outputs/rough_geometry_compare/final/comparison.html`，包含同侧与对侧 75° 掠射光平面、
侧上方照明 U 槽，以及 GPIS 步长和计算域敏感性。修改材质或其他参数时使用新的
`--stage`，以免混用现有结果。

高分辨率比较使用可断点续渲的工作流：

```powershell
python -B tools/surface_compare/rough_geometry_quality.py check-resume
python -B tools/surface_compare/rough_geometry_quality.py run --size 512
```

输出入口为 `outputs/rough_geometry_compare/high_quality_512/comparison.html`，
`status.json` 记录当前进度。先以 64×64、64 spp 检查 GPIS 的 0.002→0.001 步长
及计算域扩大后的变化；正式图固定使用 0.002 步长。三场景、三模式统一从 64 spp
逐档加倍，全部九张完成一档后才发布该档比较。正式验收至少 1024 spp，同时要求：

- 全图及中央半幅平均亮度的约 95% 采样区间半宽不超过均值的 0.5%。
- 使用统一增益的显示图，独立奇偶样本半图估计的逐像素噪声 RMS 不超过 0.01
  （显示值范围 0–1）。
- 当前与上一档的线性亮度在 32×32 分区上的变化 RMS，不超过当前全图均值的 1%。

最多到 8192 spp；达到上限仍不满足时明确标记未收敛。这里只检查固定数值设置下的
蒙特卡洛采样稳定性，不能据此宣称有限步长、有限计算域或稀疏卷积近似没有偏差。
相邻 spp 档共享已有样本，不是独立重复实验。

任务复制并校验渲染器和 DLL，避免后续编译影响长任务。每五分钟保存原生检查点；
中断后重跑相同命令会续算，已完成的各档图像保持独立。修改分辨率或参数时指定新的
`--stage`。原生续渲验证比较三种模式的 16→32 spp 和直接 32 spp，要求最终图、
两个半样本图及方差图逐项一致。报告需要 Matplotlib 和 Pillow；脚本使用本机
`gaussian_splatting` Python 环境出图。

只调整正式图分辨率时，可以复用已完成的低分辨率数值核验，例如：

```powershell
python -B tools/surface_compare/rough_geometry_quality.py run --size 256 --reuse-numerics-from high_quality_512
```

复用前会检查全部九次数值渲染的场景、渲染器版本、样本数、随机种子和文件哈希；
正式图会按新分辨率重新采样。后续续跑应使用同一条命令，保留复用来源参数。

## 保留的数学模型

令 `m(x)` 为 SDF，`sigma` 为场标准差，`g(x) = grad m(x)`，
`C = Cov(grad F, grad F)` 为平稳核在零距离处的混合二阶导数。
对单位传播方向 `w`：

```text
rho(x)     = phi(m/sigma) / (sigma * Phi(m/sigma))
A(x,w)     = E[max(-w.G, 0)],   G ~ Normal(g(x), C)
sigmaT(x,w)= rho(x) * A(x,w)
```

不归一化 `g(x)`，不投影掉法向梯度方差。这是 global_pointwise 模型。
镜面与导体散射按 `max(-w.G,0)` 加权的三维高斯梯度抽样，归一化后得到可见法线并反射。
这两种材质的 `eval`、`pdf`、`sample` 使用同一个 Gaussian NDF；精确 VNDF proposal
使散射权重仅剩材质反射率（导体 Fresnel 或镜面 albedo）。没有跨射线 GP 条件历史。

距离采样使用 delta tracking，阴影透射率使用 ratio tracking。
每段通过 SDF 的 Lipschitz 上界得到最小均值，再计算密度和消光上界；
不会用固定步长积分替代碰撞概率，也不会把上界违例截断到 1。
`tracking_step` 是世界空间分段长度，默认自动选择，只影响效率。
透射率很小时使用无偏俄罗斯轮盘。

## SDF、协方差及实现范围

- 自动求界：`spherical`、`linear`、`homogeneous` 均值，以及线性插值的 VDB SDF。
- VDB 在内存中复制一份原始 SDF 值并扫描边差，得到保守梯度界；这不是密度烘焙。
  值和梯度由同一三线性插值实时计算，包含变换、scale、offset 和边界 clamp。
  因此梯度与 Tungsten 通用 `TabulatedMean` 的固定 `0.001` 前向差分会有差别。
  额外内存与 SDF 包围盒中的体素数成正比，目前尚未优化为稀疏访问。
- 其他均值需显式提供 `sdf_lipschitz` 和 `gradient_bound`（世界坐标单位），
  分别约束均值变化和实际梯度求值器。不能直接假定变换后的 SDF 梯度长度为 1。
- 协方差必须平稳、可微，且得到有限正场方差与正定梯度协方差。
  已验证 squared-exponential；允许满足上述条件的其他核。
  不支持 `conditioning_data`、`embed_cov`、非平稳核、退化梯度分布或 `volume: true`。
- 必须有有限介质边界；到无限环境的射线需要先经过包围盒出口。
  VDB 超出采样域沿用已有 Grid 的 clamp 语义，介质的空间范围由包围盒确定。
- 当前不提供边缘化的自由程 PDF、BDPT/MLT 路径反演或 CUDA kernel。
  这些接口会明确报错，不会把 tracking 中的辅助权重当作 BDPT 所需 PDF。
  位置独立于方向的 `sigmaS(p)`/`sigmaT(p)` 也不适用，应调用 `extinction(p,w)`。

测试覆盖高斯尾部数值、方向相关消光、未归一化梯度、有限区间碰撞概率与解析积分、
ratio tracking 均值、VNDF 样本与独立方向积分、白炉权重、JSON 往返，
以及仅含 SDF 通道的 VDB 变换/插值/梯度。
另外有真实镜面三角形 + SDF 介质的渲染回归；检查渲染日志中的内部异常，
因为 Tungsten 捕获单条路径异常后仍可能以进程成功状态结束。
