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
散射按 `max(-w.G,0)` 加权的三维高斯梯度抽样，归一化后得到可见法线并反射。
相函数的 `eval`、`pdf`、`sample` 使用同一个 Gaussian NDF；精确 VNDF proposal
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
