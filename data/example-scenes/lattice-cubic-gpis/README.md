# 固定格点卷积 GPIS

`shaderball.json` 使用 CPU 路径追踪器、三线性 OpenVDB SDF 和各向同性平方指数核。用已有的 shaderball 体数据，从仓库根目录运行以下命令即可生成 PNG 和线性 RGB 的 PFM。

```powershell
build/Release/tungsten.exe -t 8 -s 20261024 -d outputs/lattice-shaderball data/example-scenes/lattice-cubic-gpis/shaderball.json
```

场景读取 `outputs/shaderball_vdb_ray_profiles_20261117/fields/shaderball_sdf.vdb`。新检出需要先准备这个未纳入版本控制的体数据。仓库中现有的 `tools/surface_compare/shaderball_vdb_profiles.py` 给出了用 `gpis_shaderball_bake` 以 0.005 的体素间距烘焙、再用 `gpis_nvdb_to_vdb` 转换的命令。对已有的其他线性 VDB，可以直接换掉场景中的网格路径和密度通道名。

## 非平稳 shaderball

使用同一份 VDB 和 `shaderball-nonstationary.json` 可以渲染随高度变化的振幅和长度尺度。

```powershell
build/Release/tungsten.exe -t 8 -s 20261024 -d outputs/lattice-shaderball-nonstationary data/example-scenes/lattice-cubic-gpis/shaderball-nonstationary.json
```

`gaussian_process.covariance` 使用现有的 `proc_nonstationary` 接口，`cov` 取各向同性 SE，`var` 是局部标准差的乘数，`ls` 是局部长度尺度的乘数。支持常量和单轴渐变 `bottom_top`、`front_back`、`left_right`，长度尺度在三个空间轴上相同。格点间距等于 `lattice_spacing_in_ell` 乘以全场最小长度尺度，支撑半径等于 `kernel_radius_in_ell` 乘以查询点的长度尺度。默认核尾在 2.5ℓ 到 3ℓ 之间平滑衰减。参数渐变两端作为额外结点，确保每段使用正确的单侧导数。

每段用共享高斯系数计算两个端点的值和总斜率，再做三次 Hermite 插值。表面出射时用这套离散权重联合约束起点值和轴向导数。横向梯度由两部分构成，一部分是当前系数对振幅、长度尺度空间变化的响应，另一部分由两个独立的横向高斯分量给出。这个法线模型及逐方向的一维场是对三维非平稳过程的近似。对变宽各向同性高斯核，连续一维卷积与三维核的射线限制本身也有协方差差异。

在 i9-13900HX 上固定 8 个 P 核，256 × 256、128 spp 的非平稳 shaderball 纯渲染用时 102.945 秒，包含加载和初始化的总耗时为 137.037 秒。128 × 128、32 spp 的对照中，在参数恒定区间使用局部噪声上界，纯渲染从 12.507 秒降至 6.115 秒，两张线性 PFM 逐字节相同。[完整图像和计时](../../../outputs/shaderball_first_hit_rays/nonstationary_render/full/timing.json) · [渲染 PNG](../../../outputs/shaderball_first_hit_rays/nonstationary_render/full/render.png)

在相机中心、穿过渐变、掠射和条件化出射四条光线上，每条用 8192 个 realization 与独立的三维非平稳 SE 参考比较。累计首碰概率曲线的最大绝对差分别为 1.37、1.34、2.34 和 0.76 个百分点。这些是有限采样的经验差值。[分布图](../../../outputs/shaderball_first_hit_rays/nonstationary_render/first_hit_cdf.png) · [首碰统计](../../../outputs/shaderball_first_hit_rays/nonstationary_render/first_hit_summary.json)

对掠射光线换一组种子，每种方法采样 32,768 次后，格点和参考的命中率分别为 59.56% 和 59.13%，累计概率最大差为 0.85 个百分点。[掠射复测](../../../outputs/shaderball_first_hit_rays/nonstationary_render/grazing_check/summary.json)

## 平稳射线上的实现

在射线的物理弧长坐标中，用种子和有符号格点编号生成系数。基函数由高斯卷积核在间距 Δ 的格点处的值和斜率构造，相邻结点之间采用三次 Hermite 插值。支撑半径向上取整到格点，在最外层结点同时置零函数值与斜率。每个噪声格段只组装一次，跨越多个 SDF 体素时复用。

| 场景参数 | 默认值 | 含义 |
| --- | --- | --- |
| `correlation_context` | `"renewal+"` | 使用原始 Dragon 的两项端点观测和普通表面分支规则。 |
| `step_size` | 0.01 | 入口首次采样的抖动宽度，单位为世界空间距离。后续首根仍在三次段内求解。 |
| `lattice_spacing_in_ell` | 0.5 | Δ / ℓ。减小后细化基函数，每个格段要合并更多系数。 |
| `kernel_radius_in_ell` | 3 | 截断半径 / ℓ。增大后保留更多核尾部。 |
| `geometry_pruning` | true | 启用场景共用的两级 SDF 块界。关闭后仍逐段求同一三次场的首根。 |
| `geometry_pruning_mode` | `"conservative"` | 粗剪枝可选 `"conservative"` 或 `"three_sigma"`，对平稳和非平稳核均生效。 |

在理想独立高斯系数模型中，有限维协方差直接等于基函数权重的内积。碰撞后，按起点场值和出射方向导数对起点附近的系数做线性投影。平稳核的两行权重分别为偶函数和奇函数，内积为零，只需两个预先计算的标量归一化因子。

命中点的轴向导数取自实际求根的三次式。平稳核的两个横向梯度分量各自具有方差 K(t,t)/ℓ²，只在需要法线时抽样。

## 路径记忆

平稳和非平稳场景默认使用 `renewal+`。一次更新只把场值 F(p) 和沿新方向的导数 ∇F(p)·ω 传给一组新格点系数。下一个命中点的横向梯度在新段上采样。

| 路径事件 | 下一段的起点 |
| --- | --- |
| GPIS 命中后继续追踪或直接连光 | F(p)=0，导数取该分支的碰撞梯度沿出射方向的投影。 |
| 普通网格反射，同一介质内主路径继续 | 继承上一段端点的实际场值和梯度投影。 |
| 从普通网格直接采光，包括采样光源和采样 BSDF 连灯 | 按原始 Dragon 的直接光约定，两个分支各自从新段开始。 |
| 同一介质内透明边界直穿 | 继续原有的格点实现和累计弧长。尚未到达的入口探点也沿用原位置。 |
| 首次进入，或离开后再次进入介质 | 直接采样无条件格点场，在首个抖动探点判断是否已经位于负值区。 |

对单位方向，在 `nearT + U × step_size` 处第一次检查场值，U 在 [0,1) 上均匀采样。探点落在射线终点之前且 F<0 时，直接在探点记一次几何命中并遮挡阴影。F≥0 时，从探点起查找首次下穿。探点前的区间不参与求根。如果普通表面在探点前就使路径转向，新方向继承端点观测并重新抽取入口探点。GPIS 碰撞后的出射段继续使用 F=0 和出射导数。

若负值探点处的场值沿入射方向增大，该点是背面。它照常计入几何首碰并遮挡阴影。与启用 NEE 的 bounded Dragon 一样，铜和理想镜面可沿实际场梯度反射，铜使用带符号的入射余弦计算 Fresnel 权重，背面部分颜色通道可超过 1。直接采样光源时跳过这类背面连接。Lambert 材质保持单面散射。

背面镜面反射后，`renewal+` 保留 F=0 和负的出射方向导数。沿该方向在首个抖动探点检查场值，仍为负就在那里再次命中。透明边界前后的两段共用同一个探点。本文已有 shaderball 图像和计时来自这些路径规则调整之前的版本。

Dragon 的两处地板像素各取 9 条通向面光源的方向，每个方向采样 8192 次。两种方法共用烘焙的三线性 SDF。`step_size=0.01` 时，像素 (432,432) 在首步内的加权命中率为 48.320%，bounded 为 48.245%。像素 (592,400) 分别为 2.309% 和 0.986%。[首碰分布图](../../../outputs/dragon_renewal_plus/entry_change/first_hit_cdf.png) · [逐点统计](../../../outputs/dragon_renewal_plus/entry_change/first_hit_summary.json)

接入背面散射后，用原始相机的 32×32 裁剪比较平均线性亮度。龙身下方 (592,400) 的 512 spp 结果由 0.11146 变为 0.12335，bounded 为 0.12450。另一处 (432,432) 的 128 spp 结果由 0.34442 变为 0.37032，bounded 为 0.35690。[地板裁剪对照](../../../outputs/dragon_renewal_plus/backface_alignment/floor.png) · [统计与首碰重放](../../../outputs/dragon_renewal_plus/backface_alignment/summary.json)

## 判空与首根

共用几何层级的两级块分别覆盖每轴 8 个和 16 个体素。节点下界取自所有子体素角点。默认的 `conservative` 模式用噪声幅值的保守界做第一级测试。实际 Box–Muller 生成器的径向均匀数最小为 2⁻³²，因此生成的普通系数绝对值至多为 6.67。把它与已条件化系数的实际最大绝对值取大，再乘预计算的基函数界。这个常数来自有限位宽生成器；理想高斯变量在实数轴上有无界支撑。

`three_sigma` 模式在节点均值下界大于目标核的全场最大标准差的三倍时跳到节点出口。表面出射等已条件化光线也使用这个阈值。它可能跳过尾部实现的首次下穿，使报告的碰撞变晚或整条射线漏碰。可用相同的场景、射线和种子与默认模式逐条比较。

粗界无法跳过的地方再组装当前噪声格段，利用四个 Bézier 控制值收紧噪声界。在候选体素中把三线性 SDF 限制到光线，与噪声三次式相加。总场两端同号时也会检查内部极值。相切后保持原符号的点略过，共享边界上的零由右侧段判定是否发生下穿。

噪声权重、系数缓存与点积使用 float，体素坐标、三次式及首根使用渲染器已有的 double。单调区间内使用带区间保护的 Newton 迭代，两次采样之间无需矩阵分解。默认配置在一个格点周期取 65 个相位、从 0 到 6ℓ 取 601 个距离，测得协方差最大绝对误差约为 0.000954σ²。这项采样检验刻画表示误差。对单条射线的首根保证只适用于所构造的分段三次场及其浮点精度。

在平坦三重根处，首根仍按符号变化判定。着色时将入射速度的绝对值下限设为 double 的机器精度乘以 σ/Δ，以便计算镜面反射的方向密度。

## Shaderball 实测

i9-13900HX 的 8 个物理性能核，256 × 256，128 spp，种子 20261024。场景使用 ℓ = 0.2、σ = 0.02，三项对照沿用同机同配置的历史单次运行。以下时间单位均为秒。

| medium | 纯渲染 | 总耗时 |
| --- | ---: | ---: |
| lattice_cubic_gpis | 37.653 | 73.396 |
| classic_global | 38.628 | 71.275 |
| global_conditional | 95.870 | 132.160 |
| bounded-gpis，步长 0.0025 | 408.958 | 447.393 |

新 medium 与 bounded-gpis 的线性 RGB 均值分别为 0.195265 和 0.195271，两张图之间的线性 RGB RMSE 为 0.04052。这项差异包含两种表示各自的蒙特卡洛采样噪声。[并排查看图像及详细计时](../../../outputs/shaderball_l020_s002_step00025_8t/comparison.html)

另以同一种子渲染 64 × 64、8 spp，开启与关闭几何剪枝分别用时 0.171 秒和 0.979 秒，两份 PFM 逐字节一致。开启剪枝的加载与初始化多用约 2.08 秒，这笔一次性开销计入总耗时。[剪枝测试记录](../../../outputs/shaderball_l020_s002_step00025_8t/lattice_cubic_gpis/pruning_ab/summary.json)
