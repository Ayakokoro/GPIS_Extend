# Gaussian 正值域概率网络：独立实验

本实验近似 `P(Y_1 > 0, ..., Y_n > 0)`，默认支持 **1–5 个高斯变量**。
它不改变 `sparse_history_gpis`、现有二维 CDF 表或任何渲染模式。
模型权重放在 `models/gaussian-orthant-nn/`，数据与报告放在
`outputs/gaussian-orthant-nn/`，两者都已经被仓库的 `.gitignore` 忽略。

## 两种高斯，以及相机起点正号

网络输入为标准化均值 `z_i = mu_i / sqrt(C_ii)`、相关矩阵 `R` 和变量掩码。
因此相同的网络可以计算原始高斯与起点等式条件化后的高斯。条件化操作仍由
解析公式完成，网络只近似最后的积分。网络不需要额外的“是否条件化”输入。

对单位幅度、单位相关长度的 SE 核，沿射线的原始协方差为

```text
C(q,r) = exp(-(q-r)^2 / 2).
```

给定起点场值为零及完整梯度后，沿线协方差为

```text
C_H(q,r) = exp(-(q-r)^2 / 2) - (1 + q*r) exp(-(q*q+r*r) / 2).
```

均值也需要条件化。设原始无量纲均值为 `M(q)`，起点已知轴向导数为 `d0`，则

```text
mu_H(q) = M(q) - exp(-q*q/2) M(0)
          + q exp(-q*q/2) (d0 - M'(0)).
```

各向同性 SE 下，沿当前直线的场值与起点横向梯度不相关；横向梯度不会遗漏
额外的沿线场值均值项。物理单位通过 `q=t/ell`、`M=m/sigma` 转换。

相机只知道 `F0 > 0` 时，条件分布通常不再是高斯。正确的接口为

```text
P(F1>0, ..., Fk>0 | F0>0)
  = P(F0>0, F1>0, ..., Fk>0) / Phi(mu0/sqrt(C00)).
```

这里把起点作为联合概率中的一个变量；最多 5 个变量意味着还能包含 4 个其他
位置。它与从碰撞点出发的 `F0=0, G0=g0` 等式条件化不同。
这里的 `n` 是参与这一次正域概率的**总变量数**，不是内部历史点数。
例如同时计入当前位置与 3 个内部历史点时，碰撞起射是 4 维，相机起射再显式
加入起点正号则是 5 维；碰撞起点的值和梯度已由解析条件化消去。
严格大于零的确定变量、重复变量、超出训练域的输入需要显式处理，不能简单
添加 jitter 或截断均值后宣称仍在计算原分布。

## 数据和验证

数据包括原始 SE、起点值与梯度条件化 SE，以及一般正定相关矩阵。
同一随机均值剖面的原始与条件化样本属于同一个 family，先按 family 划分
训练、验证和测试集。`mode` 仅用于分组评估，不进入网络输入。

标签使用独立随机扰动的 Sobol / Genz 数值积分，并保存误差估计与采样预算。
该误差估计是统计估计，不能当作严格误差界。未通过质量门槛的标签需要拒收，
零估计不能被 clamp 成“准确的”小概率标签。

验证需分别报告原始/条件化模式、变量个数和小概率区间。CPU 单次完整调用与
GPU 批量前向的时间分开测量；GPU 的批量吞吐不代表当前 CPU 渲染器能够达到
同样的单次调用延迟。`n=1` 使用精确的一维正态 CDF，不能据此评价多点网络误差。

## 依赖

Python 3.10+，NumPy、SciPy、PyTorch。CUDA 可选。当前运行环境已安装这三项。
所有命令从仓库根目录运行；使用 `python -B` 可避免生成 `__pycache__`。

## 复现实验

```powershell
python -B -m tools.gaussian_orthant_nn.generate --groups 35000 --backend cuda --max-power 13
python -B -m tools.gaussian_orthant_nn.train --device cuda --cpu-threads 2
python -B -m tools.gaussian_orthant_nn.evaluate --device cuda
python -B -m unittest discover -s tests -p test_gaussian_orthant_nn.py -v
```

没有 CUDA 时把三个 `--backend/--device cuda` 改成 `cpu`。数值标签生成明显更慢，
可以先将 `--groups` 调小。默认 seed 固定；生成器的 `generation.json` 保存实际
采样预算、保留/拒收数量和数据覆盖范围；训练 `config.json` 记录数据哈希、网络
设置及训练/验证/测试 family 数量。`best.pt` 按验证集误差选取，测试集不用于训练
或选择 checkpoint。训练报告包含所有维数，最终 HTML 的多点汇总会排除精确的 n=1。

默认输出：

- `outputs/gaussian-orthant-nn/report.html`：可筛选模式与维数的图表、误差和速度。
- `outputs/gaussian-orthant-nn/evaluation.json`：完整分组统计、最差样本和标签复核。
- `outputs/gaussian-orthant-nn/dataset.npz`：标签、误差估计、模式、family 和 split。
- `outputs/gaussian-orthant-nn/config.json`、`history.csv`：训练配置及学习曲线数据。
- `models/gaussian-orthant-nn/best.pt`：共享模型权重。

## Python 调用

```python
import numpy as np
from tools.gaussian_orthant_nn.gaussian import (
    se_covariance, surface_conditioned_statistics,
)
from tools.gaussian_orthant_nn.infer import OrthantPredictor

network = OrthantPredictor("models/gaussian-orthant-nn/best.pt", device="cpu")
t = np.array([0.7, 1.4, 2.2, 3.2])
M = lambda q: 0.2 + 0.3*q - 0.14*q*q + 0.25*np.sin(2*q)

# 无起点等式条件化：均值和协方差直接来自原始 GP。
unconditioned = network.predict(M(t), se_covariance(t))

# 碰撞起射：先解析条件化 F0=0，起点轴向梯度为0.4。
mu, covariance = surface_conditioned_statistics(
    t, M(t), origin_mean=0.2,
    origin_directional_derivative=0.8, outward_derivative=0.4,
)
conditioned = network.predict_conditioned(mu, covariance)

# 相机起射：起点只知道为空。joint包含起点，恰好5个变量。
t_with_origin = np.r_[0.0, t]
camera = network.predict_given_positive(
    M(t_with_origin), se_covariance(t_with_origin), origin_index=0,
)
```

这里为演示取 `sigma=ell=1`，接口也接受物理单位的 `sigma` 和 `length_scale`。
`predict_log_probability` 返回 log 概率，适合小概率的乘除。普通 `predict` 在
float64 下溢时会报错并提示使用 log 接口。`predict_conditioned` 接收已经条件化的
高斯参数，不会再次对起点条件化。

命令行也可以使用报告自动生成的示例：

```powershell
python -B -m tools.gaussian_orthant_nn.infer models/gaussian-orthant-nn/best.pt outputs/gaussian-orthant-nn/examples/unconditioned.json
python -B -m tools.gaussian_orthant_nn.infer models/gaussian-orthant-nn/best.pt outputs/gaussian-orthant-nn/examples/camera_F0_positive.json --positive-origin 0
```

## 近似范围

初始网络为 30 输入、3 层各 128 个 SiLU 隐藏单元、1 个 log 概率输出；输入包括
5 个均值特征、10 个相关系数、10 个近奇异程度特征和 5 个有效变量掩码。
输出参数化满足边缘概率上界及 Frechet 下界；它们不构成神经近似的误差保证。
训练损失使用 log 概率上的 SmoothL1。

为避免变量编号影响，先按标准化均值排序并同步重排相关矩阵。完全相同均值的
ties 仍沿用输入顺序，因此不保证严格排列不变；报告额外测试这种情况。

公共推理接口检查输入与训练覆盖范围，不运行数值积分 fallback。确定变量、
单变量、重复扰动与可解析的秩一情况使用明确的解析规则。一般秩亏或无法分辨的
近奇异输入会报 `DomainError`；生成器通过 SE 特征分解能生成其中一些情况的标签，
所以报告分别列出直接网络的测试误差与公共接口的支持/拒绝数量。
`allow_out_of_domain=True` 只显式允许训练域外推，不解除不支持的秩亏限制。
训练范围检查只覆盖部分输入特征，不能识别所有分布外组合。

## 本次运行结果（2026-11-04）

固定 seed `20261104`，生成 72,985 个合格样本：训练 62,575，验证 5,293，测试
5,117。37,121 参数的网络训练 300 epoch，按验证集选择第 299 epoch。标签生成
190.60 秒，训练 142.45 秒；环境为 NumPy 2.2.6、SciPy 1.15.3、PyTorch 2.12.0+cu130，
RTX 4060 Laptop GPU。

以下仅统计 **n=2..5 的测试样本**，排除精确一维公式：

| 分布 | 测试数量 | 概率 MAE | 相对误差中位数 | 相对误差 P95 |
| --- | ---: | ---: | ---: | ---: |
| 原始 SE | 1,877 | 0.00456 | 1.43% | 13.88% |
| 起点值与梯度条件化 SE | 1,866 | 0.00380 | 0.90% | 13.68% |
| 一般 SPD | 364 | 0.00358 | 7.12% | 94.43% |

这些是直接网络的总体指标，包含公共 API 会拒绝的近奇异样本。公共 API 在
5,117 个测试样本中支持 4,893 个，另 224 个一般近奇异输入显式拒绝。
新种子、更高预算的 48 个分层标签复核全部达到精度目标，最大 log 概率差
`4.60e-4`。19 项单元测试全部通过（18 项 CPU 与 1 项 CUDA 分别运行）。

当前局限已实测，而非仅理论上的顾虑：`P<1e-8` 区间的总体相对误差 P95 约
244%，一般 SPD 中有极大的尾部相对误差；不能以小的绝对概率误差掩盖。
四维相同均值探针在不同排列下的预测概率为 `0.10556–0.11912`，表明 ties 的
排列问题尚未解决。它们均保留在报告中，不删掉困难测试样本。

单线程 CPU 完整 API 中位计时约 1.66 ms/次；GPU 4096 批量约 0.465 微秒/个，
后者仅包括设备上已有输入的特征计算与前向，不包括验证和数据传输。
相同 24 个案例相对本实验 Python/SciPy 数值参考约快 17 倍，不能推断为 C++
渲染器加速比。当前权重是概率近似实验基线，尚未验证渲染图像误差。

## 与消光的关系

普通联合正值域概率可用于消光的分母。分子仍包含当前零点条件下的正下穿
速度权重，不能直接以一个普通正域概率替换。若以后接入渲染器，可以先用解析
条件化得到每个积分节点的高斯参数，再调用网络；距离和梯度采样还需保持同一
套历史条件。本实验仅衡量概率近似的准确性和速度，不声称已加速渲染或验证图像误差。
