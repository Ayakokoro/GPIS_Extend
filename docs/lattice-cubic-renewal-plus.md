# 固定格点三次核 GPIS：一条光线的采样、收敛与 3σ 剪枝

一条光线只抽一组随机数。把它们放在间距固定的格点上，每个格点用一个随距离衰减的函数影响周围。两个位置用到相同的随机数，它们的场值就会相关。选择这个函数的形状，可以控制相关性；把函数做成分段三次式，可以检查整段有没有碰撞。随机数按格点编号生成，因此可以先查远处，再回头细查近处，得到同一条实现。

下面先讨论保守档。这里的“保守”指剪枝对**已经选定的离散场**有界。离散场的协方差与目标核的差距、渲染器的首次探针约定，是另外两个可以分别检验的问题。

## 1. 一条光线要采样什么

沿用原推导，空间场为 $f(\mathbf x)=\mu(\mathbf x)+\psi(\mathbf x)$，其中 $\mu$ 是均值，$\psi\sim\operatorname{GP}(0,\kappa)$ 是协方差为 $\kappa$ 的零均值高斯过程。光线起点为 $\mathbf x_0$，单位方向为 $\boldsymbol\omega$，$s$ 表示世界空间弧长。记

$$
\mathbf x(s)=\mathbf x_0+s\boldsymbol\omega,\qquad
F(s)=f(\mathbf x(s)),\qquad m(s)=\mu(\mathbf x(s)),\qquad
k(s,r)=\kappa(\mathbf x(s),\mathbf x(r)).
$$

在导数存在处，$G(s)=F'(s)=\boldsymbol\omega^\top\nabla f(\mathbf x(s))$。起点场值、完整梯度和沿出射方向的梯度分别记作 $f_0,\mathbf g_0,g_{0,\parallel}=\boldsymbol\omega^\top\mathbf g_0$。原推导区分

$$
C_0^{\rm 3D}=\{f(\mathbf x_0)=f_0,\ \nabla f(\mathbf x_0)=\mathbf g_0\},
\qquad C_0=\{F(0)=f_0,\ G(0)=g_{0,\parallel}\}.
$$

取 $F>0$ 为可传播侧。对于 $f_0>0$，或 $f_0=0$ 且 $g_{0,\parallel}>0$，理想目标是排除起点的首次零点

$$
T=\inf\{s>0:F(s)=0\},\qquad \inf\varnothing=\infty .
$$

若内部零点都是简单穿越，第一个零点就是第一次下穿。令 $E_{(0,t)}=\{F(s)>0,\ 0<s<t\}$，则 $\{T>t\}=E_{(0,t)}\cap\{F(t)>0\}$。给定 $C_0$ 后，逐条采样完整实现并选其**最早**下穿，就自动包含“此前一直没有碰撞”的历史。单独在终点采一个 $F(t)=0,G(t)<0$ 会把第二次、第三次下穿也计入。对独立实现 $i=1,\ldots,N$，可用 $N^{-1}\sum_i\mathbf1\{T_i>t\}$ 估计对应的存活概率 $\mathcal T(t)$。渲染器使用首探针时，将这里的 $T_i$ 换成第 3 节定义的实际命中距离。[Notion 原推导 §1.1–1.5](https://app.notion.com/p/418eab979d1381dab0b3fdd19e2748e8)

## 2. 保守档如何构造一条平稳 SE 光线

### 2.1 先选卷积核，再把白噪声放到格点上

沿用论文的 $h$ 表示卷积核，$\kappa$ 和 $k$ 表示协方差。设 $\ell_{\rm corr}>0$ 是沿线相关长度，$\sigma>0$ 是目标场值的标准差。Notion 的平稳空间 SE 使用精度矩阵 $A$，给出 $\gamma^2=\boldsymbol\omega^\top A\boldsymbol\omega$ 和 $\ell_{\rm corr}=1/\gamma$；当前平稳实现支持各向同性 SE。我们希望得到

$$
k(s,r)=\sigma^2\exp\!\left[-\frac{(s-r)^2}{2\ell_{\rm corr}^2}\right].
$$

取如下卷积因子。把两个指数相乘后配方，就得到右侧恒等式：

$$
h(v)=\sigma\left(\frac{2}{\pi\ell_{\rm corr}^2}\right)^{1/4}
e^{-v^2/\ell_{\rm corr}^2},\qquad
\int_{\mathbb R}h(s-u)h(r-u)\,du=k(s,r).
$$

具体地，$(s-u)^2+(r-u)^2=2(u-(s+r)/2)^2+(s-r)^2/2$。若 $W$ 是一维高斯白噪声，$\int h(s-u)\,dW(u)$ 因此具有目标协方差。宽度为 $\Delta$ 的不相交小区间上的白噪声积分相互独立，方差都是 $\Delta$。所以按论文的位置和权重记号，格点取 $q_j=j\Delta$，对应权重取 $w_j=\sqrt{\Delta}\,Z_j$，其中 $j\in\mathbb Z$，$Z_j$ 独立且服从 $\mathcal N(0,1)$。这解释了权重中为什么是 $\sqrt{\Delta}$。[Chen 等，§3.2](https://cchen-77.github.io/assets/papers/bounded-gpis.pdf)

### 2.2 让同一组系数生成整段三次式

设 $R$ 是指定的物理支撑半径，$J=\lceil R/\Delta\rceil$。在 $|i|<J$ 的结点 $i\Delta$ 记录 $h(i\Delta),h'(i\Delta)$；在 $i=\pm J$ 将值和斜率都设为零，支撑外也取零。相邻结点以三次 Hermite 多项式连接，所得模板记为 $p_{\Delta,R}$。Hermite 的含义很简单：一段三次式有四个系数，刚好可以匹配两端的两个值和两个斜率。相邻段也因此有相同的结点值和斜率。

定义第 $j$ 个系数对位置 $s$ 的权重 $b_j(s)=\sqrt{\Delta}\,p_{\Delta,R}(s-q_j)$。以下把固定 $\Delta,R$ 的场记成 $F_\Delta$，其协方差记成 $k_\Delta$：

$$
\boxed{\quad F_\Delta(s)=m(s)+\sum_j b_j(s)Z_j,\qquad
k_\Delta(s,r)=\sum_j b_j(s)b_j(r).\quad}
$$

这两个等式是有限模型的定义和精确协方差。任意有限组场值都是同一个高斯向量的线性变换，因此联合高斯。两点没有共用系数时，噪声相互独立。在有限格距下，$k_\Delta(s,s)=\sum_j b_j(s)^2$ 可以随 $s$ 在一个格距内的位置而变化；程序没有强制逐点方差等于 $\sigma^2$。默认参数为 $\Delta=0.5\ell_{\rm corr}$、$R=3\ell_{\rm corr}$。这个 $3\ell_{\rm corr}$ 是**空间距离**，与第 6 节的场值阈值 $3\sigma$ 单位不同，两档都使用核的有限支撑。[模板和实际权重](../src/core/math/LatticeGaussianNoise.hpp#L34) · [默认参数](../src/core/media/LatticeCubicGpisMedium.hpp#L10)

### 2.3 在起点施加一次 renewal+ 条件

起点条件只涉及 $b_j(0)$ 或 $b'_j(0^+)$ 非零的有限个格点。把这些格点的 $Z_j$ 排成列向量 $Z$，把对应的两行权重排成矩阵 $D$。设 $y$ 是扣除均值后的观测残差：

$$
D_{1j}=b_j(0),\qquad D_{2j}=b'_j(0^+),\qquad
y=\begin{pmatrix}\Delta f_0\\\Delta g_{0,\parallel}\end{pmatrix}
=\begin{pmatrix} f_0-m(0)\\g_{0,\parallel}-m'(0^+)\end{pmatrix}.
$$

三线性 SDF 在体素交界的斜率可以不连续，故出生处取新光线的右导数。若 $DD^\top$ 可逆，抽取先验系数后做一次更新：

$$
\boxed{\quad Z^{c}=Z+D^\top(DD^\top)^{-1}(y-DZ).\quad}
$$

令 $P=D^\top(DD^\top)^{-1}D$。因 $P=P^\top=P^2$，有 $DZ^c=y$，且 $Z^c$ 的均值为 $D^\top(DD^\top)^{-1}y$，协方差为 $I-P$。这正是 $Z\mid DZ=y$ 的高斯条件分布。其余格点与观测独立，仍按同一随机种子和编号生成。平稳对称模板在格点原点的值行是偶函数、斜率行是奇函数，两行内积为零；非平稳实现则先将斜率行对值行正交化。首段另有一个只补偿舍入残差的 Hermite 修正，到下一结点时其值和斜率都为零。[平稳更新](../src/core/math/LatticeGaussianNoise.hpp#L138) · [非平稳更新](../src/core/math/NonstationaryLatticeGaussianNoise.hpp#L149) · [出生处修正](../src/core/media/LatticeCubicGpisMedium.cpp#L97)

可用原推导的连续 SE 后验检查实现。记 $\rho(s)=e^{-\gamma^2s^2/2}$，则目标为

$$
\mu_{F\mid0}(s)=m(s)+\rho(s)(\Delta f_0+s\Delta g_{0,\parallel}),\quad
k_{F\mid0}(s,r)=\sigma^2\!\left[e^{-\gamma^2(s-r)^2/2}
-\rho(s)\rho(r)(1+\gamma^2sr)\right].
$$

在该空间 SE 下，$C_0^{\rm 3D}$ 与 $C_0$ 给出相同的**沿线场值**条件分布。有限格点使用自己的 $k_\Delta$ 做条件化；其精确后验一般要到细化后才趋向上式。数学上的精确高斯性以独立正态系数和实数运算为前提，渲染器用有限位伪随机数实现这个采样模型。[Notion 原推导 §1.4](https://app.notion.com/p/418eab979d1381dab0b3fdd19e2748e8)

## 3. 保守档如何求首次命中

**先决定是否延续旧光线。** 连续直穿、方向相同的透明分段继承同一自由程的种子、格点原点和累计弧长。GPIS 散射产生新的自由程和种子，沿新方向保留 $f_0=0$ 和 $g_{0,\parallel}=\boldsymbol\omega^\top\mathbf g_0$。普通地板主路径的 BSDF 续射用旧实现的实际端点场值和投影梯度条件化；地板的直接光连接分支另起一条自由程。无继承观测的外部光线从先验开始。这个规则规定了不同自由程之间共享的信息。[自由程与端点](../src/core/media/LatticeCubicGpisMedium.cpp#L223) · [GPIS 后继状态](../src/core/media/ClassicGlobalFpGradientMedium.cpp#L17) · [普通表面的直接光分支](../src/core/integrators/TraceBase.cpp#L41)

**先执行渲染器的入口约定。** 对安排首探针的自由程，例如外部出生或出射导数为负的 GPIS 出生，令 $a$ 是近端的非负弧长，$d_{\rm step}$ 是场景的首探步长，$U$ 是均匀随机数，取 $s_p=a+Ud_{\rm step}$。内向出生若得到零偏移且 $s_p=0$，改用 $d_{\rm step}$。只在探针落入有效射线窗口时检查。令 $T_\downarrow(b)$ 表示在本次窗口中从 $b$ 开始找到的最早正到负穿越，找不到取 $\infty$。实际报告的距离为

$$
T_{\rm probe}=
\begin{cases}
s_p,& F_\Delta(s_p)<0,\\
T_\downarrow(s_p),& F_\Delta(s_p)\ge0 .
\end{cases}
$$

公式中的距离从自由程的出生点量起。若探针达到或超过本段出口，本段报告无命中，同方向透明续射继承待检查的探针。若首探针为负，它可以位于内部，且斜率可以为正。这样的几何命中若产生后继 GPIS 自由程，后继仍按表面观测 $f_0=0$ 出生。出射到正侧、无需首探针的 GPIS 自由程直接搜索下一次下穿，忽略出生零点。保守性覆盖实际搜索窗口，入口至首探针之间可能已有一对穿越。[探针和背面处理](../src/core/media/LatticeCubicGpisMedium.cpp#L274)

**先用整个块的界跳过空区。** 设 $V$ 是空间块，$I_V$ 是光线在其中的区间，$\mu_{\rm lower}(V)$ 是所含细体素角点均值的最小值，实际计算另留舍入余量。三线性权重非负且和为一，故 $m(s)\ge\mu_{\rm lower}(V)$。若 $B$ 是条件化后噪声在 $I_V$ 上的绝对值上界，则

$$
\mu_{\rm lower}(V)>B\quad\Longrightarrow\quad
F_\Delta(s)\ge\mu_{\rm lower}(V)-B>0 \quad(s\in I_V).
$$

一个确定性构造如下。设 $M$ 界住本条有限光线上所有相关的 $|Z_j^c|$，设 $\beta_{ijk}$ 是第 $i$ 个噪声段上基函数 $b_j$ 的第 $k$ 个三次 Bernstein 控制值，$k=0,1,2,3$。四个 Bernstein 基函数非负且和为一，于是可取

$$
B=M\max_{i,k}\sum_j|\beta_{ijk}|.
$$

平稳模板只需预计算一种格段。代码的 Box–Muller 使用 $U\ge2^{-32}$，故先验系数绝对值至多 $\sqrt{-2\log 2^{-32}}<6.67$；再把 $M$ 提高到覆盖已更新的条件系数。程序为舍入和出生段残差另留余量。理想正态系数无有限的确定性全局上界，但对有限光线、紧支撑模板，可以取其有限组**已抽样**系数绝对值的最大值。非平稳代码由参数及其导数的上界包住两端的 Hermite 控制值。实现使用 $8^3$ 体素的宏胞及其 $16^3$ 父块。[平稳系数界](../src/core/math/LatticeGaussianNoise.hpp#L69) · [非平稳系数界](../src/core/math/NonstationaryLatticeGaussianNoise.hpp#L37) · [几何块下界](../src/core/media/GpisSdfGrid.cpp#L596)

**剩下的区间直接求根。** 在一个噪声段内先组成其三次多项式，四个实际控制值的最大绝对值给出更紧的噪声界。再按 SDF 体素边界切段。体素内的三线性均值由 $1,x,y,z,xy,xz,yz,xyz$ 的线性组合给出；直线上的 $x,y,z$ 都是弧长的一次式。因此每个交段 $[a,b]$ 上的**总场**恰为

$$
F_\Delta(a+(b-a)u)=c_0+c_1u+c_2u^2+c_3u^3,\qquad 0\le u\le1.
$$

它的四个 Bernstein 控制值为 $c_0,\ c_0+c_1/3,\ c_0+(2c_1+c_2)/3,\ \sum_{k=0}^3c_k$。全部严格为正即整段为正。其余候选段用导数 $c_1+2c_2u+3c_3u^2=0$ 的内部根分成单调区间，从左到右找最早下穿，以带括区间的 Newton 迭代定位，必要时取中点。共享结点交给右段判定穿越方向。两端都为正也能发现中间的负槽；恰好碰零又返回正侧的切触按当前约定不算下穿。这里的三次式是所渲染场的精确分段表示，数值求根仍有普通浮点误差。[平均场沿线多项式](../src/core/media/GpisSdfGrid.cpp#L628) · [求根实现](../src/core/math/CubicDowncrossing.hpp#L28)

**若该命中继续散射，沿线斜率要与碰撞实现一致。** 令 $P_\perp=I-\boldsymbol\omega\boldsymbol\omega^\top$ 是垂直于光线的投影。在平稳各向同性分支，$G_\Delta(t)=F'_\Delta(t)$ 取自本次碰撞的沿线实现，$k_\Delta(t,t)$ 取自离散先验。取横截面的正交单位基 $\mathbf e_1,\mathbf e_2$ 和两枚独立标准正态 $\xi_1,\xi_2$，用下式补齐横向梯度。

$$
\mathbf g_1=\boldsymbol\omega G_\Delta(t)+P_\perp\nabla\mu(\mathbf x(t))
+\frac{\sqrt{k_\Delta(t,t)}}{\ell_{\rm corr}}
(\xi_1\mathbf e_1+\xi_2\mathbf e_2).
$$

程序在近乎平坦的散射事件上给纵向速度设极小的数值下限，并对负探针按背面散射规则处理。非平稳分支的横向均值还计入局部幅值和尺度的空间导数。这是从沿线实现生成散射梯度的额外模型；单凭一维 $k(s,r)$ 无法唯一确定完整三维梯度的联合分布。[平稳横向方差](../src/core/math/LatticeGaussianNoise.hpp#L81) · [非平稳横向梯度](../src/core/media/LatticeNonstationaryParameters.hpp#L177) · [碰撞与散射](../src/core/media/LatticeCubicGpisMedium.cpp#L347)

## 4. 在什么条件下能逼近目标 GP

### 4.1 任意连续平稳核的有限维收敛

**定理。** 令 $k(s,r)=k_0(s-r)$，其中 $k_0$ 是一维实值连续正半定协方差，$k_0(0)<\infty$。固定任意有限射线长度 $L$。允许按目标核和精度选择模板，则存在格距 $\Delta_n\to0$ 和紧支撑、分段三次的平移模板 $p_n$，使

$$
\sup_{s,r\in[0,L]}
\left|\Delta_n\sum_{j\in\mathbb Z}p_n(s-j\Delta_n)p_n(r-j\Delta_n)
-k_0(s-r)\right|\longrightarrow0.
$$

因此，理论上用独立正态系数生成的 $F_n$ 在任意有限组位置上的联合分布，收敛到均值为 $m$、协方差为 $k$ 的 GP。下面给出证明。它也包括带谱原子的周期协方差。

**第一步，把目标协方差写为卷积因子的内积。** 按 Bochner 定理，存在有限对称的非负谱测度 $\nu$，使 $k_0(v)=\int e^{i\lambda v}\,\nu(d\lambda)$，$\lambda$ 表示频率。一般的 $\nu$ 可以含原子。先用方差 $\varepsilon^2$ 的正态密度平滑它，得到谱密度 $d_\varepsilon(\lambda)$。相应协方差是

$$
k_\varepsilon(v)=k_0(v)e^{-\varepsilon^2v^2/2},\qquad
\sup_{|v|\le L}|k_\varepsilon(v)-k_0(v)|
\le k_0(0)(1-e^{-\varepsilon^2L^2/2}).
$$

以 $\widehat h(\lambda)=\int h(v)e^{-i\lambda v}dv$ 为傅里叶变换约定，令 $\widehat h_\varepsilon=\sqrt{2\pi d_\varepsilon}$。它是平方可积的实偶函数。Plancherel 恒等式给出一个实值 $h_\varepsilon\in L^2(\mathbb R)$，满足

$$
k_\varepsilon(s-r)=\int h_\varepsilon(s-u)h_\varepsilon(r-u)\,du .
$$

若目标本来就有谱密度，这一步可直接取该密度的平方根。平滑谱的作用是让同一证明也覆盖只有离散频率的核。[Bochner 谱表示与协方差，§4.2](https://gaussianprocess.org/gpml/chapters/RW4.pdf)

**第二步，控制有限支撑的误差。** 令 $\|\cdot\|_2$ 表示在 $\mathbb R$ 上的平方积分范数。可用光滑紧支撑函数 $q$ 任意逼近 $h_\varepsilon$。记 $q_s(u)=q(s-u)$、$h_{\varepsilon,s}(u)=h_\varepsilon(s-u)$，$\langle\cdot,\cdot\rangle$ 表示积分内积。由 Cauchy–Schwarz 不等式，

$$
\sup_{s,r}\left|
\langle q_s,q_r\rangle-\langle h_{\varepsilon,s},h_{\varepsilon,r}\rangle
\right|
\le (\|q\|_2+\|h_\varepsilon\|_2)\|q-h_\varepsilon\|_2 .
$$

**第三步，控制格点和三次插值的误差。** 先固定这个光滑紧支撑的 $q$。函数 $q(s-u)q(r-u)$ 在 $s,r\in[0,L]$ 时有共同的紧支撑区间，且对 $u$ 一致连续。因此，以 $\Delta$ 为步长的黎曼和对所有 $s,r$ 一致收敛。记 $H_\Delta q$ 为使用结点真值和真导数的三次 Hermite 插值。对结点区间 $[a,b]$ 和其中的 $v$，插值余项为

$$
q(v)-(H_\Delta q)(v)=\frac{q^{(4)}(\xi)}{4!}(v-a)^2(v-b)^2,\qquad
\|q-H_\Delta q\|_\infty\le\frac{\Delta^4}{384}\|q^{(4)}\|_\infty ,
$$

其中 $\xi\in(a,b)$，$b-a=\Delta$。插值后的支撑最多向两端各延长一个格距；只有 $O(1/\Delta)$ 项非零，每项在协方差和中都乘 $\Delta$。故以 $H_\Delta q$ 替换 $q$ 的协方差误差也一致趋零。

对于第 $n$ 次近似，依次选 $\varepsilon_n$、$q_n$、$\Delta_n<1/n$，让上述三步各自的误差小于 $1/n$，并取 $p_n=H_{\Delta_n}q_n$，便证明了协方差的一致收敛。对任意有限组位置，记其均值向量为 $\mathbf m$、第 $n$ 档协方差矩阵为 $K_n$、极限为 $K$。对任意同维实向量 $z$，高斯特征函数满足

$$
\exp(iz^\top\mathbf m-\tfrac12 z^\top K_nz)
\longrightarrow
\exp(iz^\top\mathbf m-\tfrac12 z^\top Kz).
$$

这就证明了联合分布收敛。**普适性来自可以为每个目标核重新选择卷积因子。** 当前程序内建的是 SE 因子；例如把其格距调小，仍不会自动得到有振荡相关性的周期核。

**落实到当前 SE 模板。** 对任意固定 $\sigma>0,\ell_{\rm corr}>0$，第 2 节已经给出精确的 $h$。需要让 $\Delta/\ell_{\rm corr}\to0$，同时使截尾误差消失。一个具体的充分选择是 $\eta=\Delta/\ell_{\rm corr}\to0$，并令

$$
J=\left\lceil\frac{1+\sqrt{2\log(1/\eta)}}{\eta}\right\rceil,\qquad
R=J\Delta,\qquad 0<\eta<1 .
$$

于是 $R/\ell_{\rm corr}\to\infty$。在最后一格强制端点归零所引入的斜率权重误差至多为常数乘 $(\sigma/\ell_{\rm corr})e^{-((J-1)\eta)^2}/\sqrt{\eta}$，因为 Hermite 导数对端点值的系数为 $O(1/\Delta)$，而单个格点带 $\sqrt\Delta$ 权重。每侧只有一个这样的过渡格，且该指数至多为 $\eta^2$。其余被删项的值、导数的平方和由高斯尾和控制并趋零。未截断部分是高斯函数及其导数的黎曼和与 Hermite 插值，两者都收敛。由此不仅值，连同起点斜率也可收敛。这个选择用来说明**存在**细化途径，默认的固定半径与格距是一档有限近似。理论极限按实数模型讨论，有限位生成器、索引上限和浮点运算另计。

### 4.2 导数、条件化和首次碰撞各需要什么

**值和斜率。** 对一般平稳核，一个充分条件是 $\int\lambda^2\,\nu(d\lambda)<\infty$，即目标有均方一阶导数。此时第 4.1 节可同时逼近卷积因子及其一阶导数，具体取 $\|q-h_\varepsilon\|_2+\|q'-h'_\varepsilon\|_2\to0$。谱平滑后 $k_\varepsilon=k_0e^{-\varepsilon^2v^2/2}$ 的前两阶导数在紧区间分别一致收敛到 $k'_0,k''_0$。对每个固定光滑 $q$，$(H_\Delta q)'\to q'$ 一致，值与导数的全部协方差和同样是收敛的黎曼和。这样可以同时得到 $k_n,\partial_rk_n,\partial_s\partial_rk_n$ 的局部一致收敛。

在有限位置上，将待估变量记为 $X$、起点观测向量记为 $Y$，并用 $K_{XY}$ 等记联合协方差的相应矩阵块。若目标 $K_{YY}$ 可逆，则由矩阵求逆的连续性，高斯条件均值和协方差

$$
\mathbf m_X+K_{XY}K_{YY}^{-1}(y_{\rm obs}-\mathbf m_Y),
\qquad K_{XX}-K_{XY}K_{YY}^{-1}K_{YX}
$$

也随细化收敛。这里 $y_{\rm obs}$ 是实际的场值和斜率观测。在升高精度时必须逼近目标核的导数；仅检验 $k_n\to k$ 不能推出梯度条件化正确。若目标本身没有均方导数，例如指数协方差 $k_0(v)=\sigma^2e^{-|v|/\ell_{\rm corr}}$，值过程仍可按定理逼近，但这样的 $G(0)$ 观测没有同样的数学定义。

**整条轨迹与首根。** 在 $s\in[0,L]$ 上，以上的值和导数协方差收敛给出 $\sup_{n,s}\operatorname{Var}(F'_n(s))\le C$，其中 $C$ 是有限常数，导数按光滑段理解。令 $X_n=F_n-\mathbb EF_n$。由积分和 Cauchy–Schwarz 不等式，

$$
\mathbb E|X_n(s)-X_n(r)|^2\le C|s-r|^2,\qquad
\mathbb E|X_n(s)-X_n(r)|^4\le 3C^2|s-r|^4 .
$$

后式用到中心高斯变量的四阶矩。这给出连续路径空间 $C[0,L]$ 的紧性，结合有限维收敛得到路径依分布收敛。固定可逆观测后，条件协方差减小，条件均值由高斯回归一致收敛，论证同样适用。确定性均值在这里取连续函数，三线性 SDF 满足该条件。

若搜索从 $a$ 开始且 $F(a)>0$，首次穿越为 $\tau\in(a,L)$，以下是首根稳定的充分条件。对任意足够小的 $\delta>0$，

$$
\min_{s\in[a,\tau-\delta]}F(s)>0,\qquad
\min_{s\in[\tau,\tau+\delta]}F(s)<0 .
$$

与 $F$ 一致足够接近的连续曲线，前一段必定为正，后一段必定到达负侧，首次接触因此被夹在 $[\tau-\delta,\tau+\delta]$。对光滑的简单下穿和具有孤立零点的近似曲线，同一结论适用于下穿距离。整段严格为正的未命中也稳定。若这些条件几乎必然成立，连续映射定理给出首碰距离依分布收敛；相切、终点刚好碰零等路径要另作处理。在 $F(0)=0,G(0)>0$ 的出生点，可要求出生邻域内方向导数也一致收敛，从而有一段共同的正侧区间，再用上述证明。固定非零首探步长时，能比较的是**相同首探约定**的分布；如果目标是从零弧长开始的连续首次穿越，还应使首探偏移趋零，并处理出生侧条件。

## 5. 非平稳核怎样处理

### 5.1 当前实现的目标

让卷积因子 $h(s,u)$ 同时依赖查询弧长 $s$ 和噪声源位置 $u$。若 $\sigma_s=\sigma(\mathbf x(s))$ 是局部场值标准差，$\ell_s=\ell_{\rm corr}(\mathbf x(s))>0$ 是局部相关长度，当前一维构造的无截尾因子为

$$
h(s,u)=\sigma_s\left(\frac{2}{\pi\ell_s^2}\right)^{1/4}
\exp\!\left[-\frac{(s-u)^2}{\ell_s^2}\right].
$$

每个结点仍使用相同索引的 $Z_j$。代码取 $\Delta=\mathtt{lattice\_spacing\_in\_ell}\,\min\ell_s$，其中最小值来自场景参数界。令 $d$ 为配置的无量纲支撑半径，$z=|s-q_j|/\ell_s$，$w_d(z)$ 为平滑降到零的窗口。写 $w=\min(0.5,d/2)$，则当 $z\le d-w$ 时 $w_d=1$，当 $z\ge d$ 时 $w_d=0$，过渡带上 $w_d=v^2(3-2v)$、$v=(d-z)/w$。结点处的权重和斜率取

$$
\widetilde b_j(s)=\sqrt\Delta\,h(s,q_j)\,w_d(z),\qquad
\widetilde b'_j(s)=\frac{d}{ds}\big[\sqrt\Delta\,h(s,q_j)\,w_d(z)\big].
$$

这个全导数包含位置移动、$\sigma_s'$、$\ell_s'$ 三项影响。将规则格点与参数场的斜率折点合在一起切段，在左右两端取段内一侧的导数，按第 2 节做 Hermite 插值。由于插值是线性的，先求各端点的噪声值和斜率再插值，等价于对每个 $\widetilde b_j$ 插值后再求和。于是每段仍为三次，沿线条件化公式和首根算法也继续适用。当前支持局部各向同性 SE，参数场为支持的常量或单轴渐变，有已知正的尺度下界、有限幅值及导数界。[非平稳核](../src/core/math/NonstationaryLatticeGaussianNoise.hpp#L64) · [参数支持范围](../src/core/media/LatticeNonstationaryParameters.hpp#L111) · [沿线切段](../src/core/media/LatticeCubicGpisMedium.cpp#L65)

这个实现细化并去掉截尾后，究竟逼近哪个核，可以直接积分验证。令 $Q(s,r)=2\ell_s\ell_r/(\ell_s^2+\ell_r^2)$。两个不同宽度的高斯相乘再配方给出

$$
\boxed{\quad k_{\rm 1D}(s,r)=\int h(s,u)h(r,u)\,du
=\sigma_s\sigma_r\,Q(s,r)^{1/2}
\exp\!\left[-\frac{(s-r)^2}{\ell_s^2+\ell_r^2}\right].\quad}
$$

这与 bounded 源码中的标量变尺度协方差公式一致。主库还定义了从三维局部各向同性高斯卷积得到的空间核。把它限制在同一光线上，得到

$$
k_{\rm 3D|ray}(s,r)=\sigma_s\sigma_r\,Q(s,r)^{3/2}
\exp\!\left[-\frac{(s-r)^2}{\ell_s^2+\ell_r^2}\right].
$$

三维积分中两个横向方向各贡献 $Q^{1/2}$，乘起来比一维多一个 $Q$。当 $\ell_s=\ell_r$ 时两个目标相同；若 $\ell_s\ne\ell_r$，则 $0<Q<1$，一维构造的互协方差更大。**“比较当前固定格点与 bounded”以及“逼近主库指定的三维 GPIS”由此成为两个可区分的问题。** 例如若要恢复三维目标，可再引入变量 $v\ge0$ 及函数 $a_s(v)=\sqrt2\,\ell_s e^{-\ell_s^2v}$。由于 $\int_0^\infty a_s(v)a_r(v)\,dv=Q(s,r)$，用 $h(s,u)a_s(v)$ 对二维白噪声做积分，协方差就恰为 $k_{\rm 3D|ray}$。这是一个可证明的扩展，当前渲染采用上面的一维公式。[bounded 的标量核](../bounded-gpis/src/core/math/GPFunctions.cpp#L886) · [主库的三维核](../src/core/math/GPFunctions.cpp#L1760) · [Paciorek 与 Schervish，式 1](https://proceedings.neurips.cc/paper_files/paper/2003/file/326a8c055c0d04f5b06544665d8bb3ea-Paper.pdf)

### 5.2 一般非平稳核的充分条件

把目标写成 $k(s,r)=\langle h_s,h_r\rangle$，其中 $h_s$ 是查询位置 $s$ 对各个白噪声源的权重函数。用中心在 $q_j$、宽 $\Delta_n$ 且互不重叠的区间 $I_{n,j}$ 表示噪声源。若有限模型实际使用的三次权重是 $b_{n,j}(s)$，定义阶梯函数

$$
h_{n,s}(u)=\sum_j\frac{b_{n,j}(s)}{\sqrt{\Delta_n}}\mathbf1_{I_{n,j}}(u).
$$

则 $\langle h_{n,s},h_{n,r}\rangle=\sum_j b_{n,j}(s)b_{n,j}(r)=k_n(s,r)$。设 $M_h=\sup_{s\in[0,L]}\|h_s\|_2<\infty$，若

$$
e_n=\sup_{s\in[0,L]}\|h_{n,s}-h_s\|_2\to0,\qquad
\text{则}\quad \sup_{s,r}|k_n(s,r)-k(s,r)|\le (2M_h+e_n)e_n\to0.
$$

证明为对内积作差两次使用 Cauchy–Schwarz 不等式。对 $\partial_s h_s$ 也满足这样的逼近，并有可逆的起点观测协方差时，第 4.2 节的条件化结论继续成立；在已知参数折点处按给定一侧取导数。对上述正尺度、有界、分段光滑的变宽高斯，先令截尾半径足够大，使 $L^2$ 尾误差一致足够小，再细化格点、在折点切开做插值，即可使这些误差趋零。支撑过渡处的导数可分侧控制；第 4.1 节的四阶经典误差式只用于四次连续可微的模板。

对于没有已知卷积公式的**任意连续正半定核**，在有限区间上仍有一条存在性证明。Mercer 展开给出 $k(s,r)=\sum_{j\ge1}\lambda_j e_j(s)e_j(r)$，其中 $\lambda_j>0$ 为积分算子的正特征值，$e_j$ 为归一化连续特征函数，该级数一致收敛。任取误差 $\epsilon>0$，先选有限 $J$ 使级数尾的一致误差小于 $\epsilon$。对每个 $v_j(s)=\sqrt{\lambda_j}e_j(s)$，以光滑函数再以固定格点上的三次式 $p_j(s)$ 一致逼近。若 $\delta_j=\|p_j-v_j\|_\infty$，则

$$
\sup_{s,r}\left|\sum_{j=1}^Jp_j(s)p_j(r)-k(s,r)\right|
\le\epsilon+\sum_{j=1}^J(2\|v_j\|_\infty+\delta_j)\delta_j .
$$

有限和的每个 $\delta_j$ 都可以任意小，所以 $\sum_jp_j(s)Z_j$ 的协方差和有限维分布都能逼近目标。这给出了用确定性三次基和高斯系数的普适表示；不同 $p_j$ 可能横跨整条光线，无法由定理保证当前局部卷积的查询成本。若还要保留完整 $C_0^{\rm 3D}$ 对射线的影响，需要从目标空间核额外计算 $\operatorname{Cov}(F(s),\nabla f(\mathbf x_0))=\nabla_{\mathbf y}\kappa(\mathbf x(s),\mathbf y)|_{\mathbf y=\mathbf x_0}$，并将观测一并条件化。一般非平稳核没有第 2.3 节的梯度投影等价性。[Mercer 展开，§4.3](https://gaussianprocess.org/gpml/chapters/RW4.pdf)

## 6. 在保守档上加 3σ 剪枝

两档生成同一个格点模型。设 $\sigma_{\max}$ 是场景给出的幅值上界，平稳时就是 $\sigma$。启用几何剪枝和 three_sigma 后，外层空间块采用

$$
\boxed{\qquad\mu_{\rm lower}(V)>3\sigma_{\max}
\quad\Longrightarrow\quad\text{直接跳到块出口。}\qquad}
$$

改变的是**决定是否访问该块的阈值**。随机系数在需要时按原分布生成，尾部值仍可出现。已进入的候选段仍由实际噪声控制值判界，并用同一个三次求根器求交。首探针的符号先由保守界证明，或直接求值，负值即命中。bounded 论文第 5.4 节同样使用 $3\sigma$ 做概率性的区域排除；固定格点当前在这一层使用场景 $\sigma_{\max}$ 和正侧判空。[固定格点的两档分支](../src/core/media/LatticeCubicGpisMedium.cpp#L286) · [Chen 等，§5.4 与偏差讨论](https://cchen-77.github.io/assets/papers/bounded-gpis.pdf)

其偏差可以写清楚。令 $\Phi$ 为标准正态的累积分布函数。对于一个预先固定的、未条件化的点，若其实际噪声标准差是 $\sigma_s>0$ 且 $m(s)>3\sigma_s$，则

$$
\mathbb P(F(s)<0)=\Phi(-m(s)/\sigma_s)<\Phi(-3)\simeq0.00135.
$$

这是**单个点、单侧**的概率。对有限格点要使用其实际标准差 $\sqrt{k_\Delta(s,s)}$。在条件射线上，若条件方差非零，单点概率为 $\Phi(-\mu_{F\mid0}(s)/\sqrt{k_{F\mid0}(s,s)})$，有限模型使用相应离散后验。丢掉一次下穿涉及的是整个被跳过区间的事件，例如

$$
\mathbb P\!\left(\inf_{s\in I_V}F_\Delta(s)<0\ \middle|\ C_0\right).
$$

单点公式不提供这个概率的同一上界。即使格距趋零并完全恢复目标核，固定 $3\sigma_{\max}$ 的跳块规则仍可能跳过尾部穿越。因此“GP 近似是否收敛”可以用保守档检验；“在固定渲染预算下，$3\sigma$ 换来多少加速和首碰偏差”则可在**同一批射线、同一随机种子**下单独比较两档。
