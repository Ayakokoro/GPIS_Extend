# 从 Tungsten 场景生成单点历史训练数据

入口是现有的 Tungsten `scene.json`。C++ 工具使用 `Scene::load`、场景的资源路径解析和渲染器的 SDF 表示，解析计算起点条件化后，再生成一个历史约束 `F(u)>0` 的消光率标签。Python 只读取已导出的数值数据，不重新解析或插值 SDF。

## 构建和使用

从项目根目录运行：

```powershell
cmake -S . -B build -DTUNGSTEN_BUILD_SINGLE_POINT_TRAINING_TESTS=ON
cmake --build build --config Release --target gpis_single_point_data single_point_history_teacher_test single_point_scene_field_test -- /m:1
ctest --test-dir build -C Release -R '^single_point_' --output-on-failure
```

上面的 `/m:1` 是当前 Windows / MSBuild 环境的参数。其他生成器可使用常规 `cmake --build ... --parallel 4`。

先检查已有场景和 SDF：

```powershell
build/Release/gpis_single_point_data.exe --scene gpis-scenes/bunny/bunny.json --output outputs/single_point_bunny_inspect --inspect
```

直接生成数据：

```powershell
build/Release/gpis_single_point_data.exe --scene gpis-scenes/bunny/bunny.json --output outputs/single_point_bunny_data --views 12 --rays-per-view 256 --queries-per-ray 16 --threads 8
python -B -m tools.single_point_training.train --data outputs/single_point_bunny_data --output models/single_point_bunny --epochs 100 --device cpu --cpu-threads 4 --correction-cap 2
```

每次使用新的输出目录，工具会拒绝覆盖已有数据。`--inspect` 只验证输入，不生成样本。完整参数见 `--help`。训练需要 NumPy、PyTorch；当前可用的 Python 为 `C:/Users/user/miniconda3/envs/gaussian_splatting/python.exe`。

## 一个场景、一个 GPIS SDF

验证同时检查顶层 `media`、`camera.medium`、几何的 `int_medium` 和 `ext_medium`。同一个命名介质被引用多次只计一次；两个独立 GPIS 介质会被拒绝，即使它们引用相同文件。零个 GPIS、GP CSG、非平稳核和各向异性核也会明确报错。第一版接受固定的各向同性 SE 核，实际解析出的 `sigma` 和 `length_scale` 写入 manifest。

支持以下现有输入：

- **OpenVDB**：`gaussian_process.mean.type=tabulated`，使用 `VdbGrid` 和 `GpisSdfGrid`。保留渲染器中的网格变换、线性插值、均值 scale/offset 和边界 clamp。场景中应关闭 `request_sdf`，因为这里读取已烘焙 SDF，不执行重新烘焙。
- **NanoVDB**：`classic_local` 或 `neural_causal_renewal` 的 `field.file`，复用 `NeuralSdfGrid`。当前要求 float、名为 `sdf`、完整活动节点覆盖、轴对齐均匀体素的网格，支持 `classic_local` 的外部 `field.transform`。其他稀疏窄带布局或原生变换会明确拒绝，避免将背景值误当作完整场。读取现有 neural 场景时不会加载模型权重。
- **程序化 SDF**：复用已有 SDF 查询接口，但必须提供 `--bounds xmin ymin zmin xmax ymax zmax`。例如当前双平面可使用 `--bounds -1.1 -0.3 -1.1 1.1 0.3 1.1`。

网格内其他 density、alpha、sigma 数据不作为训练场。SDF 有效表示及域来自原读取器；不会将 sigma 烘焙进新体积。NanoVDB 的范围和变换限制沿用该读取器，不能据此声称支持任意第三方 SDF 格式。

已有场景可以直接使用。若只有一个 NanoVDB 文件，可建立普通 Tungsten 场景：

```json
{
  "media": [{
    "name": "gp",
    "type": "classic_local",
    "field": {"file": "fields/object.nvdb", "sdf_grid": "sdf"},
    "covariance": {"type": "squared_exponential", "sigma": 0.02, "lengthScale": 0.2},
    "material": {"type": "lambert", "albedo": 0}
  }],
  "camera": {
    "type": "pinhole", "resolution": [256, 256], "fov": 40,
    "medium": "gp",
    "transform": {"position": [0, 0, 3], "look_at": [0, 0, 0], "up": [0, 1, 0]}
  }
}
```

`fields/object.nvdb` 相对于场景文件解析，不相对于运行命令的目录。这个数据工具不需要灯光或渲染输出设置。

## 数据生成的含义

第一视角使用场景相机；其余视角围绕 SDF 域生成，仍调用 Tungsten 的相机采样器。每条射线截取 SDF 域内的区间，调用与 `global_conditional` 相同的 `startConditionalExterior`，采样一次完整起点 `F0` 和梯度。所有查询共享该起点观测。

查询距离连续随机采样，一半沿区间均匀采样，一半偏向 SDF 零值带。历史候选来自已通过的 SDF 插值单元边界、中点及读取器能提供的驻点，使用现有相关性/Mills 评分选取一个；没有候选时使用 `u=t/2`。这不是 GRU 的固定推进网格。默认排除小于 `0.05*ell` 的查询年龄，历史间隔下限为 `0.01*ell`，都记录在 manifest。

这些是**局部条件分布的训练探针**，不模拟整条路径的实际存活概率，也不处理普通几何遮挡或反射后的起点分布。它们用于学习单点公式，不能直接当作渲染亮度或完整 first-passage 的标签。后续用于反射场景时需要补充相应的起点分布覆盖。

视角整体分成 train/validation/test，射线中的查询不会跨集合。使用 `--camera-only` 仅生成单相机诊断数据，不能直接作为需要三个集合的训练输入。相同 seed 和设置保持样本可复现，线程数不影响射线随机流或行顺序。

## 输出和网络

- `samples.csv`：六个特征、基础/单点消光率的对数、历史正值概率的对数、修正倍率的对数，以及视角/射线/集合编号、实际起点和查询位置。
- `manifest.json`：场景与 SDF 路径、解析出的介质配置和域、固定参数、采样规则、视角、保留行数、拒收原因及耗时。
- 出错或中断时保留 `samples.partial.csv`；只有完整且各集合非空才发布 `samples.csv`。

特征顺序固定为 `[z_x, z_u, z_v, rho_xu, rho_xv, rho_uv]`，其中 `X=F(t)`、`U=F(u)`、`V=-dF(t)/dt`，所有统计量都给定完整起点观测。数值标签使用该三变量高斯的联合正值概率和正向交叉通量；近奇异、未收敛和无法解析的小概率样本显式拒收并统计，不伪造标签。

训练器使用 `6→32→32→1` 的 ReLU MLP（1,313 个参数），输出残差 `f`，然后计算

```text
log R = -log[p_u + (1-p_u) exp(-f)]
alpha_neural = alpha_start_conditioning * R
```

末层零初始化，初始 `R=1`；输出满足 `R<=1/p_u`。标准化参数只由训练集计算。训练输出 `best.pt`、`model.json`、`config.json`、`metrics.json`、`history.csv`；最优 checkpoint 由验证集选择，测试视角只作最终评估。

新增的人工上限 `--correction-cap C` 默认是 `2`，要求有限且 `C>=1`；`--correction-cap none` 关闭人工上限。它只用于推断输出 `R_render=min(R,C)`，不会设置下限。数据标签、训练损失和最优 checkpoint 的选择都使用未加人工上限的修正。现有解析包络 `R<=1/p_u` 仍是网络结构的一部分，参与训练。

`metrics.json` 保留原始预测误差，并在每个集合的 `render_capped` 中额外报告截断后相对于原始教师的误差、预测截断数量、教师超限数量。改变人工上限不需要重新生成数据或重新训练权重；推断时必须同步使用对应的消光率和上界，不能只降低采样上界。固定历史点的区间可以使用 `M_base*min(1/p_u,C)`，一般保守选择是 `C*M_base`。

导出的模型格式更新为 `tungsten.single_point_history_mlp.v2`，将上限写入 `deployment.correction_cap`，明确标记 `inference_only`。Python 的 `model.forward(...)` 始终保留未截断输出；需要部署结果时调用 `model.inference_log_correction(..., correction_cap=C)`。原始标签和权重不因这个设置而改变。

## 渲染模式

`global_conditional_neural` 是独立于原 `global_conditional` 的实验模式。沿用原场景，只修改介质类型并添加权重路径：

```json
{
  "type": "global_conditional_neural",
  "single_point_neural": {"model": "model/model.json", "correction_cap": 2},
  "fp_gradient": {"enabled": false}
}
```

它在距离采样和阴影透射率中使用同一修正消光率，所有候选采样分支均使用 `C*M_base` 上界。碰撞点梯度分布和端点 F/梯度采样仍沿用原解析实现；当前模式拒绝同时开启 `fp_gradient`。网络推断是原生 C++，无需运行时加载 PyTorch，也不需要网络推进步长。

训练和渲染共用历史点选择代码，以模型记录的固定空间域的射线出口作为候选范围，独立于积分分段和随机候选点。`two_slabs` 额外提供解析的几何分支切换位置；这些历史候选不会改变原模式的边界或上界计算。小于 `0.05*ell` 的查询以及模型域外查询使用原解析消光率，回退和超训练特征范围次数写入日志。

可用 `--feature-transform asinh_z` 对前三个标准化均值特征做可逆 `asinh` 变换，再计算训练集均值/标准差。相关系数保持原值，没有输入截断；导出的 `feature_transform` 由 C++ 同样执行。这样可避免起点附近数值很大的特征主导标准化。

双平面复现实验入口为 `tools/surface_compare/gpis_two_slabs_neural_single_history.py`，数据与模型分别放在输出目录的 `data` 和 `model` 中。运行 `run` 会单独保存当前编译的渲染器、渲染两个种子并用本地 FLIP 比较已有参考结果。

当前训练覆盖固定场景、固定 σ/ℓ 的外部入射起点；保留视角只用于验证这一范围内的预测，不能保证其他场景或反射后的起点分布。数值教师是“完整起点观测 + 一个历史正值事件”的近似，不是 GPIS Global 的完整历史。单点近似、网络误差和人工上限的误差应分别解释。

## 继续训练

使用 `--resume 原目录/best.pt` 或 `--resume 原目录/latest.pt`，同时指定新的输出目录。数据哈希、特征顺序/变换、归一化和模型结构必须匹配；`--epochs` 在续训时表示追加轮数。

旧版检查点仅有权重，会明确记录为 `warm_start`：从其最佳轮次继续，重新初始化 AdamW、调度器和随机数状态，使用命令指定的学习率。双平面第一轮选中了 epoch 295，当时学习率为 `0.00025`。原最佳权重参与后续验证集选择，续训没有改善时仍保留它。

新版 `latest.pt` 保存当前权重、优化器、调度器、随机数状态及历史最佳权重；`best.pt` 保存最佳时刻对应的完整状态。在相同训练配置和环境中，可从这些完整检查点恢复。`metrics.json` 分别记录 `starting_epoch`、`final_epoch`、本次执行的 `epochs_completed` 和全局 `best_epoch`。

续训对比入口是 `tools/surface_compare/gpis_two_slabs_neural_continuation.py`，复用第一轮的数据和冻结渲染器，并单独保存新的权重、渲染和 FLIP。
