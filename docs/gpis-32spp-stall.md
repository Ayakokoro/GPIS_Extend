# 512×512 / 32 spp 停在 15/32 的原因和修复

## 复现与定位

当前 shaderball 场景：512×512，32 spp，spp_step=1，max_bounces=20，路径池 4096，GPU batch 1024，4 个在途批次。seed=12345，8 个 CPU worker。

旧版本稳定停在 `Completed 15/32 spp`。这一轮完成的统计为：

```text
rays: 625091
path=419203, shadow=205888
committed=262144
```

CPU 线程快照显示一个 worker 持续运行，调用栈反复经过 `PathTracer::advanceSample()`、`SparseConvolutionNoiseMedium::se1DGpuBatchConfig()` 和场景/Embree 求交；其他 worker 在等待。

从该 worker 的路径状态读到：

```text
pixel=(80,455), absolute sample index=15
origin=(0.0475450344,-0.00568206655,-0.0035218522)
direction=(-0.191235453,0.872217834,-0.450183421)
nearT=0.000500000024, farT=0
```

因此是第 16 个样本中的一条 CPU 路径陷入循环，并非 15 个样本后的 GPU 步数累计达到上限。

## 根因

射线处于 GP cube 的 +X / -Z 棱边附近，沿 -Z 面向外离开。浮点误差让起点略微超出 +X 面。

1. `Cube::intersectMediumExit()` 恢复 epsilon 之前的出口，并将临时求交标记为 backside。
2. 原来的 `Cube::intersectionInfo()` 不使用退出面信息，而根据命中位置猜最接近的面。在这个棱边位置，它选到了 +X 面。
3. 沿 -Z 的真实退出方向与 +X 法线点乘为负。`finishSurface()` 根据该法线选择了内部 GP medium。
4. 下一次求交又恢复同一个零距离出口；空 GP 区间直接结束，然后再次处理透明 cube。
5. forward 边界不增加普通反弹计数，`max_bounces=20` 无法终止此循环。
6. 路径池按顺序归并，未归并任务窗口达到 8192 后无法继续补充任务。一条循环路径最终让全帧进度停住，GPU 无新工作。

增加 spp 暴露了这个特定样本的棱边误差；32 spp 本身没有容量限制。

## 修复

`CubeIntersection` 增加 signed face axis。正常 slab 求交和 GP 出口恢复都记录实际命中的面，`intersectionInfo()` 直接使用该面生成几何法线，切线轴也跟随该法线。平行于 slab 的射线显式处理，避免 0×infinity。

新增测试覆盖：

- 六个面的出口方向必须选择外部 medium，而不只是检查 backside。
- 24 种退出面/相邻面组合，相邻面起点向外舍入一 ULP。
- 上面实际卡住的 `(80,455)` 射线。

Release CTest 4/4 通过。完整渲染结果见 `build/gpis_32spp_boundary_fixed/report.json`；复现前的停止日志保留在 `build/gpis_hang_512x512_32spp/cuda_wavefront.log`。

修复后的同配置渲染完成 **32/32 spp**：开启性能统计时，渲染 58.64 秒、进程总耗时 59.14 秒；完成 8,388,608 个像素样本、19,976,532 个 GPIS 段。CUDA 36,707 批、平均 544.22 条；CPU fallback、StepLimit、InvalidInput 和设备回调错误均为 0。输出 PFM 的所有值均为有限数。

CPU conditioning/gradient 的累计 worker 时间分别约 103.95/77.35 秒，是并行线程时间之和，不能当作总耗时相加。上述耗时来自开启统计的一次完整验证，不是新的无统计性能基准。

## 复现命令

```powershell
cmake --build build --config Release --target tungsten tungsten_gp_wavefront_test -j 8
ctest --test-dir build -C Release --output-on-failure
python tools/verify_wavefront_gpis.py gpis-scenes/shaderball_ply/scene.json --output build/gpis_32spp_repeat --resolution 512x512 --spp 32 --spp-step 1 --bounces 20 --modes cuda_wavefront --profile
```

修复了边界几何行为，因此受影响的棱边路径可能改变图像；不能要求其保持旧版错误的路径结果。
