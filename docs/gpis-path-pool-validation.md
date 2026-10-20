# GPIS 路径池与多批次 CUDA 提交验证

后续发现的 512×512 / 32 spp 停在 15/32 问题是 cube 棱边出口法线错误，
已修复；见 [32 spp 停滞定位与修复](gpis-32spp-stall.md)。下列时序数据来自该修复之前。

后续发现的 512×512 / 32 spp 停在 15/32 问题是 cube 棱边出口法线错误，
已修复；见 [32 spp 停滞定位与修复](gpis-32spp-stall.md)。下列时序数据来自该修复之前。

## 本次实现

- `IndependentPathSampler`：Uniform PCG 和 Sobol supplemental PCG 都按 scene seed、pixel、绝对 sample index 初始化；Sobol scramble 保持每像素固定。
- `PathTraceIntegrator::renderPathPool()`：4096 个稳定路径槽位，CPU 完成事件队列、按 medium 分组的 GP 请求队列，以及最多 4 个独立在途提交。
- 路径完成后释放槽位并补入任务；handle 含 generation。颜色、AOV、自适应统计按 tile/pixel/sample 顺序归并；未归并任务窗口不超过路径池容量的两倍。
- 满批立即提交，CPU 生产者排空时提交尾批，没有定时合批等待，也没有整帧 wave 屏障。
- CUDA 后端不再等待 `inFlight_ == 0` 才接收新任务。每个 slot 拥有独立 nonblocking stream、pinned host 数组和设备射线缓冲区；mean grid 每个 medium 上传一次并共享只读。
- 取消时等待所有 future、完成通知和 CPU 任务。后端销毁时排空各 stream，再释放设备缓冲区和 grid。

CPU 条件化 context、CPU 梯度与着色仍保留。本次没有实现上一方案中的第三步。

## 配置

在现有 `integrator` 对象中加入：

```json
"use_cuda_gpis": true,
"wavefront_gpis": true,
"gpis_path_pool": true,
"independent_path_sampling": true,
"gpis_max_active_paths": 4096,
"gpis_batch_size": 1024,
"gpis_inflight_batches": 4
```

只设置 `use_cuda_gpis: true` 也会默认启用以上路径池设置。输入场景本身未改为 CUDA。

CPU 对照设置 `use_cuda_gpis: false`，保留 `independent_path_sampling: true`。可用 `wavefront_gpis: false` 比较 blocking CPU，也可保留 true 测 CPU 路径池。

旧 tile 调度设置 `gpis_path_pool: false`。要恢复历史 tile 随机数流，再设置 `independent_path_sampling: false`。新采样模式会改变历史同 seed 的具体样本；不能把不同采样模式的图像当作调度器精确对照。

## 实测

环境：Windows Release，RTX 4060，CUDA 13.1，8 个 CPU worker，seed 12345。输入为当前 shaderball 场景的外部相机，保留直接光照、max_bounces=20；使用临时场景副本。下列是单次测量，另一次 256×256 测量给出波动范围。

| 模式 | 分辨率 / spp | 渲染阶段 | 进程总耗时 | CUDA 批次 | 平均批量 | 最大批量 |
|---|---|---:|---:|---:|---:|---:|
| CPU blocking，独立采样 | 256×256 / 1 | 1.03 s | 1.313 s | — | — | — |
| CPU 路径池 | 256×256 / 1 | 1.13 s | 1.641 s | — | — | — |
| CUDA 旧 tile，独立采样，256/1 slot | 256×256 / 1 | 1.50 s | 1.859 s | 4443 | 34.93 | 232 |
| CUDA 路径池，1024/4 slots | 256×256 / 1 | 0.32 s | 0.813 s | 359 | 432.28 | 1024 |
| CPU blocking，独立采样 | 512×512 / 2 | 9.72 s | 10.141 s | — | — | — |
| CPU 路径池 | 512×512 / 2 | 11.74 s | 12.078 s | — | — | — |
| CUDA 路径池，1024/4 slots | 512×512 / 2 | 2.24 s | 2.891 s | 2035 | 612.62 | 1024 |

256×256 另一次 CUDA 测量：0.29 s 渲染、0.672 s 总耗时、329 批次、平均 471.70 条。提交时机随 CPU/GPU 完成顺序变化，批次数不是固定值；图像结果保持相同。

当前规模的总耗时加速约 1.6–2.3 倍；512×512 / 2 spp 的单次总耗时加速约 3.5 倍。CPU 路径池本身有调度与归并开销，因此 CPU blocking 仍作为性能参考。

两种规模 CUDA 均没有 CPU fallback。路径池实际峰值为 4096 条活动路径、4 个在途提交，CUDA 执行槽峰值为 3。单独的多 medium 并发测试实际达到 4 个执行槽同时在途。

报告与输出：

- `build/gpis_path_pool_final_256x256_1spp/report.json`
- `build/gpis_path_pool_256x256_1spp/report.json`
- `build/gpis_tile_independent_256x256_1spp/report.json`
- `build/gpis_path_pool_512x512_2spp/report.json`

## 正确性和退出验证

Release CTest：4/4 通过，包含：

- 历史 Uniform/Sobol tile 模式与 blocking CPU 精确一致。
- 新独立 Uniform/Sobol 模式：blocking CPU、路径池 CPU、拒绝 GPU 后的 CPU fallback，颜色以及 depth/normal/albedo/visibility 精确一致。
- 非整 tile 尺寸 17×13、多 spp、不同 spp_step、31/67/512 个路径槽的结果一致。
- 20 spp 自适应采样的颜色和 AOV 精确一致。
- 4 个提交同时未完成，反向发布结果/通知后，最终图像与参考精确一致。
- 主路径及阴影查询取消：future 就绪后仍等待 withheld notification；排空所有通知后才退出，不调用完成回调、不推进 spp。
- 多 medium、多个请求所有者、unsupported rejection、13,200 条跨批请求的顺序与生命周期验证；8802 个受支持 GPU 查询无 fallback。
- 原有 SE、ray segment、VDB mean 和 GP 体积边界测试。

CLI 额外对照：

- 同样独立采样的 256×256 CPU tile 与 CPU pool 图像精确一致；CUDA tile 与 CUDA pool 图像也精确一致。
- 64×64 / 2 spp，1 与 8 个 CPU worker，CPU blocking、CPU pool、CUDA pool 各自图像精确一致。
- 256×256 / 1 spp 和 512×512 / 2 spp，CPU blocking 与 CPU pool 精确一致。

CPU/CUDA 本身的交点数值差异仍存在，不能宣称两者精确一致。最终 256×256 比较有 323 个像素最大通道误差 >1e-3，mean_abs≈0.000566；512×512 有 2519 个此类像素，mean_abs≈0.000610。旧 CUDA tile 与新 CUDA pool 精确一致说明这些差异不是本次调度修改引入的。

## 复现

```powershell
cmake --build build --config Release --target tungsten tungsten_gp_wavefront_test -j 8 -- /p:UseMultiToolTask=true /p:EnforceProcessCountAcrossBuilds=true
ctest --test-dir build -C Release --output-on-failure

python tools/verify_wavefront_gpis.py gpis-scenes/shaderball_ply/scene.json --output build/gpis_path_pool_repeat --resolution 256x256 --spp 1 --bounces 20 --modes cpu_legacy,cpu_wavefront,cuda_wavefront

python tools/verify_wavefront_gpis.py gpis-scenes/shaderball_ply/scene.json --output build/gpis_path_pool_highres_repeat --resolution 512x512 --spp 2 --spp-step 2 --bounces 20 --modes cpu_legacy,cpu_wavefront,cuda_wavefront

python tools/verify_wavefront_gpis.py gpis-scenes/shaderball_ply/scene.json --output build/gpis_tile_repeat --resolution 256x256 --spp 1 --bounces 20 --tile-scheduler --batch-size 256 --inflight-batches 1 --modes cpu_wavefront,cuda_wavefront
```

脚本默认让所有模式使用同一种独立采样；`--legacy-tile-sampling` 恢复历史 tile 调度和随机数流。可用 `--active-paths`、`--batch-size`、`--inflight-batches` 调整容量，`--profile` 查看 CPU GP 工作及 GPU 拒绝原因。
