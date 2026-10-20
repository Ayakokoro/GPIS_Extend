# 渲染 `gpis-scenes/` 场景

`run.ps1` 是本地用的批量渲染脚本，包一层 `build/Release/tungsten.exe`，负责场景选择、
输出命名、日志目录和耗时统计。本机构建环境见项目记忆（CMake 必须 3.31.x，vcpkg 在 `D:\vcpkg`）。

## 前置条件

渲染器已经构建好：

```powershell
cmake --build build --config Release --parallel
```

产物是 `build/Release/tungsten.exe`。脚本会检查它是否存在，不存在会直接报错。
加 `-Rebuild` 可以让脚本在渲染前先跑一次上面这条命令。

## 快速开始

```powershell
.\run.ps1 -List                     # 列出所有场景，不渲染
.\run.ps1 veach-mis:uni -Spp 1      # 冒烟测试：只渲染一个变体，1 spp，几秒钟出图
.\run.ps1 all -Spp 64 -Threads 16   # 全量渲染
```

> **先跑 1 spp 确认出图，再跑完整设置。** 场景之间耗时差好几个数量级
> （见下方[实测成本](#场景清单与实测成本)），直接上默认 64 spp 可能要跑几个小时。

## 选择器语法

第一个位置参数是场景选择器，可以给多个，也可以重复给（会自动去重）：

| 写法 | 含义 | 例子 |
| --- | --- | --- |
| `all` | 全部 19 个场景 | `.\run.ps1 all` |
| `<文件夹>` | 该文件夹下所有变体 | `veach-mis`（3 个） |
| `<文件夹>:<后缀>` | 文件夹 + 变体后缀 | `veach-mis:uni` |
| `<文件名>` | 直接按 json 文件名（不带 `.json`） | `dragon-sparse-conv-ensemble` |
| 通配符 | 对文件夹名和文件名做 `-like` 匹配 | `dragon-*`、`chess*` |
| 完整路径 | 指向任意 json 文件 | `gpis-scenes\dragon\dragon-function-space.json` |

`<文件夹>:<后缀>` 先按**完整标签**匹配，匹配不到再退回子串匹配。所以
`shaderball:stationary-27` 只命中一个场景，而 `shaderball:27` 会同时命中
`stationary-27` 和 `stationary-270`。拿不准时用 `-DryRun` 先看一眼。

匹配不到任何场景时只警告、不报错，继续处理其余选择器；全部匹配不到才终止。

## 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `-Scene`（位置参数） | 无 | 场景选择器，必需 |
| `-Spp <int>` | `0` | 每像素采样数，覆盖场景文件里的值。`0` = 用场景自己的设置 |
| `-Threads <int>` | `0` | 渲染线程数。`0` = 让 Tungsten 决定（核心数 − 1） |
| `-OutputDir <path>` | `renders` | 输出目录。相对路径以**仓库根**为基准，不是当前目录 |
| `-Seed <int>` | `-1` | 随机种子，覆盖 Tungsten 内置的 `0xBA5EBA11`。`-1` = 不传 |
| `-List` | | 只列出可用场景，不渲染 |
| `-Rebuild` | | 渲染前先 `cmake --build build --config Release --parallel` |
| `-DryRun` | | 只打印将要执行的命令，不渲染 |

注意 `-Spp` 和 `-Threads` 的 `0` 是**哨兵值表示"不传这个参数"**，不是"渲染 0 个采样/用 0 个线程"。

## 输出

```
renders/
├── veach-mis-uni.png       # 8-bit 预览
├── veach-mis-uni.exr       # 完整动态范围，用于后期
├── coffee-mug-ensemble-average.png
├── coffee-mug-ensemble-average.exr
└── log/
    ├── veach-mis-uni.log
    └── coffee-mug-ensemble-average.log
```

- 文件名是 `文件夹-变体`，去掉重复前缀：`veach-mis/veach-mis-uni.json` → `veach-mis-uni`。
- 两个输出格式都是**绝对路径**传给 Tungsten 的，所以从哪里调用脚本结果都一样，
  并且**不会覆盖** `gpis-scenes/` 里原有的结果。
- `log/` 由脚本预先建好。Tungsten 只在 `<exr 目录>/log/` 存在时才写日志，目录不存在会静默丢弃。
- 脚本退出码：全部成功为 `0`，有场景失败为 `1`。单个场景失败不会中断整个批次。

## 示例

```powershell
# 列出场景，确认选择器写对了
.\run.ps1 -List

# 冒烟测试单个变体，先确认能出图
.\run.ps1 veach-mis:uni -Spp 1 -Threads 8

# 渲染 veach-mis 的全部三个变体做对比（uni / nee / mis）
.\run.ps1 veach-mis -Spp 64 -Threads 16

# 换个随机种子渲染单张 realization，输出到单独目录
.\run.ps1 dragon:sparse-conv-single-realization -Seed 7 -OutputDir renders\seed7

# 先构建，再渲染两个场景
.\run.ps1 chess-set shaderball -Rebuild -Spp 32

# 只想看看命令长什么样
.\run.ps1 statue -DryRun
```

## 场景清单与实测成本

实测环境：本机 32 逻辑核，`1 spp`，默认线程数（31）。**耗时高度依赖分辨率**，所以一并列出来。
场景之间差了三个数量级，先看这张表再决定一次渲染多少。

| 文件夹 | 变体 | 分辨率 | 依赖资源 | 1 spp 实测 |
| --- | --- | --- | --- | --- |
| veach-mis | `uni` / `nee` / `mis` | 1000×563 ⚠️ | 无（纯解析 GP） | **1.4 s** |
| shaderball | `stationary-27` | 512×512 | FlowCheck.hdr | **35 s** |
| shaderball | `stationary-270` | 512×512 | FlowCheck.hdr | **204 s** |
| shaderball | `nonstationary` | 512×512 | FlowCheck.hdr | 未测 |
| chess-set | `function-space` / `-ensemble` / `-single-realization` | 1024×512 | 两个 6.5 MB vdb + obj | 单帧超过 10 分钟（未跑完） |
| coffee-mug | `ensemble-average` | 768×1024 | cup-3.vdb（11 MB） | **~197 s** |
| coffee-mug | `spatial-average` ×3 | 256×256 / 512×512 | cup-3.vdb | 未测（分辨率小，应显著更快） |
| dragon | 三种 | 1024×512 | dragon.vdb（65 MB）+ obj | 未测，应比 coffee-mug 更重 |
| statue | 三种 | 1024×512 | rhino-1024.vdb（2.2 GB）+ bressant-1024.vdb（1.4 GB） | 未测，光加载就要几 GB 内存 |

几个说明：

- 这些数字是**同一台机器上的量级参考**，线程数、分辨率、机器负载都会明显影响；
  另一次用 `-t 8` 跑 coffee-mug 得到 169 s（比 31 线程还快），说明线程数开满不一定最优。
- 场景文件里默认 `spp` 是 64，完整渲染大致按 spp 倍数放大（约几十倍），但首次采样还要
  构建加速结构，所以不是严格线性。
- `dragon/dragon-sdf.vdb`（446 MB）在这三个 dragon 场景里都没被引用，可以不用管。
- 想更快可以调小分辨率，或在场景 json 里把 `"resolution"` 改小再跑。

**开发调试建议**：改 `src/core/math/` 下的代码时用 `veach-mis` 验证（秒级，且不需要 VDB）；
要测 VDB 这条路，`coffee-mug:spatial-average-zoomin-surf` 是 256×256 的小场景，比
`chess-set` 和 `ensemble-average` 便宜得多。

### ⚠️ veach-mis 的分辨率写错了

三个 veach-mis 场景文件里的键是 `"resolutin"`（少一个 `o`），渲染器认的是 `"resolution"`，
所以文件里写的 `1080,720` **不生效**，实际用相机默认值 **1000×563** 渲染。
实测产出的 PNG 就是 1000×563。改掉这个拼写就能按 1080×720 出图。

## 注意事项

### 中文乱码 / 脚本报解析错误

`run.ps1` 是 **UTF-8 with BOM** 保存的。Windows PowerShell 5.1 会把没有 BOM 的 `.ps1`
按系统 ANSI（本机是 GBK）解码，中文注释会被拆坏进而导致解析失败。编辑时请保持 BOM
（VS Code 会保留），或者用 PowerShell 7 运行。

### 执行策略禁止运行脚本

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1 -List
```

或者对当前用户放开：`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。

### 资源路径是相对场景文件解析的

场景里的 `"file": "dragon.vdb"` 由 Tungsten 按**场景文件所在目录**解析
（`StandaloneRenderer::renderScene()` 用 `Path(currentScene).parent()` 作为 input directory），
所以不需要 `cd` 进场景目录。脚本也是靠这个性质直接用绝对路径调用渲染器的。

### 中断与续算

场景文件里 `enable_resume_render` 是 `false`，所以默认中断后从头开始，
不会写 `TungstenRenderState.dat`。脚本没有暴露这一项；需要长任务续算时直接调用
渲染器加 `-c <间隔>`（保存检查点）配合场景里的 `enable_resume_render`。

## 不用脚本，直接调用渲染器

脚本只是薄薄一层包装，需要更细的控制（检查点、`--timeout`、`-p` 像素调试）时直接用：

```powershell
build/Release/tungsten.exe -t 8 --spp 1 -o out.png -e out.exr gpis-scenes/veach-mis/veach-mis-uni.json
```

渲染器自己的选项：

| 选项 | 说明 |
| --- | --- |
| `-t, --threads N` | 线程数，默认核心数 − 1 |
| `--spp N` | 每像素采样数，覆盖场景文件 |
| `-o, --output-file F` | 覆盖场景文件的 `output_file` |
| `-e, --hdr-output-file F` | 覆盖场景文件的 `hdr_output_file` |
| `-d, --output-directory D` | 输出目录，覆盖场景文件 |
| `-i, --input-directory D` | 输入目录，默认取场景文件所在目录 |
| `-s, --seed N` | 随机种子 |
| `-c, --checkpoint D` | 保存检查点的间隔，`0` 关闭 |
| `-r, --restart` | 忽略已有检查点，从 0 spp 重来 |
| `--timeout D` | 最长渲染时间，`0` 不限 |
| `-p, --data-dump-pixel X,Y` | dump 某像素主光线的调试数据 |
| `-h, --help` / `-v, --version` | 帮助 / 版本 |

`-c` 和 `--timeout` 接受时长字符串，单位 `ms` / `s` / `m` / `h` / `d`，**不带单位按分钟算**
（`StringUtils::parseDuration`）。所以间隔半小时写 `30m`，200 秒写 `200s`。

可以一次传多个场景，串行渲染；也可以只跑 `build/Release/json2xml.exe`、`scenemanip.exe`
等附带工具（见 `src/` 下对应目录）。
