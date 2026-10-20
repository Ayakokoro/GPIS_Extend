<#
.SYNOPSIS
    用本地构建的 Tungsten 渲染 gpis-scenes/ 下的场景。

.DESCRIPTION
    场景里的资源路径是相对场景文件自身所在目录解析的，所以从哪里调用都行。
    默认把结果写到 <仓库根>/renders/，每个场景一个 .png + .exr，渲染日志写到
    <仓库根>/renders/log/<名字>.log（Tungsten 只在 log/ 目录已存在时才写日志，
    这个脚本会先建好）。

.PARAMETER Scene
    场景选择器，可以给多个。支持这些写法：
        all                     全部场景
        veach-mis               整个文件夹（veach-mis 的三个变体全渲染）
        veach-mis:mis           文件夹 + 变体后缀，只渲染 veach-mis-mis.json
        veach-mis-mis           直接按 json 文件名（不带扩展名）
        dragon-*                通配符
        gpis-scenes/dragon/dragon-sparse-conv-ensemble.json   完整路径

.PARAMETER Spp
    每个像素的采样数，覆盖场景文件里的设置。0（默认）= 用场景自己的值。

.PARAMETER Threads
    渲染线程数。0（默认）= 让 Tungsten 自己决定（核心数 - 1）。

.PARAMETER OutputDir
    输出目录。相对路径以仓库根为基准。默认 renders。

.PARAMETER Seed
    随机种子，覆盖 Tungsten 内置的默认值。-1（默认）= 不传。

.PARAMETER List
    只列出可用场景，不渲染。

.PARAMETER Rebuild
    渲染前先跑一次 cmake --build build --config Release --parallel。

.PARAMETER DryRun
    只打印将要执行的命令，不真的渲染。

.EXAMPLE
    .\run.ps1 -List

.EXAMPLE
    # 快速冒烟测试：只渲染 veach-mis 的 uni 变体，1 spp
    .\run.ps1 veach-mis:uni --spp 1

.EXAMPLE
    .\run.ps1 all -Spp 64 -Threads 16 -OutputDir renders\final
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]] $Scene,

    [int]    $Spp       = 0,
    [int]    $Threads   = 0,
    [string] $OutputDir = 'renders',
    [int]    $Seed      = -1,

    [switch] $List,
    [switch] $Rebuild,
    [switch] $DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root      = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$Exe       = [IO.Path]::Combine($Root, 'build', 'Release', 'tungsten.exe')
$SceneRoot = [IO.Path]::Combine($Root, 'gpis-scenes')

if (-not (Test-Path -LiteralPath $SceneRoot -PathType Container)) {
    throw "找不到场景目录: $SceneRoot"
}

$AllScenes = @(Get-ChildItem -LiteralPath $SceneRoot -Recurse -File -Filter *.json |
               Sort-Object FullName)

# 场景的"变体标签"：文件名去掉它自己的文件夹前缀。
# chess-set/chess-set-function-space.json -> function-space
function Get-SceneTag {
    param($File)

    $prefix = $File.Directory.Name + '-'
    if ($File.BaseName.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        return $File.BaseName.Substring($prefix.Length)
    }
    return $File.BaseName
}

function Show-Scenes {
    param($Files, [string] $SceneRoot)

    Write-Host ''
    Write-Host '可用场景（选择器可用简写）:' -ForegroundColor Cyan
    foreach ($f in $Files) {
        $rel    = $f.FullName.Substring($SceneRoot.Length) -replace '^[\\/]+', ''
        $folder = $f.Directory.Name
        $tag    = Get-SceneTag $f
        $size   = '{0,7:N1} KB' -f ($f.Length / 1KB)
        Write-Host ("  {0,-12} {1,-28} {2,-11} {3}" -f $folder, $tag, $size, $rel)
    }
    Write-Host ''
    Write-Host '  例: .\run.ps1 veach-mis:uni -Spp 1' -ForegroundColor DarkGray
    Write-Host '      .\run.ps1 dragon -Threads 16' -ForegroundColor DarkGray
    Write-Host ''
}

function Resolve-Scenes {
    param([string[]] $Selectors, $Files, [string] $SceneRoot)

    $picked = New-Object System.Collections.Generic.List[object]

    foreach ($sel in $Selectors) {
        $hits = @()

        if ($sel -eq 'all') {
            $hits = $Files
        }
        elseif ($sel.Contains(':')) {
            $bits     = $sel -split ':', 2
            $folder   = $bits[0]
            $tag      = $bits[1]
            $inFolder = $Files | Where-Object { $_.Directory.Name -like $folder }

            # 先按完整标签匹配（"stationary-27" 只命中 stationary-27），
            # 匹配不到再退回子串匹配（"27" 同时命中 stationary-27 和 stationary-270）
            $hits = $inFolder | Where-Object { (Get-SceneTag $_) -like $tag }
            if (-not $hits) {
                $hits = $inFolder | Where-Object { (Get-SceneTag $_) -like "*$tag*" }
            }
        }
        else {
            $asPath  = [IO.Path]::Combine($SceneRoot, $sel)
            $asJson  = $asPath + '.json'

            if (Test-Path -LiteralPath $asPath -PathType Leaf) {
                $hits = @(Get-Item -LiteralPath $asPath)
            }
            elseif (Test-Path -LiteralPath $asPath -PathType Container) {
                $hits = $Files | Where-Object { $_.DirectoryName -eq (Get-Item -LiteralPath $asPath).FullName }
            }
            elseif (Test-Path -LiteralPath $asJson -PathType Leaf) {
                $hits = @(Get-Item -LiteralPath $asJson)
            }
            elseif (Test-Path -LiteralPath $sel -PathType Leaf) {
                $hits = @(Get-Item -LiteralPath $sel)   # 用户直接给了绝对/相对路径
            }
            else {
                # 按文件夹名 / 文件名做通配匹配
                $hits = $Files | Where-Object {
                    $_.Directory.Name -like $sel -or
                    $_.BaseName       -like $sel -or
                    $_.BaseName       -like "$sel-*"
                }
            }
        }

        if (-not $hits) {
            Write-Warning "选择器 '$sel' 没有匹配到任何场景"
            continue
        }
        foreach ($h in $hits) {
            if (-not ($picked | Where-Object { $_.FullName -eq $h.FullName })) {
                $picked.Add($h)
            }
        }
    }

    return @($picked | Sort-Object FullName)
}

if ($List) {
    Show-Scenes -Files $AllScenes -SceneRoot $SceneRoot
    exit 0
}

if (-not $Scene) {
    Show-Scenes -Files $AllScenes -SceneRoot $SceneRoot
    throw '没有指定场景。上面是可用列表，用法见注释帮助（Get-Help .\run.ps1 -Full）。'
}

if (-not (Test-Path -LiteralPath $Exe -PathType Leaf)) {
    throw "找不到渲染器: $Exe`n先构建: cmake --build build --config Release --parallel"
}

if ($Rebuild) {
    Write-Host '重新构建...' -ForegroundColor Cyan
    Push-Location $Root
    try {
        & cmake --build build --config Release --parallel
        if ($LASTEXITCODE -ne 0) { throw "构建失败（退出码 $LASTEXITCODE）" }
    }
    finally {
        Pop-Location
    }
}

# OutputDir 相对路径按仓库根解析，输出位置不随调用目录变化
$OutDir = if ([IO.Path]::IsPathRooted($OutputDir)) { $OutputDir }
          else { [IO.Path]::Combine($Root, $OutputDir) }
$Null = New-Item -ItemType Directory -Path $OutDir -Force
# Tungsten 把日志写到 <exr 所在目录>/log/<名字>.log，目录不存在就静默丢弃
$Null = New-Item -ItemType Directory -Path ([IO.Path]::Combine($OutDir, 'log')) -Force

# @() 是必须的：只匹配到一个场景时 PowerShell 会把数组拆包，后面 $scenes.Count 会报错
$scenes = @(Resolve-Scenes -Selectors $Scene -Files $AllScenes -SceneRoot $SceneRoot)
if (-not $scenes) { throw '没有匹配到任何场景，用 -List 看可用列表。' }

Write-Host ''
Write-Host ("渲染 {0} 个场景 -> {1}" -f $scenes.Count, $OutDir) -ForegroundColor Cyan
if ($Spp     -gt 0) { Write-Host ("  spp     : {0}" -f $Spp) }
if ($Threads -gt 0) { Write-Host ("  threads : {0}" -f $Threads) }

$failed = New-Object System.Collections.Generic.List[string]
$total  = [Diagnostics.Stopwatch]::StartNew()

foreach ($s in $scenes) {
    if ($s.FullName.StartsWith($SceneRoot, [StringComparison]::OrdinalIgnoreCase)) {
        $rel = $s.FullName.Substring($SceneRoot.Length) -replace '^[\\/]+', ''
    }
    else {
        $rel = $s.Name
    }
    # 文件名本来就带文件夹前缀（veach-mis/veach-mis-uni.json），不再重复一层
    $name = $s.Directory.Name + '-' + (Get-SceneTag $s)
    $exr  = [IO.Path]::Combine($OutDir, "$name.exr")
    $png  = [IO.Path]::Combine($OutDir, "$name.png")

    $cliArgs = New-Object System.Collections.Generic.List[string]
    if ($Threads -gt 0) { $cliArgs.Add('-t'); $cliArgs.Add("$Threads") }
    if ($Spp     -gt 0) { $cliArgs.Add('--spp'); $cliArgs.Add("$Spp") }
    if ($Seed    -ge 0) { $cliArgs.Add('-s'); $cliArgs.Add("$Seed") }
    $cliArgs.Add('-o'); $cliArgs.Add($png)
    $cliArgs.Add('-e'); $cliArgs.Add($exr)
    $cliArgs.Add($s.FullName)

    Write-Host ''
    Write-Host "=== $name ===" -ForegroundColor Yellow
    Write-Host ("  {0}" -f $rel) -ForegroundColor DarkGray

    if ($DryRun) {
        Write-Host ("  {0} {1}" -f $Exe, ($cliArgs -join ' ')) -ForegroundColor DarkGray
        continue
    }

    $sw = [Diagnostics.Stopwatch]::StartNew()
    Push-Location $Root
    try {
        & $Exe @cliArgs
        $code = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    $sw.Stop()

    if ($code -eq 0) {
        Write-Host ("  完成，用时 {0:N1}s" -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green
    }
    else {
        Write-Host ("  失败，退出码 {0}，用时 {1:N1}s" -f $code, $sw.Elapsed.TotalSeconds) -ForegroundColor Red
        $failed.Add($name)
    }
}

$total.Stop()
Write-Host ''
Write-Host ("全部结束，总用时 {0:N1}s" -f $total.Elapsed.TotalSeconds) -ForegroundColor Cyan

if ($failed.Count -gt 0) {
    Write-Host ("失败的场景: {0}" -f ($failed -join ', ')) -ForegroundColor Red
    exit 1
}
