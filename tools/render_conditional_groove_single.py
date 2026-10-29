"""Render one fixed 3D U-groove realization with the existing frozen renderer."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import copy
import html
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

import test_conditional_groove as g

OUT = g.OUT / 'single_realization'
FIELD_SEED = 17429
RENDER_SEED = 252610281
THREADS = 28
FOLDER = OUT / f'field_seed{FIELD_SEED}'
SOURCE = g.OUT / 'res256_spp256/A_strong/view45/total/gpis_renewal_plus/seed202610281'
EXE = g.ROOT / 'build/Release/tungsten-endpoint-sync.exe'
EXE_HASH = '26b498527cf2fd7c968b18b819a2bf51c27fc73bd433bc3b8b0def8a6deb8a73'
EXPOSURE = .25


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(value, encoding='utf-8')
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.1)


def publish(status):
    g.save(OUT / 'status.json', status)
    refresh = '<meta http-equiv="refresh" content="30">' if status['state'] == 'running' else ''
    shown = FOLDER / 'exposure025.png'
    preview = '<p>等待首个 checkpoint。</p>'
    if shown.exists():
        stamp = shown.stat().st_mtime_ns
        original = 'render.png' if status['state']=='complete' else 'render_checkpoint.png'
        preview = f'<a href="field_seed{FIELD_SEED}/{original}"><img src="field_seed{FIELD_SEED}/exposure025.png?v={stamp}" alt="固定随机场渲染"></a>'
    state_text = {'running':'渲染中', 'complete':'完成', 'failed':'失败'}[status['state']]
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">{refresh}
<meta name="viewport" content="width=device-width, initial-scale=1"><title>U 槽 · single realization</title>
<style>body{{font:16px system-ui,"Microsoft YaHei UI";margin:30px;background:#f4f6f9;color:#24384b;line-height:1.7}}main{{max-width:1100px;margin:auto}}.images{{display:flex;gap:24px;flex-wrap:wrap}}figure{{margin:0;padding:20px;background:white;border:1px solid #d9e2e8;border-radius:12px}}img{{width:384px;max-width:100%;height:auto;display:block}}figcaption{{max-width:384px}}a{{color:#2475ba}}code{{background:#e6edf3;padding:2px 5px}}.status{{padding:12px 18px;background:#e5eef8}}</style><main>
<h1>U 槽：一份固定的随机场</h1>
<p>σ=0.025，ℓ=0.4；256×256，目标 256 spp。沿用正式版的 45° 相机、右上方面积光、U 槽 SDF 和 mirror albedo=0.85。</p>
<p class="status">{state_text} · {status['completed_spp']}/256 spp · 已用时 {status['elapsed_seconds']/60:.1f} 分钟</p>
<div class="images"><figure>{preview}<figcaption><b>Single realization · field seed={FIELD_SEED}</b><br>曝光 0.25 + filmic；所有像素、路径和反弹共享同一份 3D 随机场。增加 spp 只增加光路采样。</figcaption></figure>
<figure><a href="../res256_spp256/A_strong/view45/total/gpis_renewal_plus/seed202610281/render.png"><img src="../res256_spp256/A_strong/view45/total/gpis_renewal_plus/seed202610281/exposure025.png" alt="此前 GPIS renewal+ 平均图"></a><figcaption><b>此前 GPIS renewal+ · 256 spp</b><br>相同曝光和构图；按其 Renewal+ 规则逐路径、逐段采样，供外观参考。</figcaption></figure></div>
<p><code>single_realization=true</code> · <code>correlation_context=global</code> · <code>1D_sampling=false</code> · impulse_density=10 · step_size=0.003125。</p>
<p>沿用现有 sparse_conv_noise 求值和冻结的渲染器。该图是有限脉冲密度、离散步进下的一份固定场样本；右图也更换了相关性处理方式，二者不是仅切换平均次数的配对估计。</p>
<p><a href="field_seed{FIELD_SEED}/scene.json">场景 JSON</a> · <a href="field_seed{FIELD_SEED}/render.log">渲染日志</a> · <a href="manifest.json">运行配置</a> · <a href="../layout.html">相机与光源示意图</a> · <a href="../res256_spp256/comparison.html">正式对比</a></p>'''
    if status['state'] == 'complete':
        body += f'<p><a href="field_seed{FIELD_SEED}/render.png">原始 PNG</a> · <a href="field_seed{FIELD_SEED}/render.pfm">线性 PFM</a> · <a href="field_seed{FIELD_SEED}/result.json">完成记录</a></p>'
    if status.get('error'):
        body += '<p>' + html.escape(status['error']) + '</p>'
    atomic_text(OUT / 'comparison.html', body + '</main></html>')


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    if g.digest(EXE) != EXE_HASH:
        raise RuntimeError('Frozen renderer differs')
    scene = copy.deepcopy(g.load(SOURCE / 'scene.json'))
    scene['media'][0].update(single_realization=True, correlation_context='global', seed=FIELD_SEED)
    scene['renderer'].update(output_directory=FOLDER.as_posix(), spp=256, spp_step=4,
                             checkpoint_interval='1s', enable_resume_render=False)
    scene_path = FOLDER / 'scene.json'
    if scene_path.exists() and g.load(scene_path) != scene:
        raise RuntimeError('Existing single-realization scene differs')
    if (FOLDER / 'result.json').exists():
        raise RuntimeError('This fixed realization already has a result; preserve it')
    g.save(scene_path, scene)
    assets = [Path(scene['media'][0]['gaussian_process']['mean']['grid']['file'])]
    preserved = {str(path):g.digest(path) for path in assets}
    for name in ('scene.json','render.png','render.pfm','render.log','result.json'):
        path = SOURCE / name
        preserved[str(path)] = g.digest(path)
    for name in ('comparison.html','summary.json'):
        path = g.OUT / 'res256_spp256' / name
        preserved[str(path)] = g.digest(path)
    command = [str(EXE), '-t', str(THREADS), '-s', str(RENDER_SEED), '-d', str(FOLDER), str(scene_path)]
    manifest = dict(created_utc=now(), scene=scene_path.as_posix(), scene_sha256=g.digest(scene_path),
        executable=EXE.as_posix(), executable_sha256=EXE_HASH, command=command,
        field_seed=FIELD_SEED, render_seed=RENDER_SEED, threads=THREADS,
        resolution=[256,256], spp=256, display_exposure=EXPOSURE, preserved_files=preserved,
        runner_sha256=g.digest(Path(__file__)), source_scene=(SOURCE/'scene.json').as_posix(),
        fixed_field_definition='computeSeed returns only medium.seed when single_realization is true; conditioning is disabled.',
        field_implementation='Existing 3D sparse convolution, impulse_density=10, 1D_sampling=false.',
        caveat='One fixed approximate field, not an ensemble average or an exact GP realization.')
    g.save(OUT / 'manifest.json', manifest)
    (OUT/'runner_at_launch.py').write_bytes(Path(__file__).read_bytes())
    start = time.monotonic()
    state = dict(state='running', completed_spp=0, target_spp=256, elapsed_seconds=0,
                 updated_utc=now(), error=None)
    publish(state)
    env = os.environ.copy()
    for key in ('TUNGSTEN_GPIS_CONDITIONAL_PROFILE','TUNGSTEN_GPIS_PROFILE','TUNGSTEN_GPIS_FS_PROFILE'):
        env.pop(key, None)
    with (FOLDER / 'render.log').open('wb') as log:
        process = subprocess.Popen(command, cwd=g.ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        g.save(OUT / 'process.json', dict(driver_pid=os.getpid(), renderer_pid=process.pid, started_utc=now()))
        print(f'START renderer_pid={process.pid} threads={THREADS}', flush=True)
        last_spp, last_pfm = -1, None
        while process.poll() is None:
            progress = (FOLDER / 'render.log').read_text(encoding='utf-8', errors='replace')
            matches = re.findall(r'Completed\s+(\d+)/256\s+spp', progress)
            completed = int(matches[-1]) if matches else 0
            pfm = FOLDER / 'render_checkpoint.pfm'
            try:
                modified = pfm.stat().st_mtime_ns if pfm.exists() else None
                if modified and modified != last_pfm:
                    image = g.read_image(pfm)
                    g.write_png(FOLDER/'exposure025.png', g.filmic(image*EXPOSURE))
                    last_pfm = modified
            except (OSError, ValueError):
                pass  # The renderer may be writing a checkpoint at this moment.
            state.update(completed_spp=completed, elapsed_seconds=time.monotonic()-start, updated_utc=now())
            publish(state)
            if completed != last_spp:
                print(f'PROGRESS {completed}/256 elapsed={state["elapsed_seconds"]:.1f}s', flush=True)
                last_spp = completed
            time.sleep(10)
    code = process.returncode
    row = dict(group='A_strong', angle=45, mode='gpis_single_realization', variant='total',
        size=256, spp=256, seed_id=FIELD_SEED, field_seed=FIELD_SEED, render_seed=RENDER_SEED,
        seconds=time.monotonic()-start, returncode=code, command=command,
        scene=scene_path.relative_to(g.OUT).as_posix(), scene_sha256=g.digest(scene_path),
        executable_sha256=EXE_HASH, status='complete', error=None)
    try:
        if code:
            raise RuntimeError(f'Renderer exited with code {code}')
        image = g.validate_output(FOLDER, dict(size=256, spp=256))
        if image.max() <= 0:
            raise RuntimeError('Empty radiance image')
        for path, digest in preserved.items():
            if g.digest(Path(path)) != digest:
                raise RuntimeError('Preserved input changed: ' + path)
        g.write_png(FOLDER/'exposure025.png', g.filmic(image*EXPOSURE))
        row.update(image_mean=float(image.mean()), image_maximum=float(image.max()),
                   nonzero_pixels=int(np.any(image>0, axis=-1).sum()))
        g.save(OUT/'validation.json', dict(passed=True, resolution=[256,256], spp=256,
            finite_nonnegative=True, preserved_files=len(preserved),
            single_realization=True, field_seed=FIELD_SEED,
            png_sha256=g.digest(FOLDER/'render.png'), pfm_sha256=g.digest(FOLDER/'render.pfm')))
    except (OSError, RuntimeError, ValueError) as exc:
        row.update(status='failed', error=str(exc))
    g.save(FOLDER/'result.json', row)
    state.update(state=row['status'], completed_spp=256 if row['status']=='complete' else state['completed_spp'],
                 elapsed_seconds=row['seconds'], updated_utc=now(), error=row['error'])
    publish(state)
    print(f'{row["status"].upper()} elapsed={row["seconds"]:.1f}s error={row["error"]}', flush=True)
    if row['status'] != 'complete':
        raise RuntimeError(row['error'])


def preview_loop():
    """Refresh previews for an already running render without restarting it."""
    previous = None
    while True:
        state = g.load(OUT/'status.json')
        if state['state'] != 'running':
            return
        checkpoint = FOLDER/'render_checkpoint.pfm'
        try:
            modified = checkpoint.stat().st_mtime_ns
            if modified != previous:
                image = g.read_image(checkpoint)
                temporary = FOLDER/'exposure025.tmp.png'
                g.write_png(temporary, g.filmic(image*EXPOSURE))
                os.replace(temporary, FOLDER/'exposure025.png')
                previous = modified
                print(f'PREVIEW {state["completed_spp"]} spp', flush=True)
        except (OSError, ValueError):
            pass
        time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    preview_loop() if args.preview else main()
