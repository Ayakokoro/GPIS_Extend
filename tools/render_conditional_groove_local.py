"""Complete the native U-groove matrix with B/C/D classic_local renders.

Reuse the frozen renderer and existing scene settings. D needs density baked for
its own sigma; no renderer, random-field evaluator, or original artifact changes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import ctypes
from datetime import datetime, timezone
import html
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time
from urllib.parse import unquote

import numpy as np

import render_conditional_groove_renewal as r
from report_conditional_groove_renewal import save_json, save_text
import preview_conditional_renewal_progress as preview

g, p = r.g, r.p
OUT, STAGE = r.OUT, r.STAGE
MODE, SEED, THREADS, WORKERS = 'classic_local', 222610281, 8, 3
LOCAL_FIELD = OUT/'fields/classic_local_D_small_sigma'
LOCAL_PAIRS = [('global_conditional', MODE, 'conditional_minus_local'),
               (r.MODE, MODE, 'renewal_plus_minus_local'),
               (MODE, 'classic_global', 'local_minus_classic_global')]


def job_for(group):
    return dict(group=group, angle=45, mode=MODE, variant='total', size=256,
                spp=256, seed_id=202610281)


def local_field(group):
    return LOCAL_FIELD if group == 'D_small_sigma' else OUT/'fields/main'


def make_scene(group):
    scene = g.load(STAGE/group/'view45/total/classic_global/seed202610281/scene.json')
    source = scene['media'][0]
    medium = copy.deepcopy(g.load(STAGE/'A_strong/view45/total/classic_local/seed202610281/scene.json')['media'][0])
    medium.update(name=source['name'], max_bounces=source['max_bounces'],
                  material=copy.deepcopy(source['material']),
                  covariance=copy.deepcopy(source['gaussian_process']['covariance']))
    medium['field']['file'] = (local_field(group)/'u_groove.nvdb').as_posix()
    scene['media'] = [medium]
    scene['renderer']['output_directory'] = (STAGE/g.job_key(job_for(group))).as_posix()
    return scene


def verify_field(group):
    meta = g.load(local_field(group)/'u_groove.nvdb.json')
    original = g.load(OUT/'fields/main/u_groove.nvdb.json')
    for name in ('geometry', 'geometry_bounds', 'process_box', 'origin', 'resolution',
                 'dx', 'coverage', 'halo_voxels', 'polygon_xy'):
        if meta[name] != original[name]:
            raise RuntimeError('Mean geometry or grid differs: '+name)
    if not np.isclose(meta['sigma'], g.GROUPS[group][0]*g.W, rtol=0, atol=1e-15):
        raise RuntimeError('Baked density sigma mismatch')
    if not all(meta['verification'][k] for k in ('density_verified', 'sdf_formats_identical')):
        raise RuntimeError('Field verification failed')
    return meta


def prepare():
    baseline = g.load(STAGE/'manifest.json')
    renewal = g.load(STAGE/'renewal_plus_manifest.json')
    if g.digest(Path(baseline['executable'])) != p.EXE_HASH:
        raise RuntimeError('Frozen renderer changed')
    baker = g.ROOT/'build/Release/bake_contact_fields.exe'
    if g.digest(baker) != baseline['baker_sha256']:
        raise RuntimeError('Frozen baker changed')
    request = g.load(OUT/'fields/main/request.json')
    request.update(output_directory=LOCAL_FIELD.as_posix(),
                   sigma=g.GROUPS['D_small_sigma'][0]*g.W,
                   length_scale=g.GROUPS['D_small_sigma'][1]*g.W)
    request_path = LOCAL_FIELD/'request.json'
    if request_path.exists() and g.load(request_path) != request:
        raise RuntimeError('Existing D density request differs')
    if not request_path.exists(): save_json(request_path, request)
    if not (LOCAL_FIELD/'u_groove.nvdb.json').exists():
        print('BAKE D density for sigma='+str(request['sigma']), flush=True)
        with (LOCAL_FIELD/'bake.log').open('wb') as log:
            subprocess.run([str(baker), str(request_path)], cwd=g.ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    for group in g.GROUPS: verify_field(group)
    old_jobs = baseline['jobs']+renewal['jobs']
    jobs = [job_for(group) for group in g.GROUPS if group != 'A_strong']
    path = STAGE/'classic_local_manifest.json'
    if path.exists():
        manifest = g.load(path)
        if manifest['runner_sha256'] != g.digest(Path(__file__)):
            raise RuntimeError('Local runner changed; preserve the existing run')
    else:
        preserved = {}
        for job in old_jobs:
            folder = STAGE/g.job_key(job)
            row = g.load(folder/'result.json')
            if row['status'] != 'complete': raise RuntimeError('Existing render incomplete')
            g.validate_output(folder, job)
            for file in [folder/name for name in ('scene.json', 'result.json', 'render.png', 'render.pfm', 'render.log')]+[OUT/row['scene']]:
                preserved[file.relative_to(OUT).as_posix()] = g.digest(file)
        for name in ('manifest.json', 'validation.json', 'recovery_manifest.json',
                     'renewal_plus_manifest.json', 'renewal_plus_validation.json'):
            file = STAGE/name
            preserved[file.relative_to(OUT).as_posix()] = g.digest(file)
        for name in ('comparison.html', 'summary.json', 'status.json', 'renewal_plus_progress.html'):
            source = STAGE/name
            shutil.copy2(source, STAGE/(source.stem+'_before_classic_local'+source.suffix))
        assets = {}
        for folder in (OUT/'fields/main', LOCAL_FIELD):
            for name in ('u_groove.nvdb', 'u_groove_sdf.vdb', 'u_groove.nvdb.json', 'request.json'):
                file = folder/name
                assets[file.relative_to(OUT).as_posix()] = g.digest(file)
        manifest = dict(jobs=jobs, existing_jobs=old_jobs, executable=baseline['executable'],
            executable_sha256=p.EXE_HASH, baker_sha256=baseline['baker_sha256'],
            runner_sha256=g.digest(Path(__file__)), helper_sha256=g.digest(Path(g.__file__)),
            preserved_files=preserved, assets=assets, threads=THREADS, workers=WORKERS,
            render_seed=SEED, resolution=[256,256], spp=256,
            backend='Existing classic_local with baked density and stationary covariance.',
            density_note='B/C reuse sigma=0.025 density; D rebakes sigma=0.0025 with identical geometry, domain and voxel size. Covariance supplies the gradient statistics; alpha grid is unused.',
            caveat='One seed per image. classic_local interpolates baked density and normalizes the mean gradient; it is an auxiliary comparison. D sigma=0.0025 is below dx=0.003125, so discretization remains a limitation.')
        save_json(path, manifest)
        (STAGE/'classic_local_runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    if manifest['helper_sha256'] != g.digest(Path(g.__file__)):
        raise RuntimeError('Original helper changed')
    for job in jobs:
        scene_path = STAGE/g.job_key(job)/'scene.json'
        scene = make_scene(job['group'])
        if scene_path.exists() and g.load(scene_path) != scene:
            raise RuntimeError('Existing local scene differs')
        if not scene_path.exists(): save_json(scene_path, scene)
    return manifest


def render(job, manifest):
    folder = STAGE/g.job_key(job)
    command = [manifest['executable'], '-t', str(THREADS), '-s', str(SEED),
               '-d', str(folder), str(folder/'scene.json')]
    environment = os.environ.copy()
    for key in ('TUNGSTEN_GPIS_CONDITIONAL_PROFILE', 'TUNGSTEN_GPIS_PROFILE', 'TUNGSTEN_GPIS_FS_PROFILE'):
        environment.pop(key, None)
    print('START '+g.job_key(job), flush=True)
    start = time.monotonic()
    status, error, code = 'complete', None, None
    try:
        with (folder/'render.log').open('wb') as log:
            proc = subprocess.run(command, cwd=g.ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  env=environment, timeout=43200)
        code = proc.returncode
        if code: raise RuntimeError(f'Exit code {code}')
        image = g.validate_output(folder, job)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        status, error = 'failed', str(exc)
    row = dict(job, status=status, error=error, returncode=code, render_seed=SEED,
        seconds=time.monotonic()-start, command=command, executable_sha256=p.EXE_HASH,
        scene=(folder/'scene.json').relative_to(OUT).as_posix(),
        scene_sha256=g.digest(folder/'scene.json'), execution_threads=THREADS,
        manifest='res256_spp256/classic_local_manifest.json')
    if status == 'complete':
        mask = g.roi_mask(g.load(folder/'scene.json')['camera'])
        row.update(roi_pixels=int(mask.sum()), roi_mean=float(image[mask].mean()),
                   image_mean=float(image.mean()), image_maximum=float(image.max()))
        edge = mask & ~(np.roll(mask,1,0)&np.roll(mask,-1,0)&np.roll(mask,1,1)&np.roll(mask,-1,1))
        for exposure, name in ((1, 'roi.png'), (p.EXPOSURE, 'roi_exposure025.png')):
            display = g.filmic(image*exposure)
            display[edge] = [1, .15, .1]
            g.write_png(folder/name, display)
        g.write_png(folder/'exposure025.png', g.filmic(image*p.EXPOSURE))
    save_json(folder/'result.json', row)
    print(f'{status.upper()} {job["group"]} seconds={row["seconds"]:.1f} ROI={row.get("roi_mean")} error={error}', flush=True)
    return row


def publish(manifest, active=()):
    jobs = manifest['existing_jobs']+manifest['jobs']
    rows, arrays = {}, {}
    for job in jobs:
        folder = STAGE/g.job_key(job)
        if (folder/'result.json').exists():
            row = g.load(folder/'result.json')
            rows[job['group'],job['mode']] = row
            if row['status'] == 'complete': arrays[job['group'],job['mode']] = g.read_image(folder/'render.pfm')
    complete = sum(row['status']=='complete' for row in rows.values())
    failed = sum(row['status']!='complete' for row in rows.values())
    status = dict(state='complete' if complete==len(jobs) else 'failed' if failed and not active else 'running',
                  completed=complete, total=len(jobs), failed=failed, active=list(active),
                  updated_utc=datetime.now(timezone.utc).isoformat())
    comparisons, image_rows = [], []
    diff_rows = [[], []]
    for group, (sigma_w, ell_w) in g.GROUPS.items():
        label = f'<th>{group}<br>σ={sigma_w*g.W:g}<br>ℓ={ell_w*g.W:g}</th>'
        cells = [label]
        comparison = dict(group=group, sigma=sigma_w*g.W, ell=ell_w*g.W, roi={}, differences={})
        for mode in r.MODES:
            row = rows.get((group,mode))
            if row is None: cells.append('<td>渲染中</td>'); continue
            if row['status'] != 'complete':
                cells.append('<td>失败：'+html.escape(str(row['error']))+'</td>'); continue
            key = g.job_key(row)
            actual_scene = (OUT/row['scene']).relative_to(STAGE).as_posix()
            comparison['roi'][mode] = row['roi_mean']
            cells.append(f'<td><a href="{key}/exposure025.png"><img src="{key}/exposure025.png" width="256" height="256"></a>'
                f'<br>ROI {row["roi_mean"]:.6g}<br><a href="{key}/render.png">原始 PNG</a> · <a href="{key}/render.pfm">线性 PFM</a> · '
                f'<a href="{key}/roi_exposure025.png">ROI</a><br><a href="{actual_scene}">场景</a> · <a href="{key}/render.log">日志</a></td>')
        image_rows.append('<tr>'+''.join(cells)+'</tr>')
        for index, pairs in enumerate((r.PAIRS, LOCAL_PAIRS)):
            cells = [label]
            for a, b, name in pairs:
                if (group,a) not in arrays or (group,b) not in arrays:
                    cells.append('<td>等待成对结果</td>'); continue
                diff = arrays[group,a]-arrays[group,b]
                folder = STAGE/('renewal_plus_differences' if index==0 else 'classic_local_differences')/group
                png, pfm = folder/(name+'.png'), folder/(name+'.pfm')
                g.write_png(png, g.signed_color(diff.mean(axis=-1), .5)); g.write_pfm(pfm, diff)
                delta = comparison['roi'][a]-comparison['roi'][b]
                relative = delta/comparison['roi'][b]
                comparison['differences'][name] = dict(delta=delta, relative_delta=relative, ci95=None)
                png, pfm = png.relative_to(STAGE).as_posix(), pfm.relative_to(STAGE).as_posix()
                cells.append(f'<td><a href="{png}"><img src="{png}" width="256" height="256"></a><br>ROI Δ={delta:+.6g} ({relative:+.2%})<br><a href="{pfm}">线性差分 PFM</a></td>')
            diff_rows[index].append('<tr>'+''.join(cells)+'</tr>')
        legacy = comparison['differences']['conditional_minus_classic']
        comparison.update(delta=legacy['delta'], relative_delta=legacy['relative_delta'], delta_ci95=None)
        comparisons.append(comparison)
    save_json(STAGE/'status.json', status)
    save_json(STAGE/'summary.json', dict(**status, resolution=[256,256], spp=256,
        seeds_per_image=1, display_exposure=p.EXPOSURE, difference_scale=[-.5,.5],
        inference=manifest['caveat'], comparisons=comparisons, results=list(rows.values())))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>U 槽 · 四模式 · 256×256 · 256 spp</title>
<style>body{font:15px system-ui;margin:28px;color:#243047;background:#f5f7fa}p{max-width:1180px;line-height:1.75}table{border-collapse:collapse;background:white}th,td{padding:12px;border:1px solid #ccd4df;text-align:center}img{display:block;max-width:none}a{color:#175ab0}</style>
<h1>U 槽：256×256 / 256 spp</h1><p><a href="../comparison.html">全部实验</a> · <a href="../plan.html">实验方案</a> · <a href="summary.json">线性统计</a> · <a href="classic_local_manifest.json">classic_local 配置</a> · <a href="renewal_plus_manifest.json">Renewal+ 配置</a> · <a href="comparison_before_classic_local.html">补齐前的 13 张</a></p>
<p>固定 45° 视角、同一 U 槽几何、面积光和 mirror albedo=0.85。四组 × classic_global / global_conditional / GPIS renewal+ / classic_local，共 16 张。每张原生 256×256、256 spp、1 个独立 seed。统一曝光 0.25 后 filmic 显示，保留原始 PNG 和线性 PFM，无降噪或放大。</p>
<p>GPIS renewal+ 沿用现有 3D sparse_conv_noise，impulse_density=10，1D_sampling=false，起点条件化场值及完整梯度；最大步长 min(ℓ/4,w/32)=0.003125，min_step=8。它是有限密度与离散步进近似，并非完整 GP 后验。</p>
<p>classic_local 使用烘焙密度插值和单位均值梯度，作为辅助对照。A/B/C 复用 σ=0.025 的密度网格；D 单独按 σ=0.0025 烘焙，相同几何、范围和 dx=0.003125。核提供各组梯度统计，不使用 alpha 网格。D 的 σ 小于网格间距，其差异仍包含离散化影响。</p>
<p>ROI 为预先固定的右内壁投影，在线性 RGB 上求均值。差分均为左项减右项，统一蓝 −0.5 / 白 0 / 红 +0.5；仅显示饱和，数值不截断。单 seed 无重复试验置信区间。</p>'''
    page += f'<p>已完成 {complete} / {len(jobs)} 张；失败 {failed} 张。更新于 {status["updated_utc"]}</p>'
    page += '<table><tr><th>参数</th>'+''.join(f'<th>{r.LABELS[m]}</th>' for m in r.MODES)+'</tr>'+''.join(image_rows)+'</table>'
    for pairs, entries in zip((r.PAIRS, LOCAL_PAIRS), diff_rows):
        page += '<p>差分：左项减右项</p><table><tr><th>参数</th>'+''.join(f'<th>{r.LABELS[a]} − {r.LABELS[b]}</th>' for a,b,_ in pairs)+'</tr>'+''.join(entries)+'</table>'
    save_text(STAGE/'comparison.html', page+'</html>')
    g.MODES = r.MODES
    for attempt in range(20):
        try: g.report(OUT); break
        except PermissionError:
            if attempt==19: raise
            time.sleep(.25)
    p.link_from_index(OUT)
    for name in ('comparison.html', 'findings.html'):
        target = OUT/name
        text = target.read_text(encoding='utf-8')
        text = re.sub(r'(<a href="res256_spp256/comparison.html">)[^<]*(</a>)',
                      r'\g<1>256×256 / 256 spp 正式出图（16 张，四模式）\g<2>', text)
        text = text.replace('<h1>U 槽：classic 与 global_conditional</h1>', '<h1>U 槽：classic、global_conditional 与 GPIS renewal+</h1>')
        text = text.replace(f'<th>{r.MODE}</th>', f'<th>{r.LABELS[r.MODE]}</th>')
        save_text(target, text)
    preview.main()
    return status


def validate(manifest):
    for relative, digest in {**manifest['preserved_files'], **manifest['assets']}.items():
        if g.digest(OUT/relative) != digest: raise RuntimeError('Changed artifact: '+relative)
    if g.digest(Path(manifest['executable'])) != p.EXE_HASH: raise RuntimeError('Renderer changed')
    results = []
    for job in manifest['jobs']:
        folder = STAGE/g.job_key(job)
        row, scene = g.load(folder/'result.json'), g.load(folder/'scene.json')
        if row['status']!='complete' or scene!=make_scene(job['group']) or row['scene_sha256']!=g.digest(folder/'scene.json'):
            raise RuntimeError('Local scene or result mismatch')
        verify_field(job['group'])
        image = g.validate_output(folder, job)
        if not np.isclose(image[g.roi_mask(scene['camera'])].mean(), row['roi_mean'], rtol=0, atol=1e-12):
            raise RuntimeError('ROI mismatch')
        for name in ('render.png', 'exposure025.png', 'roi.png', 'roi_exposure025.png'):
            data = (folder/name).read_bytes()
            if data[:8]!=b'\x89PNG\r\n\x1a\n' or struct.unpack('>II',data[16:24])!=(256,256):
                raise RuntimeError('PNG dimensions mismatch')
        results.append(dict(group=job['group'], png_sha256=g.digest(folder/'render.png'), pfm_sha256=g.digest(folder/'render.pfm')))
    counts = {}
    for page in (STAGE/'comparison.html', STAGE/'renewal_plus_progress.html', OUT/'comparison.html', OUT/'findings.html'):
        parser = r.Links(); parser.feed(page.read_text(encoding='utf-8'))
        links = [v for v in parser.links if not v.startswith(('http:', 'https:', 'data:', '#'))]
        if any(not (page.parent/unquote(link)).exists() for link in links): raise RuntimeError('Broken page link: '+str(page))
        counts[page.relative_to(OUT).as_posix()] = len(links)
    save_json(STAGE/'classic_local_validation.json', dict(state='passed', completed=16, new_renders=3,
        preserved_files_checked=len(manifest['preserved_files']), matched_parameter_groups=4,
        local_links_checked=counts, checks=['256/256 spp and native finite nonnegative PFM',
            'PNG dimensions and ROI recomputation', 'matched physical scenes',
            'density sigma, identical mean geometry and grid coverage', 'original 13 renders preserved',
            'frozen renderer and field hashes', 'comparison and preview links'], images=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    g.save = p.atomic_save = save_json
    p.atomic_text = save_text
    manifest = prepare()
    pending = []
    for job in manifest['jobs']:
        path = STAGE/g.job_key(job)/'result.json'
        if not path.exists(): pending.append(job)
        elif g.load(path)['status']!='complete': raise RuntimeError('Failed attempt retained: '+str(path))
        else: g.validate_output(path.parent, job)
    publish(manifest, [g.job_key(j) for j in pending])
    if args.prepare_only: return
    if args.report_only and pending: raise RuntimeError('Missing local renders')
    keep_awake = ctypes.windll.kernel32 if os.name=='nt' else None
    if keep_awake: keep_awake.SetThreadExecutionState(0x80000001)
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(render,j,manifest):j for j in pending}
            while futures:
                done, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                for future in done: future.result(); del futures[future]
                state = publish(manifest, [g.job_key(j) for j in futures.values()])
                print(f'PROGRESS {state["completed"]}/16 failed={state["failed"]}', flush=True)
        state = publish(manifest)
        if state['completed']!=16: raise RuntimeError('Render failures retained')
        validate(manifest)
        print('COMPLETE: all 16 native images validated', flush=True)
    finally:
        if keep_awake: keep_awake.SetThreadExecutionState(0x80000000)


if __name__=='__main__': main()
