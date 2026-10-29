"""Render the frozen U-groove matrix at native 256 x 256 and 256 spp.

Only schedules the existing renderer and derives display images from linear PFM.
The original experiment helper and all renderer/field evaluation code stay fixed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
from datetime import datetime, timezone
import html
import os
from pathlib import Path
import re
import struct

import numpy as np

import test_conditional_groove as g

STAGE = 'res256_spp256'
EXE_HASH = '26b498527cf2fd7c968b18b819a2bf51c27fc73bd433bc3b8b0def8a6deb8a73'
EXPOSURE = .25
ORIGINAL_SAVE = g.save


def atomic_save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    ORIGINAL_SAVE(temporary, value)
    os.replace(temporary, path)


def atomic_text(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(value, encoding='utf-8')
    os.replace(temporary, path)


def publish(out, jobs, active=()):
    stage = out / STAGE
    rows, arrays, cells = {}, {}, []
    for job in jobs:
        key = g.job_key(job)
        folder = stage / key
        if not (folder/'result.json').exists():
            continue
        row = g.load(folder/'result.json')
        rows[key] = row
        if row['status'] != 'complete':
            continue
        image = g.read_image(folder/'render.pfm')
        arrays[(job['group'], job['mode'])] = image
        display = g.filmic(image * EXPOSURE)
        g.write_png(folder/'exposure025.png', display)
        mask = g.roi_mask(g.load(folder/'scene.json')['camera'])
        edge = mask & ~(np.roll(mask,1,0)&np.roll(mask,-1,0)&np.roll(mask,1,1)&np.roll(mask,-1,1))
        display[edge] = [1, .15, .1]
        g.write_png(folder/'roi_exposure025.png', display)
    comparisons = []
    for group, (sigma_w, ell_w) in g.GROUPS.items():
        comparison = dict(group=group, sigma=sigma_w*g.W, ell=ell_w*g.W, roi={})
        entries = [f'<th>{group}<br>σ={sigma_w*g.W:g}<br>ℓ={ell_w*g.W:g}</th>']
        for mode in g.MODES:
            job = next((j for j in jobs if j['group']==group and j['mode']==mode), None)
            if job is None:
                entries.append('<td>—</td>')
                continue
            key = g.job_key(job)
            row = rows.get(key)
            if row is None:
                label = '渲染中' if key in active else '排队中'
                entries.append(f'<td>{label}</td>')
            elif row['status'] != 'complete':
                entries.append(f'<td>失败：{html.escape(str(row.get("error")))}<br><a href="{key}/render.log">日志</a></td>')
            else:
                comparison['roi'][mode] = row['roi_mean']
                entries.append(f'<td><a href="{key}/exposure025.png"><img src="{key}/exposure025.png" width="256" height="256"></a>'
                               f'<br>ROI {row["roi_mean"]:.6g}<br><a href="{key}/render.png">原始 PNG</a> · '
                               f'<a href="{key}/render.pfm">线性 PFM</a> · <a href="{key}/roi_exposure025.png">ROI</a><br>'
                               f'<a href="{key}/scene.json">场景</a> · <a href="{key}/render.log">日志</a></td>')
        if all((group,m) in arrays for m in g.MODES[:2]):
            delta_rgb = arrays[(group,'global_conditional')] - arrays[(group,'classic_global')]
            derived = stage/'differences'/group
            derived.mkdir(parents=True, exist_ok=True)
            g.write_pfm(derived/'conditional_minus_classic.pfm', delta_rgb)
            g.write_png(derived/'difference.png', g.signed_color(delta_rgb.mean(axis=-1), .5))
            delta = comparison['roi']['global_conditional'] - comparison['roi']['classic_global']
            relative = delta / comparison['roi']['classic_global']
            comparison.update(delta=delta, relative_delta=relative, delta_ci95=None)
            entries.append(f'<td><a href="differences/{group}/difference.png"><img src="differences/{group}/difference.png" width="256" height="256"></a>'
                           f'<br>ROI Δ={delta:+.6g} ({relative:+.2%})<br><a href="differences/{group}/conditional_minus_classic.pfm">线性差分 PFM</a></td>')
        else:
            entries.append('<td>等待成对结果</td>')
        comparisons.append(comparison)
        cells.append('<tr>' + ''.join(entries) + '</tr>')
    complete = sum(r['status']=='complete' for r in rows.values())
    failed = [r for r in rows.values() if r['status']!='complete']
    status = dict(state='complete' if complete==len(jobs) else 'failed' if failed and not active else 'running',
                  completed=complete, total=len(jobs), failed=len(failed), active=list(active),
                  updated_utc=datetime.now(timezone.utc).isoformat())
    atomic_save(stage/'status.json', status)
    atomic_save(stage/'summary.json', dict(**status, resolution=[256,256], spp=256, seeds_per_image=1,
                display_exposure=EXPOSURE, difference_scale=[-.5,.5],
                inference='Single-seed presentation images; no repeat-based confidence interval.',
                comparisons=comparisons, results=list(rows.values())))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>U 槽 · 256×256 · 256 spp</title>
<style>body{font:15px system-ui;margin:28px;color:#243047;background:#f5f7fa}p{max-width:1150px;line-height:1.75}
table{border-collapse:collapse;background:white}th,td{padding:12px;border:1px solid #ccd4df;text-align:center;min-width:100px}
img{display:block;max-width:none}a{color:#175ab0}small{color:#596578}</style>
<h1>U 槽：256×256 / 256 spp</h1>
<p><a href="../comparison.html">全部实验</a> · <a href="../findings.html">预试分析</a> · <a href="../plan.html">实验方案</a> ·
<a href="summary.json">本批数值</a> · <a href="manifest.json">运行配置</a></p>
<p>固定 45° 视角，沿用已验证的 U 槽、面积光和 mirror albedo=0.85。四组 classic_global / global_conditional，加 A 组 classic_local，共 9 张。
每张原生 256×256、256 spp、1 个独立 seed。页面统一曝光 0.25 后 filmic 显示；原始 PNG 和线性 PFM 均保留。没有降噪或放大。</p>
<p>ROI 为固定均值几何的右内壁可见投影，统计在线性 RGB 上求均值。差分为 conditional − classic_global，色标统一为蓝 −0.5 / 白 0 / 红 +0.5，超出部分仅在显示中饱和。
本批用于观察图像细节；单 seed 不提供重复试验的置信区间，也不等同于方案中的 8-seed confirm 阶段。</p>'''
    page += f'<p>已完成 {complete} / {len(jobs)} 张；失败 {len(failed)} 张。更新于 {html.escape(status["updated_utc"])}</p>'
    page += '<table><tr><th>参数</th>' + ''.join(f'<th>{m}</th>' for m in g.MODES) + '<th>差分</th></tr>' + ''.join(cells) + '</table></html>'
    atomic_text(stage/'comparison.html', page)
    return status


def validate_all(out, jobs, manifest):
    passed = []
    for job in jobs:
        folder = out/STAGE/g.job_key(job)
        row = g.load(folder/'result.json')
        image = g.validate_output(folder, job)
        scene = g.load(folder/'scene.json')
        if row['status']!='complete' or g.digest(folder/'scene.json')!=row['scene_sha256']:
            raise RuntimeError(f'Invalid result/scene: {folder}')
        for name in ('render.png', 'exposure025.png', 'roi_exposure025.png'):
            data = (folder/name).read_bytes()
            if data[:8]!=b'\x89PNG\r\n\x1a\n' or struct.unpack('>II',data[16:24])!=(256,256):
                raise RuntimeError(f'Invalid native PNG: {folder/name}')
        if scene['renderer']['spp']!=256 or scene['camera']['resolution']!=[256,256]:
            raise RuntimeError('Resolution or spp mismatch')
        if not np.isclose(image[g.roi_mask(scene['camera'])].mean(), row['roi_mean'], rtol=0, atol=1e-12):
            raise RuntimeError('ROI recomputation mismatch')
        passed.append(dict(job=g.job_key(job), png_sha256=g.digest(folder/'render.png'),
                           pfm_sha256=g.digest(folder/'render.pfm')))
    for group in g.GROUPS:
        scenes = []
        for mode in g.MODES[:2]:
            job = next(j for j in jobs if j['group']==group and j['mode']==mode)
            scene = copy.deepcopy(g.load(out/STAGE/g.job_key(job)/'scene.json'))
            scene['renderer'].pop('output_directory')
            medium = scene['media'][0]
            medium.pop('type')
            for key in list(medium):
                if key.startswith('tracking_'):
                    medium.pop(key)
            scenes.append(scene)
        if scenes[0] != scenes[1]:
            raise RuntimeError(f'Unmatched scene pair: {group}')
    for asset, expected in manifest['assets']['main'].items():
        if g.digest(out/'fields/main'/asset)!=expected:
            raise RuntimeError(f'Changed asset: {asset}')
    if g.digest(Path(manifest['executable'])) != manifest['executable_sha256']:
        raise RuntimeError('Changed renderer')
    page = (out/STAGE/'comparison.html').read_text(encoding='utf-8')
    links = re.findall(r'(?:href|src)="([^"]+)"', page)
    broken = [p for p in links if not (out/STAGE/p).exists()]
    if broken:
        raise RuntimeError(f'Broken report links: {broken}')
    atomic_save(out/STAGE/'validation.json', dict(state='passed', completed=len(passed),
                matched_scene_pairs=4, local_links_checked=len(links),
                checks=['native PNG and PFM dimensions', '256/256 spp completion and error logs',
                        'finite nonnegative linear radiance', 'scene hashes and matched settings',
                        'ROI recomputation', 'frozen renderer and assets', 'report links'], images=passed))


def link_from_index(out):
    for name in ('comparison.html', 'findings.html'):
        path = out/name
        if not path.exists():
            continue
        page = path.read_text(encoding='utf-8')
        marker = '<!-- native256 -->'
        if marker not in page:
            pos = page.index('</h1>') + len('</h1>')
            page = page[:pos] + marker + '<p><a href="res256_spp256/comparison.html">256×256 / 256 spp 正式出图（9 张，逐张更新）</a></p>' + page[pos:]
            atomic_text(path, page)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=202610281)
    parser.add_argument('--timeout', type=float, default=43200)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    out = g.OUT
    stage = out/STAGE
    stage.mkdir(parents=True, exist_ok=True)
    jobs = g.jobs_for('matrix', size=256, spp=256, seeds=[args.seed])
    if args.report_only:
        manifest = g.load(stage/'manifest.json')
        publish(out, manifest['jobs'])
        validate_all(out, manifest['jobs'], manifest)
        link_from_index(out)
        return
    if args.threads < 1 or args.workers < 1:
        parser.error('threads and workers must be positive')
    g.save = atomic_save
    field, meta = g.fixture(out)
    # All jobs use the same prevalidated fixture; workers must not rewrite it.
    def fixed_fixture(requested_out, fine=False, expanded=False):
        if requested_out != out or fine or expanded:
            raise ValueError('This presentation batch uses only the main fixture')
        return field, meta
    g.fixture = fixed_fixture
    exe = g.ROOT/'build/Release/tungsten-endpoint-sync.exe'
    if g.digest(exe)!=EXE_HASH:
        raise RuntimeError('Frozen renderer hash mismatch')
    manifest = dict(stage=STAGE, jobs=jobs, executable=exe.as_posix(), executable_sha256=EXE_HASH,
                    assets={'main':{p:g.digest(field/p) for p in ('u_groove_sdf.vdb','u_groove.nvdb')}},
                    threads=args.threads, workers=args.workers, timeout=args.timeout,
                    tool_sha256=g.digest(Path(g.__file__)), runner_sha256=g.digest(Path(__file__)),
                    baker_sha256=g.digest(g.ROOT/'build/Release/bake_contact_fields.exe'),
                    purpose='Native presentation images, one seed per image; not 8-seed confirmation',
                    display_exposure=EXPOSURE)
    manifest_path = stage/'manifest.json'
    if manifest_path.exists() and g.load(manifest_path)!=manifest:
        raise RuntimeError('Existing batch manifest differs; preserve this batch')
    atomic_save(manifest_path, manifest)
    (stage/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for job in jobs:
        scene = g.make_scene(out, **{k:job[k] for k in ('group','angle','mode','size','spp','variant')})
        folder = stage/g.job_key(job)
        scene['renderer']['output_directory'] = folder.as_posix()
        if (folder/'scene.json').exists() and g.load(folder/'scene.json')!=scene:
            raise RuntimeError(f'Existing scene differs: {folder}')
        atomic_save(folder/'scene.json', scene)
    publish(out, jobs)
    link_from_index(out)
    # Cheap classic images first, then four independent conditional renders.
    ordered = sorted(jobs, key=lambda j:j['mode']=='global_conditional')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(g.render, out, STAGE, job, exe, EXE_HASH, args.threads, args.timeout):job for job in ordered}
        while pending:
            done, _ = wait(pending, timeout=45, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
                del pending[future]
            active = [g.job_key(job) for future,job in pending.items() if future.running()]
            state = publish(out, jobs, active)
            print(f'PROGRESS {state["completed"]}/{len(jobs)} complete; {state["failed"]} failed; active={active}', flush=True)
    state = publish(out, jobs)
    if state['completed']!=len(jobs):
        raise SystemExit('Render failures retained for diagnosis')
    validate_all(out, jobs, manifest)
    g.report(out)
    link_from_index(out)
    print(f'COMPLETE {stage/"comparison.html"}', flush=True)


if __name__=='__main__':
    main()
