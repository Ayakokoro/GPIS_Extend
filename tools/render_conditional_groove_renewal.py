"""Add four matched 3D sparse-convolution Renewal+ renders to the U-groove batch.

Uses the frozen executable and original field; no renderer/evaluator changes.
The original nine renders and their manifest remain immutable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import ctypes
from datetime import datetime, timezone
import html
from html.parser import HTMLParser
import os
from pathlib import Path
import shutil
import struct
import subprocess
import time
from urllib.parse import unquote

import numpy as np

import render_conditional_groove_final as p
import test_conditional_groove as g

OUT = g.OUT
STAGE = OUT/p.STAGE
MODE = 'gpis_renewal_plus'
SEED = 232610281
THREADS = 8
WORKERS = 4
MODES = ('classic_global', 'global_conditional', MODE, 'classic_local')
LABELS = {m:m for m in MODES} | {MODE:'GPIS renewal+ (3D sparse convolution)'}
PAIRS = [('global_conditional','classic_global','conditional_minus_classic'),
         (MODE,'classic_global','renewal_plus_minus_classic'),
         (MODE,'global_conditional','renewal_plus_minus_conditional')]


def make_scene(group):
    source = STAGE/group/'view45/total/classic_global/seed202610281/scene.json'
    scene = g.load(source)
    m = scene['media'][0]
    ell = m['gaussian_process']['covariance']['lengthScale']
    scene['media'] = [dict(name=m['name'], type='sparse_conv_noise', max_bounces=m['max_bounces'],
        gaussian_process=copy.deepcopy(m['gaussian_process']),
        phase_function={'type':'brdf','bsdf':copy.deepcopy(m['material'])},
        sigma_a=0, sigma_s=1, density=1, step_size=min(ell*.25,g.W/32), min_step=8,
        seed=17429, impulse_density=10, single_realization=False, correlation_context='renewal+',
        intersect_method='gp_discrete', normal_method='conditioned_gaussian',
        **{'1D_sampling':False, 'isotropic_3D_sampling':False, '1D_sampling_scheme':'uni',
           '1D_gradient_correlationXY':False})]
    scene['renderer'].update(output_directory=(STAGE/g.job_key(job_for(group))).as_posix(),
                             checkpoint_interval='1m', enable_resume_render=False)
    return scene


def job_for(group):
    return dict(group=group, angle=45, mode=MODE, variant='total', size=256, spp=256, seed_id=202610281)


def prepare():
    baseline = g.load(STAGE/'manifest.json')
    if g.digest(Path(baseline['executable'])) != p.EXE_HASH:
        raise RuntimeError('Frozen renderer changed')
    path = STAGE/'renewal_plus_manifest.json'
    if path.exists():
        manifest = g.load(path)
        if manifest['runner_sha256'] != g.digest(Path(__file__)):
            raise RuntimeError('Renewal runner changed; preserve the existing run')
    else:
        preserved = {}
        for job in baseline['jobs']:
            folder = STAGE/g.job_key(job)
            row = g.load(folder/'result.json')
            if row['status'] != 'complete':
                raise RuntimeError('Baseline batch is incomplete')
            g.validate_output(folder, job)
            for name in ('scene.json','result.json','render.png','render.pfm','render.log'):
                q = folder/name
                preserved[q.relative_to(OUT).as_posix()] = g.digest(q)
            actual = OUT/row['scene']
            preserved[actual.relative_to(OUT).as_posix()] = g.digest(actual)
        for name in ('manifest.json','validation.json','recovery_manifest.json'):
            q = STAGE/name
            preserved[q.relative_to(OUT).as_posix()] = g.digest(q)
        for name in ('comparison.html','summary.json','status.json'):
            shutil.copy2(STAGE/name, STAGE/(Path(name).stem+'_before_renewal'+Path(name).suffix))
        manifest = dict(jobs=[job_for(k) for k in g.GROUPS], executable=baseline['executable'],
            executable_sha256=p.EXE_HASH, runner_sha256=g.digest(Path(__file__)),
            helper_sha256=g.digest(Path(g.__file__)), baseline_manifest_sha256=g.digest(STAGE/'manifest.json'),
            assets=baseline['assets'], preserved_files=preserved, threads=THREADS, workers=WORKERS,
            render_seed=SEED, display_exposure=p.EXPOSURE, difference_scale=[-.5,.5],
            backend='Existing 3D sparse_conv_noise; correlation_context=renewal+; full gradient conditioning.',
            numerical_settings={'impulse_density':10,'max_step':'min(ell/4,w/32) = 0.003125','min_step':8},
            step_reason='Cap the previous shaderball ell/4 step at the groove voxel spacing to resolve the narrow mean geometry.',
            caveat='Finite-density, discretely marched sparse convolution; not an exact GP reference. One seed, no repeat-based CI.',
            checkpoint_note='Intermediate images only; native sampler resume is disabled because its scene checksum is unstable.')
        g.save(path, manifest)
        (STAGE/'renewal_plus_runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    if manifest['baseline_manifest_sha256'] != g.digest(STAGE/'manifest.json'):
        raise RuntimeError('Baseline manifest changed')
    if manifest['helper_sha256'] != g.digest(Path(g.__file__)):
        raise RuntimeError('Original experiment helper changed')
    for job in manifest['jobs']:
        folder = STAGE/g.job_key(job)
        scene = make_scene(job['group'])
        if (folder/'scene.json').exists() and g.load(folder/'scene.json') != scene:
            raise RuntimeError('Existing Renewal+ scene differs')
        g.save(folder/'scene.json', scene)
    return baseline, manifest


def render(job, manifest):
    folder = STAGE/g.job_key(job)
    command = [manifest['executable'],'-t',str(THREADS),'-s',str(SEED),'-d',str(folder),str(folder/'scene.json')]
    env = os.environ.copy()
    for key in ('TUNGSTEN_GPIS_CONDITIONAL_PROFILE','TUNGSTEN_GPIS_PROFILE','TUNGSTEN_GPIS_FS_PROFILE'):
        env.pop(key,None)
    start = time.monotonic()
    status, error, code = 'complete', None, None
    print('START '+g.job_key(job),flush=True)
    try:
        with (folder/'render.log').open('wb') as log:
            proc = subprocess.run(command,cwd=g.ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=43200)
        code = proc.returncode
        if code:
            raise RuntimeError(f'Exit code {code}')
        image = g.validate_output(folder,job)
    except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as exc:
        status,error = 'failed',str(exc)
    row = dict(job,status=status,error=error,returncode=code,render_seed=SEED,seconds=time.monotonic()-start,
        command=command,executable_sha256=p.EXE_HASH,scene=(folder/'scene.json').relative_to(OUT).as_posix(),
        scene_sha256=g.digest(folder/'scene.json'),execution_threads=THREADS,
        mode_description=manifest['backend'],manifest='res256_spp256/renewal_plus_manifest.json')
    if status == 'complete':
        mask = g.roi_mask(g.load(folder/'scene.json')['camera'])
        row.update(roi_pixels=int(mask.sum()),roi_mean=float(image[mask].mean()),
                   image_mean=float(image.mean()),image_maximum=float(image.max()))
        boundary = mask & ~(np.roll(mask,1,0)&np.roll(mask,-1,0)&np.roll(mask,1,1)&np.roll(mask,-1,1))
        for exposure,name in ((1,'roi.png'),(p.EXPOSURE,'roi_exposure025.png')):
            view = g.filmic(image*exposure)
            view[boundary] = [1,.15,.1]
            g.write_png(folder/name,view)
        g.write_png(folder/'exposure025.png',g.filmic(image*p.EXPOSURE))
    g.save(folder/'result.json',row)
    print(f'{status.upper()} {job["group"]} seconds={row["seconds"]:.1f} ROI={row.get("roi_mean")} error={error}',flush=True)
    return row


def publish(baseline, manifest, active=()):
    jobs = baseline['jobs']+manifest['jobs']
    rows, arrays = {}, {}
    for job in jobs:
        key = g.job_key(job)
        folder = STAGE/key
        if (folder/'result.json').exists():
            row = g.load(folder/'result.json')
            rows[(job['group'],job['mode'])] = row
            if row['status']=='complete': arrays[(job['group'],job['mode'])] = g.read_image(folder/'render.pfm')
    complete = sum(r['status']=='complete' for r in rows.values())
    failed = sum(r['status']!='complete' for r in rows.values())
    status = dict(state='complete' if complete==13 else 'failed' if failed and not active else 'running',
                  completed=complete,total=13,failed=failed,active=list(active),updated_utc=datetime.now(timezone.utc).isoformat())
    comparisons, images_table, differences_table = [], [], []
    for group,(sigma_w,ell_w) in g.GROUPS.items():
        label = f'{group}<br>σ={sigma_w*g.W:g}<br>ℓ={ell_w*g.W:g}'
        cells,dcells = [f'<th>{label}</th>'],[f'<th>{label}</th>']
        comparison = dict(group=group,sigma=sigma_w*g.W,ell=ell_w*g.W,roi={},differences={})
        for mode in MODES:
            row = rows.get((group,mode))
            if row is None:
                cells.append('<td>渲染中</td>' if mode==MODE else '<td>—</td>')
            elif row['status']!='complete':
                cells.append('<td>失败：'+html.escape(str(row['error']))+'</td>')
            else:
                key = g.job_key(row)
                actual_scene = (OUT/row['scene']).relative_to(STAGE).as_posix()
                comparison['roi'][mode] = row['roi_mean']
                cells.append(f'<td><a href="{key}/exposure025.png"><img src="{key}/exposure025.png" width="256" height="256"></a>'
                    f'<br>ROI {row["roi_mean"]:.6g}<br><a href="{key}/render.png">原始 PNG</a> · <a href="{key}/render.pfm">线性 PFM</a>'
                    f' · <a href="{key}/roi_exposure025.png">ROI</a><br><a href="{actual_scene}">场景</a> · <a href="{key}/render.log">日志</a></td>')
        for a,b,name in PAIRS:
            if (group,a) not in arrays or (group,b) not in arrays:
                dcells.append('<td>等待成对结果</td>'); continue
            diff = arrays[(group,a)]-arrays[(group,b)]
            derived = STAGE/'renewal_plus_differences'/group
            png,pfm = derived/(name+'.png'),derived/(name+'.pfm')
            g.write_png(png,g.signed_color(diff.mean(axis=-1),.5)); g.write_pfm(pfm,diff)
            delta = comparison['roi'][a]-comparison['roi'][b]
            relative = delta/comparison['roi'][b]
            comparison['differences'][name] = dict(delta=delta,relative_delta=relative,ci95=None)
            png,pfm = png.relative_to(STAGE).as_posix(),pfm.relative_to(STAGE).as_posix()
            dcells.append(f'<td><a href="{png}"><img src="{png}" width="256" height="256"></a><br>ROI Δ={delta:+.6g} ({relative:+.2%})<br><a href="{pfm}">线性差分 PFM</a></td>')
        if 'conditional_minus_classic' in comparison['differences']:
            legacy = comparison['differences']['conditional_minus_classic']
            comparison.update(delta=legacy['delta'],relative_delta=legacy['relative_delta'],delta_ci95=None)
        comparisons.append(comparison)
        images_table.append('<tr>'+''.join(cells)+'</tr>'); differences_table.append('<tr>'+''.join(dcells)+'</tr>')
    g.save(STAGE/'status.json',status)
    g.save(STAGE/'summary.json',dict(**status,resolution=[256,256],spp=256,seeds_per_image=1,display_exposure=p.EXPOSURE,
        difference_scale=[-.5,.5],inference=manifest['caveat'],comparisons=comparisons,results=list(rows.values())))
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>U 槽 · 256×256 · 256 spp · Renewal+</title>
<style>body{font:15px system-ui;margin:28px;color:#243047;background:#f5f7fa}p{max-width:1180px;line-height:1.75}table{border-collapse:collapse;background:white}th,td{padding:12px;border:1px solid #ccd4df;text-align:center}img{display:block;max-width:none}a{color:#175ab0}</style>
<h1>U 槽：256×256 / 256 spp</h1><p><a href="../comparison.html">全部实验</a> · <a href="../plan.html">实验方案</a> · <a href="summary.json">线性统计</a> · <a href="renewal_plus_manifest.json">Renewal+ 配置</a> · <a href="comparison_before_renewal.html">原始九张对比</a></p>
<p>固定 45° 视角、同一 U 槽 SDF、面积光和 mirror albedo=0.85。四组 classic_global / global_conditional / GPIS renewal+，加 A 组 classic_local，共 13 张。
每张原生 256×256、256 spp、1 个独立 seed。统一曝光 0.25 后 filmic 显示，保留原始 PNG 和线性 PFM，没有降噪或放大。</p>
<p>GPIS renewal+ 沿用前面 GPIS 的 3D sparse_conv_noise 求值，impulse_density=10，correlation_context=renewal+，1D_sampling=false。
新光线段在起点条件化场值和完整梯度。最大步长 min(ℓ/4,w/32)=0.003125，min_step=8；用槽的体素间距限制步长以解析窄槽。
这是有限脉冲密度与离散步进的近似，单 seed 不提供重复试验置信区间，也不等同于完整 GP 后验或 8-seed confirm 阶段。</p>
<p>ROI 仍为预先固定的右内壁投影，在线性 RGB 上求均值。下表各差分均使用蓝 −0.5 / 白 0 / 红 +0.5 的固定色标；仅显示饱和，数值不截断。</p>'''
    page += f'<p>已完成 {complete} / 13 张；失败 {failed} 张。更新于 {status["updated_utc"]}</p>'
    page += '<table><tr><th>参数</th>'+''.join(f'<th>{LABELS[m]}</th>' for m in MODES)+'</tr>'+''.join(images_table)+'</table>'
    page += '<p>差分：左项减右项</p><table><tr><th>参数</th>'+''.join(f'<th>{LABELS[a]} − {LABELS[b]}</th>' for a,b,_ in PAIRS)+'</tr>'+''.join(differences_table)+'</table></html>'
    p.atomic_text(STAGE/'comparison.html',page)
    g.MODES = ('classic_global','global_conditional','classic_local',MODE)
    g.report(OUT)
    p.link_from_index(OUT)
    index = (OUT/'comparison.html').read_text(encoding='utf-8').replace('（9 张，逐张更新）','（13 张，含 GPIS renewal+）')
    index = index.replace('<h1>U 槽：classic 与 global_conditional</h1>','<h1>U 槽：classic、global_conditional 与 GPIS renewal+</h1>')
    index = index.replace(f'<th>{MODE}</th>',f'<th>{LABELS[MODE]}</th>')
    p.atomic_text(OUT/'comparison.html',index)
    return status


class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links=[]
    def handle_starttag(self,tag,attrs): self.links.extend(v for k,v in attrs if k in ('href','src') and v)


def validate(baseline, manifest):
    for relative,digest in manifest['preserved_files'].items():
        if g.digest(OUT/relative)!=digest: raise RuntimeError('Original artifact changed: '+relative)
    if g.digest(Path(manifest['executable']))!=p.EXE_HASH: raise RuntimeError('Renderer changed')
    for name,digest in manifest['assets']['main'].items():
        if g.digest(OUT/'fields/main'/name)!=digest: raise RuntimeError('Field changed')
    results = []
    for job in manifest['jobs']:
        folder = STAGE/g.job_key(job)
        scene,row = g.load(folder/'scene.json'),g.load(folder/'result.json')
        if scene!=make_scene(job['group']) or row['scene_sha256']!=g.digest(folder/'scene.json'):
            raise RuntimeError('Renewal+ scene mismatch')
        image = g.validate_output(folder,job)
        if not np.isclose(image[g.roi_mask(scene['camera'])].mean(),row['roi_mean'],rtol=0,atol=1e-12):
            raise RuntimeError('ROI mismatch')
        for name in ('render.png','exposure025.png','roi.png','roi_exposure025.png'):
            data=(folder/name).read_bytes()
            if data[:8]!=b'\x89PNG\r\n\x1a\n' or struct.unpack('>II',data[16:24])!=(256,256):
                raise RuntimeError('PNG dimensions mismatch')
        results.append(dict(group=job['group'],pfm_sha256=g.digest(folder/'render.pfm'),png_sha256=g.digest(folder/'render.png')))
    counts = {}
    for page in (STAGE/'comparison.html',OUT/'comparison.html'):
        parser=Links(); parser.feed(page.read_text(encoding='utf-8'))
        if any(not (page.parent/unquote(link)).exists() for link in parser.links):
            raise RuntimeError('Broken comparison link')
        counts[page.relative_to(OUT).as_posix()] = len(parser.links)
    g.save(STAGE/'renewal_plus_validation.json',dict(state='passed',completed=13,new_renders=4,
        preserved_files_checked=len(manifest['preserved_files']),matched_physical_scenes=4,local_links_checked=counts,
        checks=['all new logs 256/256 spp','native PNG/PFM dimensions and finite radiance','ROI recomputation',
                'matched frozen physical scene with only medium representation and checkpoint I/O changes',
                'original nine renders unchanged','frozen executable and fields','local comparison links'],images=results))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    g.save=p.atomic_save
    baseline,manifest=prepare()
    pending=[]
    for job in manifest['jobs']:
        record=STAGE/g.job_key(job)/'result.json'
        if not record.exists(): pending.append(job)
        elif g.load(record)['status']!='complete': raise RuntimeError('Failed run preserved: '+str(record))
        else: g.validate_output(record.parent,job)
    publish(baseline,manifest,[g.job_key(j) for j in pending])
    if args.prepare_only: return
    if args.report_only and pending: raise RuntimeError('Missing Renewal+ images')
    keep_awake=ctypes.windll.kernel32 if os.name=='nt' else None
    if keep_awake: keep_awake.SetThreadExecutionState(0x80000001)
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures={pool.submit(render,j,manifest):j for j in pending}
            while futures:
                done,_=wait(futures,timeout=30,return_when=FIRST_COMPLETED)
                for f in done: f.result(); del futures[f]
                state=publish(baseline,manifest,[g.job_key(j) for j in futures.values()])
                print(f'PROGRESS {state["completed"]}/13 failed={state["failed"]}',flush=True)
        state=publish(baseline,manifest)
        if state['completed']!=13: raise RuntimeError('Incomplete renders retained for diagnosis')
        validate(baseline,manifest)
        print('COMPLETE: all 13 native images validated',flush=True)
    finally:
        if keep_awake: keep_awake.SetThreadExecutionState(0x80000000)


if __name__=='__main__': main()
