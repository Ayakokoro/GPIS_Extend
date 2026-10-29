"""Complete interrupted native U-groove renders with saved sampler checkpoints.

The frozen batch manifest and seven completed images are retained. Runtime scenes
only enable checkpoint I/O; the recovery manifest records those overrides and the
actual 16-thread commands. No renderer or random-field code is changed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import ctypes
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time

import numpy as np

import render_conditional_groove_final as p
import test_conditional_groove as g

OUT = g.OUT
STAGE = OUT/p.STAGE
OVERRIDES = dict(enable_resume_render=True, checkpoint_interval='1m',
                 resume_render_file='TungstenRenderState.dat')
THREADS = 16


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def archive_attempt(folder):
    archive = STAGE/'interrupted'/stamp()/folder.relative_to(STAGE)
    archive.mkdir(parents=True, exist_ok=True)
    for name in ('render.log', 'scene.json', 'scene.checkpoint.json', 'result.json'):
        source = folder/name
        if source.exists():
            target = archive/('previous_result.json' if name=='result.json' else name)
            shutil.copy2(source, target)
    g.save(archive/'attempt.json', dict(state='interrupted_or_incomplete',
           retained_checkpoint=(folder/'TungstenRenderState.dat').exists(),
           folder=folder.relative_to(OUT).as_posix()))


def publish(jobs, active=()):
    state = p.publish(OUT, jobs, active)
    page = (STAGE/'comparison.html').read_text(encoding='utf-8')
    for job in jobs:
        key = g.job_key(job)
        if (STAGE/key/'scene.checkpoint.json').exists():
            page = page.replace(f'href="{key}/scene.json"', f'href="{key}/scene.checkpoint.json"')
    note = '<p>中断的 B、C 两张已转为保存采样状态的后台任务；实际运行配置与原始场景的对应关系见 <a href="recovery_manifest.json">恢复记录</a>。</p>'
    page = page.replace('<table>', note+'<table>', 1)
    p.atomic_text(STAGE/'comparison.html', page)
    return state


def render_job(job, manifest):
    folder = STAGE/g.job_key(job)
    baseline = g.load(folder/'scene.json')
    runtime = copy.deepcopy(baseline)
    runtime['renderer'].update(OVERRIDES)
    runtime_path = folder/'scene.checkpoint.json'
    if runtime_path.exists() and g.load(runtime_path)!=runtime:
        raise RuntimeError('Checkpoint scene changed; preserve the saved state')
    archive_attempt(folder)
    g.save(runtime_path, runtime)
    seed = job['seed_id'] + 10000000*g.MODES.index(job['mode'])
    command = [manifest['executable'], '-t', str(THREADS), '-s', str(seed),
               '-d', str(folder), str(runtime_path)]
    print(f'START {g.job_key(job)}; checkpoint={(folder/"TungstenRenderState.dat").exists()}', flush=True)
    environment = os.environ.copy()
    for name in ('TUNGSTEN_GPIS_CONDITIONAL_PROFILE','TUNGSTEN_GPIS_PROFILE','TUNGSTEN_GPIS_FS_PROFILE'):
        environment.pop(name, None)
    start = time.monotonic()
    status, error, code = 'complete', None, None
    try:
        with (folder/'render.log').open('wb') as log:
            proc = subprocess.run(command, cwd=g.ROOT, stdout=log, stderr=subprocess.STDOUT,
                                  env=environment, timeout=43200)
        code = proc.returncode
        if code:
            raise RuntimeError(f'Exit code {code}')
        image = g.validate_output(folder, job)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        status, error = 'failed', str(exc)
    row = dict(job, status=status, error=error, returncode=code, render_seed=seed,
               executable_sha256=manifest['executable_sha256'], command=command,
               seconds=time.monotonic()-start, scene=runtime_path.relative_to(OUT).as_posix(),
               scene_sha256=g.digest(runtime_path),
               reference_scene=(folder/'scene.json').relative_to(OUT).as_posix(),
               reference_scene_sha256=g.digest(folder/'scene.json'),
               renderer_io_overrides=OVERRIDES, execution_threads=THREADS,
               recovery_manifest=(STAGE/'recovery_manifest.json').relative_to(OUT).as_posix())
    if status=='complete':
        mask = g.roi_mask(runtime['camera'])
        row.update(roi_pixels=int(mask.sum()), roi_mean=float(image[mask].mean()),
                   image_mean=float(image.mean()), image_maximum=float(image.max()))
        display = g.filmic(image)
        edge = mask & ~(np.roll(mask,1,0)&np.roll(mask,-1,0)&np.roll(mask,1,1)&np.roll(mask,-1,1))
        display[edge] = [1,.15,.1]
        g.write_png(folder/'roi.png', display)
    g.save(folder/'result.json', row)
    print(f'{status.upper()} {g.job_key(job)}; ROI={row.get("roi_mean")}; error={error}', flush=True)
    return row


def validate(jobs, manifest, recovery):
    records, scenes = [], {}
    for job in jobs:
        folder = STAGE/g.job_key(job)
        row = g.load(folder/'result.json')
        if row['status']!='complete':
            raise RuntimeError(f'Incomplete image: {folder}')
        image = g.validate_output(folder, job)
        actual_path = OUT/row['scene']
        if g.digest(actual_path)!=row['scene_sha256']:
            raise RuntimeError('Actual scene hash mismatch')
        actual = g.load(actual_path)
        baseline = g.load(folder/'scene.json')
        normalized = copy.deepcopy(actual)
        if 'renderer_io_overrides' in row:
            if row['renderer_io_overrides']!=OVERRIDES:
                raise RuntimeError('Unexpected runtime overrides')
            for key, value in OVERRIDES.items():
                if actual['renderer'].get(key)!=value:
                    raise RuntimeError('Runtime scene override mismatch')
                if key in baseline['renderer']:
                    normalized['renderer'][key] = baseline['renderer'][key]
                else:
                    normalized['renderer'].pop(key)
        if normalized!=baseline:
            raise RuntimeError('Render differs from frozen scene beyond checkpoint I/O')
        for name in ('render.png', 'exposure025.png', 'roi_exposure025.png'):
            data = (folder/name).read_bytes()
            if data[:8]!=b'\x89PNG\r\n\x1a\n' or struct.unpack('>II', data[16:24])!=(256,256):
                raise RuntimeError(f'Wrong PNG dimensions: {folder/name}')
        if actual['camera']['resolution']!=[256,256] or actual['renderer']['spp']!=256:
            raise RuntimeError('Resolution/spp mismatch')
        if not np.isclose(image[g.roi_mask(actual['camera'])].mean(), row['roi_mean'], rtol=0, atol=1e-12):
            raise RuntimeError('ROI recomputation mismatch')
        normalized['renderer'].pop('output_directory')
        medium = normalized['media'][0]
        medium.pop('type')
        for key in list(medium):
            if key.startswith('tracking_'):
                medium.pop(key)
        scenes[(job['group'],job['mode'])] = normalized
        records.append(dict(job=g.job_key(job), actual_scene=row['scene'],
                             png_sha256=g.digest(folder/'render.png'), pfm_sha256=g.digest(folder/'render.pfm')))
    for group in g.GROUPS:
        if scenes[(group,'classic_global')]!=scenes[(group,'global_conditional')]:
            raise RuntimeError(f'Unmatched physical scene pair: {group}')
    for asset, digest in manifest['assets']['main'].items():
        if g.digest(OUT/'fields/main'/asset)!=digest:
            raise RuntimeError('Changed field asset')
    if g.digest(Path(manifest['executable']))!=manifest['executable_sha256']:
        raise RuntimeError('Changed renderer')
    for path, digest in recovery['preserved_files'].items():
        if g.digest(OUT/path)!=digest:
            raise RuntimeError(f'Previously completed artifact changed: {path}')
    page = (STAGE/'comparison.html').read_text(encoding='utf-8')
    links = re.findall(r'(?:href|src)="([^"]+)"', page)
    if any(not (STAGE/link).exists() for link in links):
        raise RuntimeError('Broken local report link')
    g.save(STAGE/'validation.json', dict(state='passed', completed=9, matched_scene_pairs=4,
           preserved_files_checked=len(recovery['preserved_files']), local_links_checked=len(links),
           checks=['256/256 spp logs', 'native PNG/PFM dimensions and finite radiance',
                   'actual runtime scenes differ only in checkpoint I/O', 'ROI recomputation',
                   'frozen renderer and fields', 'seven earlier images unchanged', 'report links'], images=records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    g.save = p.atomic_save
    manifest = g.load(STAGE/'manifest.json')
    jobs = manifest['jobs']
    if g.digest(Path(manifest['executable']))!=p.EXE_HASH:
        raise RuntimeError('Frozen renderer mismatch')
    recovery_path = STAGE/'recovery_manifest.json'
    if recovery_path.exists():
        recovery = g.load(recovery_path)
        if recovery['runner_sha256']!=g.digest(Path(__file__)):
            raise RuntimeError('Recovery runner changed')
        if recovery['original_manifest_sha256']!=g.digest(STAGE/'manifest.json'):
            raise RuntimeError('Frozen batch manifest changed')
    else:
        preserved, pending = {}, []
        for job in jobs:
            folder = STAGE/g.job_key(job)
            if (folder/'result.json').exists() and g.load(folder/'result.json')['status']=='complete':
                g.validate_output(folder, job)
                for name in ('result.json','scene.json','render.png','render.pfm'):
                    path = folder/name
                    preserved[path.relative_to(OUT).as_posix()] = g.digest(path)
            else:
                pending.append(job)
        recovery = dict(original_manifest_sha256=g.digest(STAGE/'manifest.json'),
                        runner_sha256=g.digest(Path(__file__)), jobs=pending, threads=THREADS,
                        workers=2, renderer_io_overrides=OVERRIDES, preserved_files=preserved,
                        explanation='Interrupted runs had no saved sample state; rerender only those two with checkpoint I/O.')
        g.save(recovery_path, recovery)
        (STAGE/'recovery_runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    pending = [j for j in jobs if not (STAGE/g.job_key(j)/'result.json').exists()
               or g.load(STAGE/g.job_key(j)/'result.json')['status']!='complete']
    if args.report_only and pending:
        raise RuntimeError('Pending renders still exist')
    keep_awake = ctypes.windll.kernel32 if os.name=='nt' else None
    if keep_awake:
        keep_awake.SetThreadExecutionState(0x80000001)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(render_job, job, manifest):job for job in pending}
            while futures:
                done, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    del futures[future]
                active = [g.job_key(j) for j in futures.values()]
                state = publish(jobs, active)
                print(f'PROGRESS {state["completed"]}/9; failed={state["failed"]}', flush=True)
                g.report(OUT)
                p.link_from_index(OUT)
        state = publish(jobs)
        if state['completed']!=9:
            raise RuntimeError('Incomplete renders retained for diagnosis')
        validate(jobs, manifest, recovery)
        g.report(OUT)
        p.link_from_index(OUT)
        print('COMPLETE: all 9 native images validated', flush=True)
    finally:
        if keep_awake:
            keep_awake.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    main()
