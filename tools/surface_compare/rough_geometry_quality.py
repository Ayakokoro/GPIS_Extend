"""Resumable, equal-spp high-resolution rough-geometry comparison.

python -B tools/surface_compare/rough_geometry_quality.py run --size 512
python -B tools/surface_compare/rough_geometry_quality.py check-resume

Each completed spp level is immutable. The native resume state retains the
sample means, even/odd halves and Welford variance. All nine images advance to
the same spp before a comparison is published. A finite maximum is a compute
limit, not evidence of convergence.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import rough_geometry as rg

PLOT_PYTHON = Path('C:/Users/user/miniconda3/envs/gaussian_splatting/python.exe')
FILES = ('scene.json', 'render.png', 'render.pfm', 'samplesA.pfm',
         'samplesB.pfm', 'samplesVariance.pfm', 'render.log')
FAILURE = re.compile(r'Caught an internal error|unrecoverable|Gradient invalid|Unable to load|'
                     r'Wrong normal direction|Failed to sample gradient|Gradient zero|'
                     r'Clamp one (?:inf|nan) sample', re.I)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def replace_with_retry(temporary, path):
    # Windows readers / antivirus can briefly hold a destination without delete
    # sharing. Keep the last valid file visible and retry the atomic replacement.
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05 * 2**attempt)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    replace_with_retry(temporary, path)


@contextmanager
def job_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'worker.lock').open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


_blocked_progress_files = set()


def publish_status(root, status):
    # A failed progress-page refresh must not terminate a costly native render.
    # Scene, checkpoint and result writes still fail loudly through atomic_json.
    try:
        _publish_status(root, status)
    except PermissionError as error:
        key = str(root)
        if key not in _blocked_progress_files:
            print('Progress file is locked; rendering continues and refresh will be retried: '
                  + str(error), file=sys.stderr, flush=True)
            _blocked_progress_files.add(key)
    else:
        _blocked_progress_files.discard(str(root))


def _publish_status(root, status):
    status = dict(status, updated_utc=now(), worker_pid=os.getpid())
    atomic_json(root / 'status.json', status)
    config = rg.load(root / 'config.json') if (root / 'config.json').exists() else {}
    labels = {'running': '运行中', 'complete': '达到预设采样收敛阈值',
              'limit_reached': '达到采样上限，尚未满足全部收敛阈值',
              'failed': '运行失败，请查看日志', 'preparing': '准备中',
              'superseded': '已停止，已切换到新的分辨率任务'}
    label = labels.get(status['state'], status['state'])
    latest = status.get('latest_level')
    link = (f'<p><a href="{latest}/comparison.html">查看最新同分辨率、同 spp 对比</a></p>'
            if latest else '<p>第一档九张图尚未全部完成。</p>')
    if status.get('replacement_page'):
        link = f'<p><a href="{html.escape(status["replacement_page"], quote=True)}">查看新的对比任务</a></p>' + link
    detail = html.escape(status.get('detail', ''))
    phase = html.escape(status.get('phase', ''))
    validation_link = ('<a href="validation.json">GPIS 步长与计算域核验</a> · '
                       if (root / 'validation.json').exists() else '')
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60">
<title>粗糙平面与 U 槽 · 高质量比较进度</title>
<style>body{{font:17px/1.7 system-ui;max-width:1000px;margin:40px auto;padding:0 24px;color:#243044}}a{{color:#24699b}}code{{background:#eef1f5;padding:3px 5px}}</style>
<h1>粗糙平面与圆滑 U 槽 · 高质量比较</h1><h2>{label}</h2>
<p>三种模式：classic_global、global_conditional、GPIS global。统一 {config.get('size','—')}×{config.get('size','—')}；最低 {config.get('min_spp','—')} spp，必要时逐档加倍。</p>
<p>{phase}</p><p>{detail}</p>{link}<p>{validation_link}<a href="status.json">任务状态</a> · <a href="config.json">参数与收敛阈值</a></p>
<p>采样收敛只针对固定场景和固定 GPIS 数值设置。步长与计算域敏感性另行核验，稀疏卷积近似误差不包含在采样阈值内。原图不降噪，各模式使用相同显示增益。</p>
<p>更新时间（UTC）：{status['updated_utc']}</p></html>'''
    temporary = root / 'comparison.html.tmp'
    temporary.write_text(body, encoding='utf-8')
    replace_with_retry(temporary, root / 'comparison.html')


def snapshot_renderer(root):
    target = root / 'renderer'
    metadata = target / 'provenance.json'
    if metadata.exists():
        info = rg.load(metadata)
        for name, digest in info['files'].items():
            if rg.sha(target / name) != digest:
                raise RuntimeError('Renderer snapshot changed: ' + name)
        return target / 'tungsten.exe', info['files']['tungsten.exe']
    target.mkdir(parents=True, exist_ok=True)
    source = rg.ROOT / 'build/Release'
    paths = [source / 'tungsten.exe'] + sorted(source.glob('*.dll'))
    hashes = {}
    for path in paths:
        digest = rg.sha(path)
        shutil.copy2(path, target / path.name)
        if rg.sha(target / path.name) != digest or rg.sha(path) != digest:
            raise RuntimeError('Renderer changed while making snapshot')
        hashes[path.name] = digest
    atomic_json(metadata, dict(created_utc=now(), source=str(source), files=hashes))
    return target / 'tungsten.exe', hashes['tungsten.exe']


def resume_spp(path):
    if not path.exists():
        return 0
    with path.open('rb') as handle:
        header = handle.read(4096).split(b'\0', 1)[0]
    return int(json.loads(header)['current_spp'])


def arguments(config, stage, spp):
    return argparse.Namespace(stage=stage, size=config['size'], spp=spp,
        threads=config['threads'], seed=config['seed'], step=config['step'],
        domain_scale=config.get('domain_scale', 1.), material='lambert', cases=list(rg.CASES), modes=list(rg.MODES))


def resume_scene(config, stage, spp, case, mode):
    scene = rg.scene(arguments(config,stage,spp),case,mode)
    scene['renderer'].update(enable_resume_render=True,resume_render_file='resume.dat',
                             checkpoint_interval='5m',spp_step=2)
    return scene


def validate_result(folder, result):
    for name, digest in result['sha256'].items():
        if rg.sha(folder / name) != digest:
            raise RuntimeError('Changed result: ' + str(folder / name))
    rgb = rg.pfm(folder / 'render.pfm')
    if rgb.shape != (*result['resolution'], 3):
        raise RuntimeError('Unexpected image size')
    if not np.allclose((rg.pfm(folder / 'samplesA.pfm') + rg.pfm(folder / 'samplesB.pfm'))/2,
                       rgb, rtol=3e-6, atol=3e-7):
        raise RuntimeError('Split means do not match final image')
    rg.pfm(folder / 'samplesVariance.pfm')


def render_one(root, config, exe, version, case, mode, target, status, progress_root=None):
    progress_root = progress_root or root
    stage = (root / 'work').relative_to(rg.OUT).as_posix()
    args = arguments(config, stage, target)
    folder = rg.folder(args, case, mode)
    folder.mkdir(parents=True, exist_ok=True)
    scene = resume_scene(config,stage,target,case,mode)
    scene_path, state = folder / 'scene.json', folder / 'resume.dat'
    result_path = folder / 'result.json'
    if result_path.exists():
        old = rg.load(result_path)
        if old['renderer_sha256'] != version:
            raise RuntimeError('Renderer version mismatch')
        if old['spp'] == target:
            validate_result(folder, old)
            return folder, old
        if old['spp'] > target:
            raise RuntimeError('Requested spp is older than the working state')
    if scene_path.exists():
        old_scene = rg.load(scene_path)
        old_scene.pop('renderer'); physical_scene = copy.deepcopy(scene); physical_scene.pop('renderer')
        if old_scene != physical_scene:
            raise RuntimeError('Scene changed; use a new job directory')
    elif state.exists():
        raise RuntimeError('Resume state without scene provenance')
    start_spp = resume_spp(state)
    if start_spp > target:
        raise RuntimeError('Resume state exceeds requested spp')
    if result_path.exists() and start_spp < rg.load(result_path)['spp']:
        raise RuntimeError('Resume state is older than the validated image')
    if state.exists():
        shutil.copy2(state, folder / 'resume.previous.dat')
    atomic_json(scene_path, scene)
    log_path = folder / 'render.log'
    if log_path.exists():
        attempts = folder / 'attempts'; attempts.mkdir(exist_ok=True)
        shutil.copy2(log_path, attempts / f'{time.time_ns()}.log')
    start = time.monotonic()
    status.update(case=case, mode=mode, target_spp=target, completed_spp=start_spp,
                  detail=f'{case} / {mode}：{start_spp}/{target} spp')
    publish_status(progress_root, status)
    print(case, mode, f'{start_spp}/{target} spp', flush=True)
    try:
        with log_path.open('w', encoding='utf-8') as log:
            proc = subprocess.Popen([str(exe), '-t', str(config['threads']), '-s', str(config['seed']),
                                     '-d', str(folder), str(scene_path)], cwd=rg.ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                creationflags=getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
            try:
                for line in proc.stdout:
                    log.write(line); log.flush()
                    if start_spp and 'Resume unsuccessful' in line:
                        raise RuntimeError('Native resume refused an existing state')
                    if FAILURE.search(line):
                        raise RuntimeError(line.strip())
                    match = re.search(r'Completed (\d+)/(\d+) spp', line)
                    if match:
                        count = int(match[1])
                        remaining = ((time.monotonic()-start)*(target-count)/(count-start_spp)
                                     if count>start_spp else None)
                        status.update(completed_spp=count, remaining_job_seconds=remaining,
                                      detail=f'{case} / {mode}：{count}/{target} spp')
                        publish_status(progress_root, status)
                code = proc.wait()
            except BaseException:
                proc.terminate(); proc.wait()
                raise
        log = log_path.read_text(encoding='utf-8')
        if code or 'Finished render.' not in log or resume_spp(state) != target:
            atomic_json(folder/'failure.json',dict(returncode=code,target_spp=target,
                saved_spp=resume_spp(state),elapsed_seconds=time.monotonic()-start,log=str(log_path)))
            raise RuntimeError(f'Incomplete render (exit {code}): ' + str(log_path))
        if start_spp and 'Resume successful' not in log:
            raise RuntimeError('Missing native resume confirmation')
        discarded = re.search(r'discarded surface births: (\{.*\})', log)
        result = dict(seconds=time.monotonic()-start, renderer_sha256=version,
                      resolution=[config['size']]*2, spp=target, seed=config['seed'], resumed_from_spp=start_spp,
                      mean_luminance=float(rg.pfm(folder/'render.pfm')[...,0].mean()),
                      discarded_surface_births=json.loads(discarded[1]) if discarded else None,
                      zero_length_messages=log.count('zero-length'),
                      sha256={name:rg.sha(folder/name) for name in FILES})
        validate_result(folder, result)
        atomic_json(result_path, result)
        return folder, result
    finally:
        status.pop('remaining_job_seconds', None)


def snapshot_level(root, config, target, case, mode, folder, result):
    dest = root / 'levels' / f'spp{target:06d}' / case / mode / f'seed{config["seed"]}'
    if (dest / 'result.json').exists():
        saved = rg.load(dest / 'result.json'); validate_result(dest, saved)
        if saved != result:
            raise RuntimeError('Conflicting immutable level')
        return
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(folder/name, dest/name)
    atomic_json(dest/'result.json', result)


def block_average(values, count=32):
    h,w = values.shape
    return values.reshape(count,h//count,count,w//count).mean((1,3))


def filmic(values, gain):
    x=np.maximum(0,values*gain-.004)
    return np.clip(x*(6.2*x+.5)/(x*(6.2*x+1.7)+.06),0,1)


def convergence(root, config, target, previous):
    checks=[]
    level=root/'levels'/f'spp{target:06d}'
    for case in rg.CASES:
        for mode in rg.MODES:
            d=level/case/mode/f'seed{config["seed"]}'
            y=rg.pfm(d/'render.pfm')[...,0]
            v=rg.pfm(d/'samplesVariance.pfm')[...,0]
            a,b=[rg.pfm(d/f'samples{c}.pfm')[...,0] for c in ('A','B')]
            n=y.shape[0]; sl=np.s_[n//4:3*n//4,n//4:3*n//4]
            gain=rg.GAINS['lambert'][case]
            rows=[]
            for name,roi in (('full',np.s_[:,:]),('center_half',sl)):
                mean=float(y[roi].mean())
                relative_ci=float(1.96*np.sqrt(v[roi].sum())/y[roi].size/mean)
                display_rms=float(.5*np.sqrt(np.mean((filmic(a[roi],gain)-filmic(b[roi],gain))**2)))
                rows.append(dict(region=name,mean=mean,relative_mean_95_interval=relative_ci,
                                 display_noise_rms=display_rms))
            change=None
            if previous is not None:
                old=rg.pfm(root/'levels'/f'spp{previous:06d}'/case/mode/f'seed{config["seed"]}'/'render.pfm')[...,0]
                change=float(np.sqrt(np.mean(block_average(y-old)**2))/y.mean())
            passed=(target>=config['min_spp'] and change is not None and
                    change<=config['block_change_limit'] and
                    all(r['relative_mean_95_interval']<=config['mean_ci_limit'] and
                        r['display_noise_rms']<=config['display_noise_limit'] for r in rows))
            checks.append(dict(case=case,mode=mode,spp=target,previous_spp=previous,regions=rows,
                               block_relative_rms_change=change,passed=bool(passed)))
    return dict(passed=all(r['passed'] for r in checks),spp=target,previous_spp=previous,checks=checks,
        thresholds={k:config[k] for k in ('min_spp','display_noise_limit','block_change_limit','mean_ci_limit')},
        interpretation='Empirical Monte Carlo sampling criteria at fixed geometry and GPIS settings. '
                       'Pixel display noise is RMS on [0,1] after the common filmic transform, estimated from '
                       'independent even/odd halves. Block change uses a 32x32 grid in linear luminance. '
                       'Nested spp levels are not independent replicates. Finite step and sparse-convolution '
                       'approximation bias are not included.')


def report_level(root, config, target, check):
    level=root/'levels'/f'spp{target:06d}'
    atomic_json(level/'quality.json',check)
    shutil.copy2(root/'validation.json',level/'validation.json')
    subprocess.run([str(PLOT_PYTHON),'-B',str(rg.ROOT/'tools/surface_compare/rough_geometry.py'),
                    'report','--stage',level.relative_to(rg.OUT).as_posix(),'--seed',str(config['seed'])],
                   cwd=rg.ROOT,check=True)
    manifest=rg.load(level/'manifest.json')
    manifest['sampling_convergence']=check
    manifest['sampling_converged']=check['passed']
    manifest['gpis_step']=config['step']
    atomic_json(level/'manifest.json',manifest)
    page=level/'comparison.html'
    body=page.read_text(encoding='utf-8')
    label='达到预设采样收敛阈值' if check['passed'] else '阶段结果，尚未达到全部采样收敛阈值'
    notice=f'<div class="note"><b>{label}</b><p><a href="quality.json">逐图噪声与相邻 spp 检查</a> · <a href="../../comparison.html">任务进度</a></p><p>固定 GPIS 步长 {config["step"]:g}。采样收敛与步长误差分别记录。</p></div>'
    body=body.replace('</h1>','</h1>'+notice,1)
    page.write_text(body,encoding='utf-8')
    for link in re.findall(r'(?:href|src)="([^"]+)"',body):
        if not (level/link).is_file():
            raise RuntimeError('Broken report link: '+link)


def numerical_validation(root, config, exe, version, status):
    """Separate paired checks of GPIS step size and finite process-box extent."""
    folders = {}
    reuse = Path(config['reuse_numerics_from']) if config.get('reuse_numerics_from') else None
    size, spp = config['numerical_size'], config['numerical_spp']
    for variant, step, scale in (('base', config['step'], 1.),
                                  ('half_step', config['step']/2, 1.),
                                  ('larger_domain', config['step'], 1.5)):
        job = dict(config, size=size, step=step, domain_scale=scale)
        work_root = root/'numerics'/variant
        atomic_json(work_root/'config.json',job)
        status['phase'] = f'GPIS 数值核验：{variant}（{size}×{size}，{spp} spp）'
        for case in rg.CASES:
            if reuse:
                source_stage = (reuse/'numerics'/variant/'work').relative_to(rg.OUT).as_posix()
                folder = rg.folder(arguments(job,source_stage,spp),case,'sparse_3d_global')
                result = rg.load(folder/'result.json')
                if (result['renderer_sha256'] != version or result['resolution'] != [size,size]
                        or result['spp'] != spp or result['seed'] != config['seed']):
                    raise RuntimeError('Numerical reuse settings differ: '+str(folder))
                if rg.load(folder/'scene.json') != resume_scene(job,source_stage,spp,case,'sparse_3d_global'):
                    raise RuntimeError('Numerical reuse scene differs: '+str(folder))
                validate_result(folder,result)
                print('Reuse numerical check:',variant,case,flush=True)
            else:
                folder, _ = render_one(work_root,job,exe,version,case,'sparse_3d_global',spp,
                                       status,progress_root=root)
            folders[(variant,case)] = folder
    checks = []
    sl = np.s_[size//4:3*size//4,size//4:3*size//4]
    for case in rg.CASES:
        base = folders[('base',case)]
        a = rg.pfm(base/'render.pfm')[...,0][sl]
        ah = [rg.pfm(base/f'samples{c}.pfm')[...,0][sl] for c in ('A','B')]
        for variant in ('half_step','larger_domain'):
            refined = folders[(variant,case)]
            b = rg.pfm(refined/'render.pfm')[...,0][sl]
            bh = [rg.pfm(refined/f'samples{c}.pfm')[...,0][sl] for c in ('A','B')]
            delta = float((b-a).mean())
            half_delta = (bh[0]-ah[0])-(bh[1]-ah[1])
            se = float(.5*np.sqrt(np.sum(half_delta**2))/a.size)
            checks.append(dict(case=case,check=variant,baseline=str(base),refined=str(refined),
                region='center_half',size=[size,size],spp=spp,baseline_mean=float(a.mean()),
                refined_mean=float(b.mean()),difference=delta,relative_change=delta/float(a.mean()),
                paired_half_standard_error=se,
                within_approximate_95_percent_sampling_interval=abs(delta)<=1.96*se))
    atomic_json(root/'validation.json',dict(renderer_sha256=version,checks=checks,
        reused_from=str(reuse) if reuse else None,
        baseline_step=config['step'],refined_step=config['step']/2,
        interpretation='Paired even/odd half estimates measure sampling uncertainty. These low-resolution '
                       'sensitivity checks do not prove step-size convergence or bound sparse-convolution '
                       'approximation error. Final sampling criteria are evaluated separately at full resolution.',
        domain_check='Expand x/z extents and lower y by 1.5; upper y is held fixed.'))


def run(args):
    root=rg.OUT/args.stage
    config=dict(size=args.size,threads=args.threads,seed=rg.SEED,step=args.step,
                min_spp=args.min_spp,max_spp=args.max_spp,first_spp=args.first_spp,
                numerical_size=args.numerical_size,numerical_spp=args.numerical_spp,
                display_noise_limit=.01,block_change_limit=.01,mean_ci_limit=.005)
    if args.reuse_numerics_from:
        config['reuse_numerics_from']=str((rg.OUT/args.reuse_numerics_from).resolve())
    with job_lock(root):
        if (root/'config.json').exists() and rg.load(root/'config.json')!=config:
            raise RuntimeError('Existing job has different settings; use its settings or a new stage')
        atomic_json(root/'config.json',config)
        status=dict(state='preparing',detail='准备固定版本渲染器与续跑目录')
        if (root/'status.json').exists():
            previous_status=rg.load(root/'status.json')
            if previous_status.get('latest_level'):
                status['latest_level']=previous_status['latest_level']
        publish_status(root,status)
        try:
            exe,version=snapshot_renderer(root)
            status['state']='running'
            numerical_validation(root,config,exe,version,status)
            status['phase']='正式图：九张图逐档同步增加 spp'
            target,previous=config['first_spp'],None
            while target<=config['max_spp']:
                level=root/'levels'/f'spp{target:06d}'
                for case in rg.CASES:
                    for mode in rg.MODES:
                        saved=level/case/mode/f'seed{config["seed"]}'/'result.json'
                        if saved.exists():
                            validate_result(saved.parent,rg.load(saved))
                            continue
                        folder,result=render_one(root,config,exe,version,case,mode,target,status)
                        snapshot_level(root,config,target,case,mode,folder,result)
                check=convergence(root,config,target,previous)
                report_level(root,config,target,check)
                status.update(latest_level=level.relative_to(root).as_posix(),latest_spp=target,
                              detail=f'九张 {config["size"]}×{config["size"]} / {target} spp 图已完成并核验')
                publish_status(root,status)
                if check['passed']:
                    status['state']='complete'; publish_status(root,status)
                    return
                previous,target=target,2*target
            status['state']='limit_reached';publish_status(root,status)
        except BaseException as error:
            status.update(state='failed',detail=str(error));publish_status(root,status)
            raise


def check_resume():
    version=rg.sha(rg.ROOT/'build/Release/tungsten.exe')
    root=rg.OUT/('quality_resume_check_'+version[:12])
    config=dict(size=32,threads=8,seed=rg.SEED,step=.004)
    results=[]
    with job_lock(root):
        atomic_json(root/'config.json',config)
        exe,version=snapshot_renderer(root)
        for mode in rg.MODES:
            status=dict(state='running')
            split=root/'split';direct=root/'direct'
            atomic_json(split/'config.json',config);atomic_json(direct/'config.json',config)
            existing=split/'work'/'u_groove_side'/mode/f'seed{config["seed"]}'/'result.json'
            if not existing.exists() or rg.load(existing)['spp']<32:
                render_one(split,config,exe,version,'u_groove_side',mode,16,status)
            a,_=render_one(split,config,exe,version,'u_groove_side',mode,32,status)
            b,_=render_one(direct,config,exe,version,'u_groove_side',mode,32,status)
            for name in ('render.pfm','samplesA.pfm','samplesB.pfm','samplesVariance.pfm'):
                if not np.array_equal(rg.pfm(a/name),rg.pfm(b/name)):
                    raise RuntimeError('Native resume differs from uninterrupted render: '+mode+' '+name)
            results.append(dict(mode=mode,passed=True,split=str(a),uninterrupted=str(b)))
            print('Resume equality passed:',mode,flush=True)
        atomic_json(root/'validation.json',dict(passed=True,renderer_sha256=version,checks=results))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','check-resume'))
    parser.add_argument('--size',type=int,default=512)
    parser.add_argument('--stage')
    parser.add_argument('--threads',type=int,default=32)
    parser.add_argument('--step',type=float,default=.002)
    parser.add_argument('--first-spp',type=int,default=64)
    parser.add_argument('--min-spp',type=int,default=1024)
    parser.add_argument('--max-spp',type=int,default=8192)
    parser.add_argument('--numerical-size',type=int,default=64)
    parser.add_argument('--numerical-spp',type=int,default=64)
    parser.add_argument('--reuse-numerics-from',help='Reuse verified numerical checks from an existing stage')
    args=parser.parse_args()
    if args.size<32 or args.size%32 or args.threads<1 or not math.isfinite(args.step) or args.step<=0:
        parser.error('size must be a multiple of 32; threads and step must be positive')
    if any(v<2 or v&(v-1) for v in (args.first_spp,args.min_spp,args.max_spp)) or not args.first_spp<=args.min_spp<=args.max_spp:
        parser.error('spp values must be increasing powers of two')
    if args.numerical_size<32 or args.numerical_size%32 or args.numerical_spp<2 or args.numerical_spp%2:
        parser.error('numerical size must be a multiple of 32 and numerical spp must be positive and even')
    args.stage=args.stage or f'high_quality_{args.size}'
    stage_path=(rg.OUT/args.stage).resolve()
    if not stage_path.is_relative_to(rg.OUT.resolve()) or stage_path==rg.OUT.resolve():
        parser.error('stage must be a subdirectory of outputs/rough_geometry_compare')
    if args.reuse_numerics_from:
        source_path=(rg.OUT/args.reuse_numerics_from).resolve()
        if (not source_path.is_relative_to(rg.OUT.resolve()) or source_path==stage_path
                or not (source_path/'validation.json').is_file()):
            parser.error('reuse source must be a different stage with completed numerical checks')
    if args.command=='check-resume':check_resume()
    else:run(args)
