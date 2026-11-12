"""Measure time to a fixed error against the requested archived GPIS global.

python -B tools/surface_compare/gpis_shaderball_equal_quality.py smoke
python -B tools/surface_compare/gpis_shaderball_equal_quality.py run
python -B tools/surface_compare/gpis_shaderball_equal_quality.py run --cases all

Both methods use one renderer, 32 threads, and independent seeds. Progressive
renders resume the actual sample accumulators; timings are measured, not
extrapolated from the old machine. Historical renders are never overwritten.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEPS = ROOT/'build/fp-gradient-validation-deps'
if DEPS.exists():
    sys.path.insert(0, str(DEPS))
import numpy as np

SOURCE = ROOT/'outputs/gpis_shaderball_parameter_comparison_20261111'
DEFAULT_OUT = ROOT/'outputs/gpis_shaderball_equal_quality_20261112'
EXE = ROOT/'build/Release/tungsten.exe'
MODES = ('global_conditional', 'gpis_renewal_plus')
DEFAULT_CASE = 'ell0p2_sigma0p02'
TARGETS = (.20, .15, .125, .11, .10)
BOTTOM = (44, 164, 213, 216)


class TimingInterruption(RuntimeError):
    """A stalled watchdog indicates that process wall time was compromised."""


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pfm(path):
    with Path(path).open('rb') as stream:
        assert stream.readline().strip() == b'PF'
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        pixels = np.fromfile(stream, dtype='<f4' if scale<0 else '>f4')
    assert pixels.size == width*height*3
    pixels = np.flipud(pixels.reshape(height, width, 3)).astype(np.float64)*abs(scale)
    assert np.isfinite(pixels).all() and (pixels>=0).all()
    return pixels


def regions(pixels):
    x0,y0,x1,y1 = BOTTOM
    return dict(full=pixels, bottom=pixels[y0:y1,x0:x1])


def errors(pixels, reference):
    result = {}
    for region, actual in regions(pixels).items():
        expected = regions(reference)[region]
        mse = float(np.mean((actual-expected)**2))
        reference_ms = float(np.mean(expected**2))
        assert reference_ms>0
        result[region] = dict(mse=mse, rmse=mse**.5,
            relative_rmse=(mse/reference_ms)**.5, reference_ms=reference_ms,
            mean_rgb=actual.mean(axis=(0,1)).tolist())
    return result


def source_dir(case, mode):
    return SOURCE/'renders'/case/mode/'seed20261024'


def verify_source(case, mode):
    folder = source_dir(case, mode)
    record = load(folder/'result.json')
    for name, digest in record['sha256'].items():
        assert sha(folder/name)==digest, str(folder/name)
    scene = load(folder/'scene.json')
    assert scene==record['identity']['scene']
    assert record['resolution']==[256,256] and record['spp']==128
    assert record['seed']==20261024
    pixels = pfm(folder/'render.pfm')
    assert pixels.shape==(256,256,3)
    return scene, record, pixels


def points(max_spp):
    assert max_spp>=2 and max_spp&(max_spp-1)==0
    return [2**n for n in range(max_spp.bit_length())]


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    cases = sorted(path.name for path in (SOURCE/'renders').iterdir() if path.is_dir())
    settings = dict(renderer=str(EXE), renderer_sha256=sha(EXE), threads=args.threads,
        seeds=args.seeds, resolution=[256,256], sample_points=points(args.max_spp),
        thresholds=list(TARGETS), modes=list(MODES), bottom_bounds=list(BOTTOM),
        reference_seed=20261024, reference_spp=128,
        metric='sqrt(mean((RGB-test - RGB-reference)^2) / mean(RGB-reference^2)); linear HDR RGB',
        timing='Sum of measured renderer process durations; includes load/resume/save; excludes metric/report work.',
        secondary_timing='Sum of renderer-reported render-loop durations; excludes process setup and final save.',
        sampling='Same spp_step=1 for both methods; adaptive sampling disabled; true saved-state continuation.',
        reference_note='Fixed noisy 128-spp image, not a noise-free ground truth. No exposure fit or noise subtraction.',
        reference_sources={}, scenes={})
    for case in cases:
        reference_scene, record, reference = verify_source(case, 'gpis_global')
        a=pfm(source_dir(case,'gpis_global')/'samplesA.pfm')
        b=pfm(source_dir(case,'gpis_global')/'samplesB.pfm')
        assert np.allclose((a+b)/2, reference, rtol=3e-6, atol=3e-7)
        noise = .5*(a-b)
        noise_stats={key:float(np.sqrt(np.mean(value**2)/np.mean(regions(reference)[key]**2)))
                     for key,value in regions(noise).items()}
        info = dict(folder=str(source_dir(case,'gpis_global').relative_to(ROOT)),
            scene_sha256=sha(source_dir(case,'gpis_global')/'scene.json'),
            pfm_sha256=sha(source_dir(case,'gpis_global')/'render.pfm'),
            png_sha256=sha(source_dir(case,'gpis_global')/'render.png'),
            result_sha256=sha(source_dir(case,'gpis_global')/'result.json'),
            renderer_sha256=record['identity']['renderer_sha256'],
            covariance=reference_scene['media'][0]['gaussian_process']['covariance'],
            estimated_reference_noise_relative_rms=noise_stats, archived_128_spp={})
        settings['scenes'][case]={}
        for mode in MODES:
            scene, mode_record, image = verify_source(case, mode)
            for section in ('camera','integrator','primitives','bsdfs'):
                assert scene[section]==reference_scene[section], (case,mode,section)
            assert scene['media'][0]['gaussian_process']==reference_scene['media'][0]['gaussian_process']
            medium=scene['media'][0]
            if mode=='gpis_renewal_plus':
                assert medium['type']=='sparse_conv_noise' and medium['correlation_context']=='renewal+'
                assert medium['1D_sampling'] and medium['1D_sampling_scheme']=='mis'
                assert medium['1D_gradient_correlationXY'] and medium['step_size']==.0025
                material=medium['phase_function']['bsdf']
            else:
                assert medium['type']==mode and not medium['fp_gradient']['enabled']
                assert medium['non_gp_surface_policy']=='retain_endpoint'
                material=medium['material']
            assert material==reference_scene['media'][0]['phase_function']['bsdf']
            info['archived_128_spp'][mode]=dict(errors=errors(image,reference),
                seconds_on_old_machine=mode_record['seconds'],
                note='Historical context only. Same seed as reference; excluded from new timing and independent-seed curves.')
            settings['scenes'][case][mode]=sha(source_dir(case,mode)/'scene.json')
        settings['reference_sources'][case]=info
    path=args.output/'manifest.json'
    if path.exists():
        assert load(path)==settings, 'Benchmark settings changed; select a new --output'
    else:
        save(path,settings)
    return settings


def make_scene(case, mode, work, spp, size=256, resume=True):
    scene=copy.deepcopy(load(source_dir(case,mode)/'scene.json'))
    scene['camera']['resolution']=[size,size]
    scene['renderer'].update(output_directory=str(work), output_file='render.png',
        hdr_output_file='render.pfm', resume_render_file=str(work/'resume.dat'),
        enable_resume_render=resume, checkpoint_interval='0', spp=spp, spp_step=1,
        timeout='0', adaptive_sampling=False, output_buffers=[])
    return scene


def render_segment(case, mode, seed, spp, work, threads, previous_spp, resume=True):
    work.mkdir(parents=True,exist_ok=True)
    scene=make_scene(case,mode,work,spp,resume=resume)
    save(work/'scene.json',scene)
    command=[str(EXE),'-t',str(threads),'-s',str(seed),'-d',str(work),str(work/'scene.json')]
    log_path=work/f'segment_{spp:04d}.log'
    start=time.perf_counter()
    with log_path.open('wb') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        while process.poll() is None:
            before_wait=time.perf_counter()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            wait_seconds=time.perf_counter()-before_wait
            if wait_seconds>45:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                log.flush()
                excluded=work/'excluded_timing'/f'spp{spp:04d}_{time.time_ns()}'
                excluded.mkdir(parents=True,exist_ok=True)
                for name in ('scene.json',log_path.name):
                    shutil.copy2(work/name,excluded/name)
                save(excluded/'reason.json',dict(reason='30-second watchdog resumed after >45 seconds; '
                    'timing contaminated by environment interruption. Retry from prior saved checkpoint.',
                    watchdog_seconds=wait_seconds, elapsed_process_seconds=time.perf_counter()-start,
                    previous_spp=previous_spp,target_spp=spp,seed=seed,mode=mode))
                raise TimingInterruption(str(excluded))
            if process.poll() is None:
                progress=[s for s in log_path.read_text(encoding='utf-8',errors='replace').splitlines() if s.startswith('Completed ')]
                print('RUNNING',case,mode,'seed',seed,'target',spp,
                    f'{time.perf_counter()-start:.0f}s',progress[-1] if progress else '',flush=True)
    seconds=time.perf_counter()-start
    log=log_path.read_text(encoding='utf-8',errors='replace')
    assert process.returncode==0 and 'Finished render.' in log and f'Completed {spp}/{spp} spp' in log, str(log_path)
    assert not re.search(r'Caught an internal error|unrecoverable|Unable to load|sanity check failed|Conditioning .* error',log,re.I), str(log_path)
    if previous_spp:
        assert 'Resume successful' in log, 'Continuation failed; refusing to count a restarted render'
        completed=[int(value) for value in re.findall(r'Completed (\d+)/',log)]
        assert completed==list(range(previous_spp+1,spp+1)), completed
    else:
        assert 'Resume successful' not in log
    elapsed=re.findall(r'Completed \d+/\d+ spp, ([\d.]+)/',log)
    assert elapsed
    warnings=[line for line in log.splitlines() if re.search(r'warning|Wrong normal direction|Failed to sample gradient|Gradient invalid|Gradient zero|farT == 0',line,re.I)]
    return dict(process_seconds=seconds, render_loop_seconds=float(elapsed[-1]),
        previous_spp=previous_spp, spp=spp, command=command,
        scene_sha256=sha(work/'scene.json'), renderer_sha256=sha(EXE),
        log_sha256=sha(log_path), warnings=warnings), pfm(work/'render.pfm')


def smoke(args):
    """Require bit-identical uninterrupted and resumed small renders."""
    result=[]
    for mode in MODES:
        home=args.output/'validation'/mode
        for variant in ('resumed','continuous'):
            work=home/variant
            assert not (work/'resume.dat').exists(), 'Use a fresh smoke output directory'
            work.mkdir(parents=True,exist_ok=True)
            targets=(1,2) if variant=='resumed' else (2,)
            previous=0
            for target in targets:
                # Keep smoke geometry and camera identical while reducing resolution.
                s=make_scene(DEFAULT_CASE,mode,work,target,size=32,resume=True)
                save(work/'scene.json',s)
                command=[str(EXE),'-t',str(args.threads),'-s','41','-d',str(work),str(work/'scene.json')]
                with (work/f'segment_{target}.log').open('wb') as log:
                    p=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                text=(work/f'segment_{target}.log').read_text(encoding='utf-8',errors='replace')
                assert p.returncode==0 and f'Completed {target}/{target} spp' in text
                if previous: assert 'Resume successful' in text
                previous=target
        a,b=(pfm(home/v/'render.pfm') for v in ('resumed','continuous'))
        row=dict(mode=mode,equal=bool(np.array_equal(a,b)),maximum_difference=float(np.max(np.abs(a-b))),
            renderer_sha256=sha(EXE),resolution=[32,32],seed=41,spp=2,threads=args.threads)
        result.append(row)
        save(args.output/'validation/resume.json',result)
        assert row['equal'], row
        print('RESUME VALIDATED',mode,flush=True)


def run(args):
    manifest=prepare(args)
    validation=load(args.output/'validation/resume.json')
    assert len(validation)==2 and all(r['equal'] and r['renderer_sha256']==manifest['renderer_sha256'] for r in validation)
    selected=sorted(manifest['reference_sources']) if args.cases==['all'] else args.cases
    assert selected and set(selected)<=set(manifest['reference_sources'])
    save(args.output/'plan.json',dict(cases=selected,seeds=args.seeds,points=manifest['sample_points'],modes=list(MODES)))
    for case in selected:
        reference=pfm(source_dir(case,'gpis_global')/'render.pfm')
        for seed in args.seeds:
            for mode in MODES:
                parent=args.output/'renders'/case/mode/f'seed{seed}'
                work=parent/'work'
                cumulative_wall,cumulative_core,previous=0.,0.,0
                for spp in manifest['sample_points']:
                    dest=parent/f'spp{spp:04d}'
                    if (dest/'result.json').exists():
                        row=load(dest/'result.json')
                        assert row['renderer_sha256']==manifest['renderer_sha256']
                        for name,digest in row['sha256'].items(): assert sha(dest/name)==digest
                        cumulative_wall=row['cumulative_process_seconds']; cumulative_core=row['cumulative_render_seconds']; previous=spp
                        continue
                    assert sha(EXE)==manifest['renderer_sha256']
                    print('START',case,mode,'seed',seed,'spp',spp,'from',previous,flush=True)
                    for attempt in range(4):
                        if previous:
                            prior=parent/f'spp{previous:04d}'
                            work.mkdir(parents=True,exist_ok=True)
                            shutil.copy2(prior/'resume.dat',work/'resume.dat')
                        else:
                            (work/'resume.dat').unlink(missing_ok=True)
                        try:
                            segment,image=render_segment(case,mode,seed,spp,work,args.threads,previous)
                            break
                        except TimingInterruption as error:
                            print('EXCLUDED TIMING; RETRY',attempt+1,str(error),flush=True)
                            if attempt==3:
                                raise
                    assert image.shape==(256,256,3)
                    cumulative_wall+=segment['process_seconds']; cumulative_core+=segment['render_loop_seconds']
                    dest.mkdir(parents=True,exist_ok=True)
                    files=('scene.json','render.png','render.pfm','resume.dat')
                    for name in files: shutil.copy2(work/name,dest/name)
                    shutil.copy2(work/f'segment_{spp:04d}.log',dest/'render.log')
                    row=dict(segment,case=case,mode=mode,seed=seed,threads=args.threads,
                        cumulative_process_seconds=cumulative_wall,cumulative_render_seconds=cumulative_core,
                        errors=errors(image,reference),reference_pfm_sha256=manifest['reference_sources'][case]['pfm_sha256'],
                        sha256={name:sha(dest/name) for name in (*files,'render.log')})
                    save(dest/'result.json',row)
                    previous=spp
                    print('DONE',case,mode,seed,spp,f'{cumulative_wall:.2f}s',
                        'NRMSE',f'{row["errors"]["full"]["relative_rmse"]:.6f}',flush=True)
                report=Path(__file__).with_name('gpis_shaderball_equal_quality_report.py')
                if report.exists():
                    subprocess.run([sys.executable,'-B',str(report),str(args.output)],cwd=ROOT,check=True)
    print('BENCHMARK COMPLETE',args.output,flush=True)


def audit(args):
    """Recompute metrics from every HDR snapshot and validate the timing chain."""
    manifest=prepare(args)
    plan=load(args.output/'plan.json')
    report=load(args.output/'results.json')
    assert report['complete'] and report['plan']==plan
    checked=0
    diagnostic_counts={mode:0 for mode in MODES}
    for case in plan['cases']:
        reference=pfm(source_dir(case,'gpis_global')/'render.pfm')
        for mode in plan['modes']:
            fingerprints={spp:[] for spp in plan['points']}
            raw={}
            for seed in plan['seeds']:
                wall,core,previous=0.,0.,0
                for spp in plan['points']:
                    folder=args.output/'renders'/case/mode/f'seed{seed}'/f'spp{spp:04d}'
                    row=load(folder/'result.json')
                    assert row['previous_spp']==previous
                    assert row['renderer_sha256']==manifest['renderer_sha256']
                    assert row['threads']==args.threads and row['seed']==seed
                    for name,digest in row['sha256'].items():
                        assert sha(folder/name)==digest, str(folder/name)
                    assert row['log_sha256']==row['sha256']['render.log']
                    assert row['scene_sha256']==row['sha256']['scene.json']
                    assert load(folder/'scene.json')==make_scene(case,mode,folder.parent/'work',spp)
                    pixels=pfm(folder/'render.pfm')
                    assert pixels.shape==(256,256,3)
                    actual=errors(pixels,reference)
                    for region in ('full','bottom'):
                        for key in ('mse','rmse','relative_rmse','reference_ms','mean_rgb'):
                            assert np.allclose(actual[region][key],row['errors'][region][key],rtol=1e-12,atol=1e-14)
                    assert row['process_seconds']>0 and row['render_loop_seconds']>0
                    wall+=row['process_seconds']; core+=row['render_loop_seconds']
                    assert abs(wall-row['cumulative_process_seconds'])<1e-9
                    assert abs(core-row['cumulative_render_seconds'])<1e-9
                    log=(folder/'render.log').read_text(encoding='utf-8',errors='replace')
                    assert [int(s) for s in re.findall(r'Completed (\d+)/',log)]==list(range(previous+1,spp+1))
                    assert ('Resume successful' in log)==bool(previous)
                    assert 'Finished render.' in log and not re.search(r'Caught an internal error|unrecoverable|sanity check failed',log,re.I)
                    diagnostic_counts[mode]+=log.count('farT == 0')
                    fingerprints[spp].append(row['sha256']['render.pfm'])
                    raw[seed,spp]=row
                    checked+=1; previous=spp
            assert all(len(set(values))==len(plan['seeds']) for values in fingerprints.values()), 'Seeds must generate distinct images'
            for region in ('full','bottom'):
                aggregate=report['cases'][case]['modes'][mode]['aggregate'][region]
                assert [point['spp'] for point in aggregate]==plan['points']
                for point in aggregate:
                    rows=[raw[seed,point['spp']] for seed in plan['seeds']]
                    error=np.sqrt(np.mean([r['errors'][region]['mse'] for r in rows])/rows[0]['errors'][region]['reference_ms'])
                    assert abs(point['relative_rmse']-error)<1e-12
                    assert abs(point['mean_process_seconds']-np.mean([r['cumulative_process_seconds'] for r in rows]))<1e-9
                    assert abs(point['mean_render_seconds']-np.mean([r['cumulative_render_seconds'] for r in rows]))<1e-9
                for hit in report['cases'][case]['modes'][mode]['crossings'][region]:
                    first=next((p for p in aggregate if p['relative_rmse']<=hit['threshold']),None)
                    assert hit['first_hit']==first
    expected=len(plan['cases'])*len(plan['modes'])*len(plan['seeds'])*len(plan['points'])
    assert checked==expected==report['observed_records']
    validation=load(args.output/'validation/resume.json')
    assert len(validation)==2 and all(v['equal'] and v['maximum_difference']==0 and
        v['renderer_sha256']==manifest['renderer_sha256'] for v in validation)
    save(args.output/'audit.json',dict(passed=True,checked_records=checked,renderer_sha256=manifest['renderer_sha256'],
        checks=['archived input hashes','all snapshot hashes','same renderer, threads, resolution and physical scene',
                'finite nonnegative HDR pixels','independent seed images differ','errors recomputed from PFM',
                'cumulative process and render-loop times recomputed','resume sample chain',
                'three-seed MSE aggregation','first observed threshold crossing','bit-exact smoke continuation'],
        path_diagnostic_occurrences=diagnostic_counts,
        excluded_timing_attempts=report['excluded_timing_attempts'],
        scripts_sha256={name:sha(Path(__file__).with_name(name)) for name in
            ('gpis_shaderball_equal_quality.py','gpis_shaderball_equal_quality_report.py')}))
    print('AUDIT PASSED',checked,'snapshots',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','smoke','run','audit'))
    parser.add_argument('--cases',nargs='+',default=[DEFAULT_CASE])
    parser.add_argument('--seeds',nargs='+',type=int,default=[41,137,313])
    parser.add_argument('--max-spp',type=int,default=64)
    parser.add_argument('--threads',type=int,default=32)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    args.output=args.output.resolve()
    assert args.threads>0 and len(args.seeds)==len(set(args.seeds)) and 20261024 not in args.seeds
    kernel=None
    if os.name=='nt' and args.action in ('smoke','run'):
        import ctypes
        kernel=ctypes.windll.kernel32
        # Temporary per-thread request; does not change the user's power plan.
        assert kernel.SetThreadExecutionState(0x80000001), 'Cannot request awake execution'
    try:
        if args.action=='smoke': smoke(args)
        elif args.action=='prepare': prepare(args)
        elif args.action=='audit': audit(args)
        else: run(args)
    finally:
        if kernel:
            kernel.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    main()
