"""Matched four-model plane experiment with full Renewal+.

python -B tools/surface_compare/plane_strong_light.py batch
python -B tools/surface_compare/plane_strong_light.py batch --emission 60 --spp 512
python -B tools/surface_compare/plane_strong_light.py pilot --light-type point --emission 60 --spp 512
<plot-python> -B tools/surface_compare/plane_strong_light.py preview --light-type point --emission 60 --spp 512 --report
<plot-python> -B tools/surface_compare/plane_strong_light.py render --light-type point --emission 60 --spp 512 --report
<plot-python> -B tools/surface_compare/plane_strong_light_figures.py

Batch validates all six cases/four modes at 32x32, 8 spp, then records
all scattering orders on the same paths at the original 256x256, 128 spp.
Point lights retain the original light center and match total emitted power:
I_point = L_area * (.45 * .45) / 4. The preview command renders 256x256 at 32 spp.
"""
from __future__ import annotations

import argparse
import copy
import math
import subprocess
import time
from pathlib import Path

import numpy as np
from plane import OUT, ROOT, CASES, SEED, folder
from plane_renewal_mis import measure, ERRORS
from run import load, save, sha, pfm

STAGE = 'strong_light_full_renewal_mis'
RENEWAL = 'sparse_1d_renewal_full_mis'
MODES = ('classic_local', 'classic_global', 'global_conditional', RENEWAL)
CHANGES = {'correlation_context': 'renewal+', '1D_sampling': True,
           '1D_sampling_scheme': 'mis', '1D_gradient_correlationXY': True}
EMISSION = 300
PILOT = STAGE+'_validated_pilot'
AOV_EXE = ROOT/'build/Release/surface_bounce_images.exe'
LAYERS = ('order00', 'order01', 'order02', 'order03', 'order04', 'order05plus', 'multiple', 'beauty')


def renderer():
    exe = ROOT/'build/Release/tungsten.exe'
    return exe, sha(exe)


def reference(case, mode):
    return folder('final', case, 'sparse_3d_global' if mode == RENEWAL else mode, 'total', SEED)


def destination(stage, case, mode, component):
    return folder(stage, case, mode, component, SEED)


def light_parameters(emission, light_type='area'):
    if light_type == 'area':
        return dict(type='area', radiance=emission)
    assert light_type == 'point'
    area = .45 * .45
    return dict(type='point', normalization='equal_total_power',
        reference_area_radiance=emission, reference_area=area,
        intensity=emission*area/4, total_power=math.pi*area*emission,
        formula='4*pi*I_point = pi*A*L_area',
        direct_estimator='NEE with weight 1 for the delta light')


def scene(stage, case, mode, component, size, spp, emission=EMISSION, light_type='area'):
    source = reference(case, mode)
    original = load(source/'scene.json')
    assert sha(source/'scene.json') == load(source/'result.json')['scene_sha256']
    s = copy.deepcopy(original)
    if mode == RENEWAL:
        s['media'][0].update(CHANGES)
    assert s['primitives'][1]['name'] == 'key'
    assert s['primitives'][1]['emission'] == 30
    s['primitives'][1]['emission'] = emission
    if light_type == 'point':
        light = s['primitives'][1]
        s['primitives'][1] = dict(name=light['name'], type='point',
            emission=light_parameters(emission, light_type)['intensity'],
            transform=dict(position=light['transform']['position']))
    else:
        assert light_type == 'area'
    s['camera']['resolution'] = [size, size]
    s['integrator'].update(min_bounces=2 if component == 'multiple_check' else 1,
                           max_bounces=2 if component == 'single' else 128)
    s['renderer'].update(spp=spp, spp_step=min(16, spp),
        output_directory=str(destination(stage, case, mode, component)))
    # Reject changes to any other physical or camera parameter.
    check = copy.deepcopy(s)
    if mode == RENEWAL:
        for key in CHANGES:
            check['media'][0][key] = original['media'][0][key]
    check['primitives'][1] = original['primitives'][1]
    check['camera']['resolution'] = original['camera']['resolution']
    check['integrator'] = original['integrator']
    check['renderer'] = original['renderer']
    assert check == original
    assert s['integrator']['enable_volume_light_sampling']
    return s


def validated_result(d, expected_scene, renderer_hash):
    row = load(d/'result.json')
    assert load(d/'scene.json') == expected_scene, str(d)
    assert row['provenance']['renderer_sha256'] == renderer_hash
    assert row['completed_spp'] == expected_scene['renderer']['spp']
    assert row['size'] == expected_scene['camera']['resolution']
    for name, key in [('scene.json', 'scene_sha256'), ('render.pfm', 'pfm_sha256'),
                      ('render.png', 'png_sha256'), ('render.log', 'log_sha256')]:
        assert sha(d/name) == row[key], str(d/name)
    for name, digest in row['auxiliary_sha256'].items():
        assert sha(d/name) == digest, str(d/name)
    measure(d)
    return row


def render(stage, size, spp, components, threads, emission=EMISSION, light_type='area'):
    exe, digest = renderer()
    manifest = dict(stage=stage, size=size, spp=spp, seed=SEED, emission=emission,
        modes=list(MODES), changes=CHANGES, executable=str(exe), renderer_sha256=digest,
        runner_sha256=sha(Path(__file__)), threads=threads,
        light=light_parameters(emission, light_type), sources=[])
    for case in CASES:
        for mode in MODES:
            source = reference(case, mode)
            provenance = dict(load(source/'result.json')['provenance'])
            provenance['renderer_sha256'] = digest
            if mode == 'classic_local':
                field = Path(load(source/'scene.json')['media'][0]['field']['file'])
                assert sha(field) == provenance['nvdb_sha256']
            for component in components:
                s = scene(stage, case, mode, component, size, spp, emission, light_type)
                d = destination(stage, case, mode, component)
                if (d/'result.json').exists():
                    row = validated_result(d, s, digest)
                    print('Reuse', case, mode, component, flush=True)
                else:
                    if (d/'render.pfm').exists():
                        raise RuntimeError('Existing unvalidated render: '+str(d))
                    save(d/'scene.json', s)
                    print('Render', stage, case, mode, component, flush=True)
                    start = time.monotonic()
                    with (d/'render.log').open('w', encoding='utf-8') as log:
                        p = subprocess.run([str(exe), '-t', str(threads), '-s', str(SEED),
                            '-d', str(d), str(d/'scene.json')], cwd=ROOT, stdout=log,
                            stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
                    log = (d/'render.log').read_text(encoding='utf-8', errors='replace')
                    if (p.returncode or 'Finished render.' not in log or ERRORS.search(log)
                            or f'Completed {spp}/{spp} spp' not in log):
                        raise RuntimeError('Render failed; inspect '+str(d/'render.log'))
                    assert pfm(d/'render.pfm').shape == (size, size, 3)
                    row = dict(seconds=time.monotonic()-start, provenance=provenance,
                        completed_spp=spp, size=[size, size], statistics=measure(d),
                        scene_sha256=sha(d/'scene.json'), pfm_sha256=sha(d/'render.pfm'),
                        png_sha256=sha(d/'render.png'), log_sha256=sha(d/'render.log'),
                        auxiliary_sha256={f'samples{c}.pfm': sha(d/f'samples{c}.pfm')
                                          for c in ('A', 'B', 'Variance')})
                    save(d/'result.json', row)
                    print('Done', case, mode, component, round(row['seconds'], 2),
                          'mean', row['statistics']['mean'], flush=True)
                manifest['sources'].append(dict(case=case, mode=mode, component=component,
                    folder=str(d), result=row, original_scene_sha256=sha(source/'scene.json')))
                save(OUT/stage/'run.json', manifest)
    return manifest


def check_decomposition(stage, emission=EMISSION, light_type='area'):
    _, digest = renderer()
    checks = []
    for case in CASES:
        for mode in MODES:
            paths = [destination(stage, case, mode, c) for c in ('total', 'single', 'multiple_check')]
            results = []
            for d in paths:
                row = load(d/'result.json')
                results.append(validated_result(d, scene(stage, case, mode, d.name,
                    row['size'][0], row['completed_spp'], emission, light_type), digest))
            assert len({(tuple(r['size']), r['completed_spp']) for r in results}) == 1
            a, b, c = [pfm(d/'render.pfm') for d in paths]
            tolerance = 2e-5*max(1., float(a.max()))
            errors = {'full': float(np.abs(a-b-c).max())}
            for letter in 'AB':
                x, y, z = [pfm(d/f'samples{letter}.pfm') for d in paths]
                errors[letter] = float(np.abs(x-y-z).max())
            assert max(errors.values()) <= tolerance, (case, mode, errors)
            assert float((a-b).min()) >= -tolerance
            checks.append(dict(case=case, mode=mode, errors=errors, tolerance=tolerance,
                size=results[0]['size'], spp=results[0]['completed_spp'],
                sources=[dict(folder=str(d), result=r) for d, r in zip(paths, results)]))
    save(OUT/stage/'decomposition_checks.json', dict(passed=True, checks=checks,
        definition='Paired total = single + independently depth-filtered multiple; full image and both sample halves.'))
    print('Validated', len(checks), 'case/model decompositions', flush=True)


def aov_scene(stage, case, mode, size, spp, emission=EMISSION, light_type='area'):
    s = scene(stage, case, mode, 'bounces', size, spp, emission, light_type)
    s['renderer']['output_buffers'] = []
    return s


def validate_aov(d):
    result = load(d/'result.json')
    arrays = {key: pfm(d/(key+'.pfm')) for key in LAYERS}
    assert result['observer_mismatches'] == 0 and result['verified_samples'] > 0
    assert np.allclose(sum(arrays[k] for k in LAYERS[:6]), arrays['beauty'], rtol=2e-6, atol=2e-7)
    assert np.allclose(arrays['order01']+arrays['multiple'], arrays['beauty'], rtol=2e-6, atol=2e-7)
    assert np.max(arrays['order00']) == 0
    assert np.allclose(sum(arrays[k] for k in LAYERS[2:6]), arrays['multiple'], rtol=2e-6, atol=2e-7)
    assert pfm(d/'termination_fractions.pfm')[:, :, 2].max() == 0, 'Clamped paths found'
    if (d/'finite_clamp_restoration.json').exists():
        restored=load(d/'finite_clamp_restoration.json')
        assert restored['passed']
        assert sha(d/'finite_clamp_replay/replay.json')==restored['replay_sha256']
        for name,digest in restored['original_sha256'].items():
            assert sha(d/'finite_clamp_replay/original'/name)==digest
        for name,digest in restored['output_sha256'].items():
            assert sha(d/name)==digest
    return result, arrays


def render_aov(stage, size, spp, threads, compare_native=False, emission=EMISSION, pilot_stage=PILOT,
               light_type='area'):
    exe, native_hash = renderer()
    manifest = dict(stage=stage, size=size, spp=spp, seed=SEED, emission=emission, pilot_stage=pilot_stage,
        modes=list(MODES), changes=CHANGES, light=light_parameters(emission, light_type), executable=str(AOV_EXE),
        renderer_sha256=native_hash, aov_sha256=sha(AOV_EXE),
        core_sha256=sha(ROOT/'build/Release/core.lib'), runner_sha256=sha(Path(__file__)),
        correction='NEE zero-length PDF matches the start-point sampler; short-segment covariance avoids cancellation. All four modes use the same updated core.',
        source_sha256={p:sha(ROOT/p) for p in ('src/core/math/SparseConvolutionNoise.cpp',
            'tools/surface_compare/bounce_images.cpp','src/core/integrators/TraceBase.cpp',
            'src/core/integrators/path_tracer/PathTracer.cpp')}, sources=[])
    for case in CASES:
        for mode in MODES:
            d = destination(stage, case, mode, 'bounces')
            s = aov_scene(stage, case, mode, size, spp, emission, light_type)
            spec = dict(scene=str(d/'scene.json'), seed=SEED, samples=spp, threads=threads,
                verify_pixels=[[0,0],[size//2,size//2],[size-1,size-1]], verify_samples=min(spp,16))
            if (d/'verified.json').exists():
                row = load(d/'verified.json')
                assert load(d/'scene.json') == s and load(d/'spec.json') == spec
                assert row['aov_sha256'] == manifest['aov_sha256']
                for name, digest in row['files'].items():
                    assert sha(d/name) == digest
                print('Reuse AOV', case, mode, flush=True)
            else:
                if (d/'beauty.pfm').exists():
                    # Recover a completed process that stopped at output validation.
                    # Unknown or abnormal exits still require a separate retry.
                    assert load(d/'scene.json')==s and load(d/'spec.json')==spec
                    assert load(d/'process.json')['returncode']==0
                    previous=load(OUT/stage/'aov_run.json')
                    assert all(previous[k]==manifest[k] for k in ('aov_sha256','renderer_sha256','core_sha256'))
                    print('Validate completed AOV',case,mode,flush=True)
                else:
                    save(d/'scene.json',s); save(d/'spec.json',spec)
                    print('Render AOV', stage, case, mode, size, spp, flush=True)
                    with (d/'render.log').open('w',encoding='utf-8') as log:
                        p = subprocess.run([str(AOV_EXE),str(d/'spec.json'),str(d)],cwd=ROOT,
                            stdout=log,stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
                    save(d/'process.json',dict(returncode=p.returncode))
                log = (d/'render.log').read_text(encoding='utf-8',errors='replace')
                returncode=load(d/'process.json')['returncode']
                assert returncode==0 and 'Finished bounce images' in log and not ERRORS.search(log), (str(d),returncode)
                if light_type=='point' and pfm(d/'termination_fractions.pfm')[:,:,2].max()>0:
                    from finite_clamp_replay import restore_finite_clamps
                    restore_finite_clamps(d)
                result, arrays = validate_aov(d)
                assert result['resolution'] == [size,size] and result['samples'] == spp
                row = dict(aov_sha256=manifest['aov_sha256'], result=result,
                    files={p.name:sha(p) for p in d.iterdir() if p.suffix in ('.pfm','.json','.log')},
                    max_reconstruction_error=float(np.abs(arrays['order01']+arrays['multiple']-arrays['beauty']).max()))
                if (d/'finite_clamp_restoration.json').exists():
                    row['finite_clamp_restoration']=load(d/'finite_clamp_restoration.json')
                if compare_native:
                    native = destination(stage,case,mode,'total')
                    delta = np.abs(arrays['beauty']-pfm(native/'render.pfm'))
                    assert np.allclose(arrays['beauty'],pfm(native/'render.pfm'),rtol=3e-5,atol=2e-7), str(d)
                    row['native_max_absolute_error'] = float(delta.max())
                save(d/'verified.json',row)
                print('Done AOV',case,mode,round(result['seconds'],2),
                    'mean',float(arrays['beauty'].mean()),flush=True)
            manifest['sources'].append(dict(case=case,mode=mode,folder=str(d),verification=row))
            save(OUT/stage/'aov_run.json',manifest)
    return manifest


def numerical_regressions(stage=STAGE, emission=EMISSION):
    exe = ROOT/'build/Release/surface_path_trace.exe'
    checks = []
    for label, case, size, pixel, sample in (
            ('zero_length', 'same_80', 32, [25,8], 7),
            ('short_segment', 'same_30', 256, [50,43], 48)):
        d = OUT/stage/'numerical_regressions'/label
        s = aov_scene(stage,case,RENEWAL,size,128,emission)
        s['renderer']['output_directory'] = str(d)
        save(d/'scene.json',s)
        spec = dict(scene=str(d/'scene.json'),seed=SEED,samples=1,sample_start=sample,
                    pixels=[dict(xy=pixel,label=label)])
        save(d/'spec.json',spec)
        with (d/'replay.log').open('w',encoding='utf-8') as log:
            subprocess.run([str(exe),str(d/'spec.json'),str(d/'replay.json')],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=load(d/'replay.json')
        assert result['observer_mismatches']==0 and result['verified_samples']==1
        path=result['paths'][0]
        assert not path['termination'].startswith(('nonfinite','runtime_error'))
        assert not path['renderer_clamped']
        for event in path['events']:
            if 'emission_added' in event:
                assert all(v is not None and np.isfinite(v) for v in event['emission_added'])
        checks.append(dict(label=label,case=case,pixel=pixel,sample=sample,
            termination=path['termination'],radiance=path['radiance'],
            replay_sha256=sha(d/'replay.json'),folder=str(d)))
    save(OUT/stage/'numerical_regressions.json',dict(passed=True,checks=checks,
        executable_sha256=sha(exe),core_sha256=sha(ROOT/'build/Release/core.lib')))
    print('Verified zero-length and short-segment regression paths',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['batch', 'pilot', 'preview', 'render', 'check'])
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--emission', type=float, default=EMISSION)
    parser.add_argument('--spp', type=int, default=128)
    parser.add_argument('--stage')
    parser.add_argument('--light-type', choices=['area', 'point'], default='area',
                        help='Point matches the total power of the area light at --emission')
    parser.add_argument('--preview-spp', type=int, default=32)
    parser.add_argument('--report', action='store_true', help='Generate figures after rendering; requires plotting packages')
    args = parser.parse_args()
    if args.emission <= 0 or not np.isfinite(args.emission) or args.spp < 2 or args.threads < 1 or args.preview_spp < 2:
        parser.error('Require positive finite emission, spp >= 2, threads >= 1')
    stage = args.stage or (STAGE if args.emission == EMISSION and args.spp == 128
                           else f'light{args.emission:g}_full_renewal_mis_spp{args.spp}')
    if args.light_type == 'point' and not args.stage:
        stage = 'point_power_'+stage
    if Path(stage).name != stage or stage in ('.','..'):
        parser.error('Stage must be a single directory name')
    pilot_stage = stage+'_validated_pilot'
    if args.command in ('batch', 'pilot'):
        numerical_regressions(stage,args.emission)
        render(pilot_stage,32,8,('total','single','multiple_check'),args.threads,args.emission,args.light_type)
        check_decomposition(pilot_stage,args.emission,args.light_type)
        render_aov(pilot_stage,32,8,args.threads,compare_native=True,
                   emission=args.emission,pilot_stage=pilot_stage,light_type=args.light_type)
    elif args.command == 'check':
        check_decomposition(pilot_stage,args.emission,args.light_type)
    if args.command in ('batch', 'preview', 'render'):
        validation = load(OUT/pilot_stage/'decomposition_checks.json')
        assert validation['passed'] and len(validation['checks']) == len(CASES)*len(MODES)
        pilot = load(OUT/pilot_stage/'aov_run.json')
        assert len(pilot['sources']) == len(CASES)*len(MODES)
        assert pilot['aov_sha256'] == sha(AOV_EXE) and pilot['renderer_sha256'] == renderer()[1]
        assert pilot['emission'] == args.emission
        assert pilot.get('light',light_parameters(args.emission)) == light_parameters(args.emission,args.light_type)
        output_stage = stage+f'_preview{args.preview_spp}' if args.command == 'preview' else stage
        output_spp = args.preview_spp if args.command == 'preview' else args.spp
        if args.command == 'preview':
            save(OUT/output_stage/'numerical_regressions.json',load(OUT/stage/'numerical_regressions.json'))
        render_aov(output_stage,256,output_spp,args.threads,emission=args.emission,
                   pilot_stage=pilot_stage,light_type=args.light_type)
        if args.report:
            from plane_strong_light_figures import main
            main(output_stage)
