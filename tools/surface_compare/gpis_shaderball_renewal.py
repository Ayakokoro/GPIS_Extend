"""Add full GPIS Renewal+ MIS to the archived A/B shaderball experiment.

python -B tools/surface_compare/gpis_shaderball_renewal.py render
<plot-python> -B tools/surface_compare/gpis_shaderball_renewal.py report

Scenes are copied from the hash-verified original runs. The five existing
comparison modes are reused, with their original renderer provenance retained.
"""
from __future__ import annotations

import argparse
import copy
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from run import ROOT, load, save, sha, pfm
from gpis_shaderball import ROI, MODEL_ROI, WEIGHTS

BASE = ROOT/'outputs/gpis_shaderball_compare'
REPORT = BASE/'ell0p2_renewal_comparison'
ORIGINAL = BASE/'ell0p2_neural_comparison/manifest.json'
MODE = 'sparse_1d_renewal_full_mis'
GLOBAL = 'sparse_3d_global'
MODES = ('classic_local', 'classic_global', 'global_conditional', 'neural', GLOBAL, MODE)
GROUPS = ('ell0p2_sigma0p02', 'ell0p2_sigma0p04')
SEED = 20261024
EXE = ROOT/'build/Release/tungsten.exe'
CHANGES = {'correlation_context': 'renewal+', '1D_sampling': True,
           '1D_sampling_scheme': 'mis', '1D_gradient_correlationXY': True}
ERRORS = re.compile(r'Caught an internal error|unrecoverable|Gradient invalid|'
    r'Unable to load|Wrong normal direction|Failed to sample gradient|Gradient zero|'
    r'Conditioning .* error|sanity check failed|GPContext does not exist', re.I)
FILES = ('scene.json', 'render.pfm', 'render.png', 'samplesA.pfm',
         'samplesB.pfm', 'samplesVariance.pfm', 'render.log')


def original(group, mode):
    return BASE/group/'final'/mode/f'seed{SEED}'


def destination(group, stage):
    return BASE/group/stage/MODE/f'seed{SEED}'


def verified(d):
    result = load(d/'result.json')
    for name, digest in result['sha256'].items():
        assert sha(d/name) == digest, str(d/name)
    rgb = pfm(d/'render.pfm')
    assert rgb.shape == (*result['resolution'], 3)
    halves = [pfm(d/f'samples{c}.pfm') for c in 'AB']
    assert np.allclose((halves[0]+halves[1])/2, rgb, rtol=3e-6, atol=3e-7)
    pfm(d/'samplesVariance.pfm')
    log = (d/'render.log').read_text(encoding='utf-8', errors='replace')
    assert 'Finished render.' in log and f"Completed {result['spp']}/{result['spp']} spp" in log
    assert not ERRORS.search(log), str(d/'render.log')
    return dict(scene=load(d/'scene.json'), rgb=rgb, y=rgb@WEIGHTS,
                halves=[x@WEIGHTS for x in halves],
                source=dict(folder=str(d), result=result))


def scene(group, stage, size, spp, step):
    d = original(group, 'sparse_3d_global')
    r = load(d/'result.json')
    assert sha(d/'scene.json') == r['sha256']['scene.json']
    old = load(d/'scene.json')
    s = copy.deepcopy(old)
    s['media'][0].update(CHANGES, step_size=step)
    s['camera']['resolution'] = [size, size]
    s['renderer'].update(spp=spp, spp_step=min(16, spp),
        output_directory=str(destination(group, stage)))
    check = copy.deepcopy(s)
    for key in (*CHANGES, 'step_size'):
        check['media'][0][key] = old['media'][0][key]
    check['camera']['resolution'] = old['camera']['resolution']
    check['renderer'] = old['renderer']
    assert check == old
    assert not s['media'][0]['single_realization']
    assert s['integrator']['enable_volume_light_sampling']
    return s, r


def render(args):
    exe_hash = sha(EXE)
    records = []
    for group in args.groups:
        s, old = scene(group, args.stage, args.size, args.spp, args.step)
        d = destination(group, args.stage)
        provenance = dict(old['provenance'], renderer_sha256=exe_hash)
        if (d/'result.json').exists():
            data = verified(d)
            assert data['scene'] == s and data['source']['result']['provenance'] == provenance
            row = data['source']['result']
            print('Reuse', group, args.stage, flush=True)
        else:
            if (d/'render.pfm').exists():
                raise RuntimeError('Unvalidated existing render: '+str(d))
            save(d/'scene.json', s)
            print('Render', group, args.stage, args.size, args.spp, flush=True)
            start = time.monotonic()
            with (d/'render.log').open('w', encoding='utf-8') as log:
                p = subprocess.run([str(EXE), '-t', str(args.threads), '-s', str(SEED),
                    '-d', str(d), str(d/'scene.json')], cwd=ROOT, stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
            if p.returncode:
                raise RuntimeError('Renderer failed: '+str(d/'render.log'))
            rgb = pfm(d/'render.pfm')
            row = dict(seconds=time.monotonic()-start, resolution=[args.size, args.size],
                spp=args.spp, seed=SEED, threads=args.threads, provenance=provenance,
                source_scene_sha256=old['sha256']['scene.json'], changes=CHANGES,
                mean_luminance=float((rgb@WEIGHTS).mean()),
                sha256={name:sha(d/name) for name in FILES})
            save(d/'result.json', row)
            verified(d)
            print('Done', group, round(row['seconds'], 2), 'mean Y', row['mean_luminance'], flush=True)
        records.append(dict(group=group, folder=str(d), result=row))
        save(REPORT/f'{args.stage}_run.json', dict(complete=len(records)==len(args.groups),
            executable=str(EXE), renderer_sha256=exe_hash, runner_sha256=sha(Path(__file__)),
            settings=dict(stage=args.stage, size=args.size, spp=args.spp, step=args.step,
                          seed=SEED, threads=args.threads), sources=records))


def compatibility(args):
    """Replay all 128 samples in fixed groove/control tiles with the current tracer."""
    exe = ROOT/'build/Release/surface_bounce_images.exe'
    records = []
    # These locations come from the existing bounce-order investigation.
    tiles = {'groove': [81,97,89,105], 'center': [122,78,130,86],
             'base': [173,162,181,170]}
    for group in args.groups:
        for mode in MODES[:-1]:
            src = original(group, mode)
            old = verified(src)
            for label, bounds in tiles.items():
                d = REPORT/'compatibility'/group/mode/label
                spec = dict(scene=str(src/'scene.json'), samples=128, threads=args.threads,
                    seed=SEED, bounds=bounds, output=str(d),
                    verify_pixels=[[bounds[0],bounds[1]]], verify_samples=128)
                # The diagnostic CLI accepts spec.json and output directory separately.
                save(d/'spec.json', spec)
                identity = dict(executable_sha256=sha(exe), spec_sha256=sha(d/'spec.json'))
                if not (d/'verification.json').exists():
                    with (d/'trace.log').open('w', encoding='utf-8') as log:
                        subprocess.run([str(exe), str(d/'spec.json'), str(d)], cwd=ROOT,
                            stdout=log, stderr=subprocess.STDOUT, check=True,
                            creationflags=getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
                else:
                    prior = load(d/'verification.json')
                    assert prior['identity'] == identity
                    assert sha(d/'beauty.pfm') == prior['beauty_sha256']
                result = load(d/'result.json')
                assert result['observer_mismatches'] == 0 and result['verified_samples'] == 128
                x0,y0,x1,y1 = bounds
                actual = pfm(d/'beauty.pfm')[y0:y1,x0:x1]
                expected = old['rgb'][y0:y1,x0:x1]
                assert np.allclose(actual, expected, rtol=3e-5, atol=2e-7), (group, mode, label)
                row = dict(group=group, mode=mode, tile=label, bounds=bounds,
                    samples=128, max_absolute_error=float(np.abs(actual-expected).max()),
                    passed=True, identity=identity, beauty_sha256=sha(d/'beauty.pfm'),
                    original_result_sha256=sha(src/'result.json'))
                save(d/'verification.json', row)
                records.append(row)
            print('Compatibility passed', group, mode, flush=True)
    save(REPORT/'renderer_compatibility.json', dict(passed=True, checks=records,
        renderer_sha256=sha(EXE), diagnostic_sha256=sha(exe),
        scope='Three fixed 8x8 tiles per group/mode, all 128 samples; not a full-image equivalence proof.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('render', 'compatibility', 'report'))
    p.add_argument('--groups', nargs='+', choices=GROUPS, default=list(GROUPS))
    p.add_argument('--stage', default='renewal_full_mis')
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--spp', type=int, default=128)
    p.add_argument('--step', type=float, default=.01)
    p.add_argument('--threads', type=int, default=32)
    args = p.parse_args()
    if min(args.size,args.spp,args.threads) <= 0 or args.size%16 or args.spp%2 or not np.isfinite(args.step) or args.step<=0:
        p.error('Require positive size divisible by 16, positive even spp, positive threads and step')
    if args.command == 'render': render(args)
    elif args.command == 'compatibility': compatibility(args)
    else:
        from gpis_shaderball_renewal_figures import report
        report(args)
