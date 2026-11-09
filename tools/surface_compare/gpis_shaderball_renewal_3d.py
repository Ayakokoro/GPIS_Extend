"""Render 3D Renewal+ by changing only the archived global memory model.

python -B tools/surface_compare/gpis_shaderball_renewal_3d.py render
<plot-python> -B tools/surface_compare/gpis_shaderball_renewal_3d.py report

The original 1D+MIS renders remain in their original folders. The comparison
page is updated to use the matched 3D Renewal+ renders when report completes.
"""
from __future__ import annotations

import argparse
import copy
import subprocess
import time
from pathlib import Path

import numpy as np
from run import load, save, sha, pfm
from gpis_shaderball_renewal import (
    ROOT, BASE, REPORT, ORIGINAL, GLOBAL, GROUPS, SEED, EXE, FILES,
    ROI, MODEL_ROI, WEIGHTS, original, verified, compatibility,
)

MODE = 'sparse_3d_renewal_plus'
MODES = ('classic_local', 'classic_global', 'global_conditional', 'neural', GLOBAL, MODE)
CHANGES = {'correlation_context': 'renewal+'}


def destination(group, stage):
    return BASE/group/stage/MODE/f'seed{SEED}'


def scene(group, stage, size, spp, step):
    source = original(group, GLOBAL)
    result = load(source/'result.json')
    assert sha(source/'scene.json') == result['sha256']['scene.json']
    old = load(source/'scene.json')
    s = copy.deepcopy(old)
    s['media'][0].update(CHANGES, step_size=step)
    s['camera']['resolution'] = [size, size]
    s['renderer'].update(spp=spp, spp_step=min(16, spp),
                         output_directory=str(destination(group, stage)))
    medium = s['media'][0]
    assert medium['type'] == 'sparse_conv_noise'
    assert not medium['1D_sampling'] and medium['1D_sampling_scheme'] == 'uni'
    assert not medium['1D_gradient_correlationXY'] and not medium['single_realization']
    check = copy.deepcopy(s)
    check['media'][0]['correlation_context'] = old['media'][0]['correlation_context']
    check['media'][0]['step_size'] = old['media'][0]['step_size']
    check['camera']['resolution'] = old['camera']['resolution']
    check['renderer'] = old['renderer']
    assert check == old, 'Unexpected scene change beyond declared render controls'
    if stage == 'renewal_3d':
        assert (size, spp, step) == (256, 128, .01)
        check = copy.deepcopy(s)
        check['media'][0]['correlation_context'] = 'global'
        check['renderer']['output_directory'] = old['renderer']['output_directory']
        assert check == old, 'Final global/renewal pair must differ only by context and output path'
    return s, result


def render(args):
    exe_hash = sha(EXE)
    records = []
    for group in args.groups:
        s, old = scene(group, args.stage, args.size, args.spp, args.step)
        d = destination(group, args.stage)
        provenance = dict(old['provenance'], renderer_sha256=exe_hash)
        if (d/'result.json').exists():
            data = verified(d)
            assert data['scene'] == s
            assert data['source']['result']['provenance'] == provenance
            row = data['source']['result']
            print('Reuse', group, args.stage, flush=True)
        else:
            if (d/'render.pfm').exists():
                raise RuntimeError('Unvalidated existing render: '+str(d))
            save(d/'scene.json', s)
            print('Render', group, args.stage, args.size, args.spp, flush=True)
            start = time.monotonic()
            with (d/'render.log').open('w', encoding='utf-8') as log:
                completed = subprocess.run(
                    [str(EXE), '-t', str(args.threads), '-s', str(SEED),
                     '-d', str(d), str(d/'scene.json')], cwd=ROOT, stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
            if completed.returncode:
                raise RuntimeError('Renderer failed: '+str(d/'render.log'))
            assert sha(EXE) == exe_hash, 'Renderer changed during the render'
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('render', 'compatibility', 'report'))
    parser.add_argument('--groups', nargs='+', choices=GROUPS, default=list(GROUPS))
    parser.add_argument('--stage', default='renewal_3d')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--spp', type=int, default=128)
    parser.add_argument('--step', type=float, default=.01)
    parser.add_argument('--threads', type=int, default=32)
    args = parser.parse_args()
    if (min(args.size, args.spp, args.threads) <= 0 or args.size % 16 or args.spp % 2
            or not np.isfinite(args.step) or args.step <= 0):
        parser.error('Require positive size divisible by 16, positive even spp, positive threads and step')
    if args.command == 'render':
        render(args)
    elif args.command == 'compatibility':
        compatibility(args)
    else:
        from gpis_shaderball_renewal_3d_figures import report
        report(args)
