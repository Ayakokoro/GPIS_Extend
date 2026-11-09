"""Run the isolated 3D Renewal+ surface-shadow conditioning experiment."""
from __future__ import annotations

import argparse
import copy
import subprocess
import time
from pathlib import Path

import numpy as np
import gpis_shaderball_renewal_3d as renewal
from run import load, save, sha, pfm

ROOT = renewal.ROOT
OUT = renewal.REPORT / 'shadow_condition_ablation'
BEFORE = ROOT / 'build/Release/tungsten_before_shadow_condition_ablation.exe'
AFTER = ROOT / 'build/Release/tungsten_shadow_condition_ablation.exe'
STAGE = 'renewal_3d_shadow_conditioned'
SOURCES = ('src/core/media/Medium.hpp', 'src/core/media/SparseConvolutionNoiseMedium.hpp',
           'src/core/media/SparseConvolutionNoiseMedium.cpp', 'src/core/integrators/TraceBase.cpp')


def archive():
    if (OUT / 'experiment.json').exists():
        raise RuntimeError('The floor-conditioned experiment is archived. Its executable, source snapshots and records must not be overwritten; use the recorded executable in a separate output directory.')
    assert sha(BEFORE) == load(renewal.REPORT / 'renewal_3d_run.json')['renderer_sha256']
    assert sha(renewal.EXE) != sha(BEFORE), 'Experimental renderer has not been built'
    AFTER.write_bytes(renewal.EXE.read_bytes())
    sources = []
    for name in SOURCES:
        before = OUT / 'source_before' / name
        after = OUT / 'source_after' / name
        after.parent.mkdir(parents=True, exist_ok=True)
        after.write_bytes((ROOT / name).read_bytes())
        sources.append(dict(path=name, before_sha256=sha(before), after_sha256=sha(after)))
    diff = subprocess.run(['git', 'diff', '--', *SOURCES], cwd=ROOT,
                          capture_output=True, check=True).stdout
    (OUT / 'experiment.patch').write_bytes(diff)
    save(OUT / 'experiment.json', dict(
        status='rendering', stage=STAGE, sources=sources,
        before_executable=str(BEFORE), before_renderer_sha256=sha(BEFORE),
        experimental_executable=str(AFTER), experimental_renderer_sha256=sha(AFTER),
        patch_sha256=sha(OUT / 'experiment.patch'),
        scope='Only same-medium ordinary-surface direct-light connections for sparse 3D Renewal+ with a valid prior observation.',
        change='Condition the shadow realization on the saved endpoint value and gradient.',
        preserved='firstScatter, first-interval marching, jitter, seeds, all scene parameters and main continuation rules.',
        controls=dict(size=32, spp=8, seed=renewal.SEED),
        full_render=dict(size=256, spp=128, step=.01, seed=renewal.SEED),
        decision_rule='Assess both groups using the pre-existing bottom ROI: signed bias, luminance RMSE, local FLIP and paired split halves. Keep only if improvement is consistent; revert if absent, worse or inconclusive. A reduction alone does not prove the entire difference is explained.',
        decision='Pending full-image FLIP and pre-existing fixed bottom ROI comparison.'))


def controls():
    rows = []
    group = renewal.GROUPS[0]
    for mode in renewal.MODES:
        source = (renewal.destination(group, 'renewal_3d') if mode == renewal.MODE
                  else renewal.original(group, mode))
        original = renewal.verified(source)
        images = {}
        for variant, exe in (('before', BEFORE), ('after', AFTER)):
            d = OUT / 'controls' / mode / variant
            s = copy.deepcopy(original['scene'])
            s['camera']['resolution'] = [32, 32]
            s['renderer'].update(spp=8, spp_step=8, output_directory=str(d))
            save(d / 'scene.json', s)
            begin = time.monotonic()
            with (d / 'render.log').open('w', encoding='utf-8') as log:
                subprocess.run([str(exe), '-t', '32', '-s', str(renewal.SEED),
                                '-d', str(d), str(d / 'scene.json')], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, check=True,
                               creationflags=getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
            assert 'Finished render.' in (d / 'render.log').read_text(errors='replace')
            images[variant] = pfm(d / 'render.pfm')
            rows.append(dict(mode=mode, variant=variant, seconds=time.monotonic()-begin,
                             renderer_sha256=sha(exe),
                             hashes={name: sha(d/name) for name in renewal.FILES}))
        error = float(np.max(np.abs(images['after']-images['before'])))
        changed = int(np.count_nonzero(np.any(images['after'] != images['before'], axis=2)))
        if mode != renewal.MODE:
            assert error == 0, (mode, error)
        else:
            assert error > 0 and changed > 0, 'Experiment did not affect Renewal+'
        rows.append(dict(mode=mode, maximum_absolute_rgb_change=error, changed_pixels=changed))
        print('Control', mode, 'max change', error, 'changed pixels', changed, flush=True)
    save(OUT / 'controls.json', dict(complete=True, records=rows,
         result='All five non-target modes exactly unchanged; target Renewal+ affected.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=32)
    args = parser.parse_args()
    archive()
    controls()
    # The dedicated executable stays available even after a possible rollback.
    renewal.EXE = AFTER
    renewal.render(argparse.Namespace(groups=list(renewal.GROUPS), stage=STAGE,
                                     size=256, spp=128, step=.01, threads=args.threads))
    doc = load(OUT / 'experiment.json')
    doc['status'] = 'rendered_pending_analysis'
    doc['controls_sha256'] = sha(OUT / 'controls.json')
    save(OUT / 'experiment.json', doc)


if __name__ == '__main__':
    main()
