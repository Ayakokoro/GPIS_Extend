"""Compare the original shaderball at (ell, sigma)=(.2,.02) and (.2,.04).

python -B tools/surface_compare/gpis_shaderball_sweep.py run
<plot-python> -B tools/surface_compare/gpis_shaderball_sweep.py figures

The conditional (.1,.02) baseline is refreshed after the endpoint precision fix.
Other baseline modes are reused after pixel checks against the previous renderer.
"""
from __future__ import annotations
import argparse
import copy
import subprocess
from pathlib import Path
from types import SimpleNamespace
from gpis_shaderball import ROOT, OUT, MODES, WEIGHTS, bake, render, folder, output
from run import load,save,pfm
from shaderball_length import digest

REPORT = OUT.parent/'ell0p2_comparison'
GROUPS = ((.1,.02),(.2,.02),(.2,.04))
PLOT_PYTHON = Path('C:/Users/user/miniconda3/envs/gaussian_splatting/python.exe')
BASELINE_REFRESH_STAGE = 'endpoint_fix'


def configuration(ell,sigma,**overrides):
    a=SimpleNamespace(ell=ell,sigma=sigma,stage='final',size=256,spp=128,
        seed=20261024,threads=32,step=.01,modes=list(MODES))
    for key,value in overrides.items(): setattr(a,key,value)
    return a


def run():
    groups=[configuration(*p) for p in GROUPS[1:]]
    for a in groups: bake(a)
    # Complete both sets of analytic modes before the more expensive GPIS images.
    for a in groups:
        b=copy.copy(a);b.modes=list(MODES[:-1]);render(b)
    # Refresh the conditional baseline after the precise-endpoint fix, and
    # verify that existing non-conditional baseline images can be reused.
    render(configuration(.1,.02,stage=BASELINE_REFRESH_STAGE,modes=['global_conditional']))
    render(configuration(.1,.02,stage='endpoint_verification',size=64,spp=8,step=.005,
        modes=['sparse_3d_global','gpis_fixed']))
    verify_renderer_compatibility()
    for a in groups:
        b=copy.copy(a);b.modes=['sparse_3d_global'];render(b)
    for a in groups:
        b=copy.copy(a);b.modes=['gpis_fixed'];b.spp=32;render(b)
    for a in groups:
        for stage,step in [('step_base',.01),('step_half',.005)]:
            b=copy.copy(a);b.modes=['sparse_3d_global'];b.stage=stage
            b.size=96;b.spp=32;b.step=step;render(b)
    subprocess.run([str(PLOT_PYTHON),'-B',str(Path(__file__).resolve()),'figures'],cwd=ROOT,check=True)
    print('Both shaderball parameter groups complete',flush=True)


def verify_renderer_compatibility():
    old=digest(ROOT/'build/Release/tungsten_before_endpoint_fix.exe')
    new=digest(ROOT/'build/Release/tungsten.exe')
    checks=[]
    for mode in ('classic_local','classic_global','sparse_3d_global','gpis_fixed'):
        if mode in ('classic_local','classic_global'):
            a=folder(configuration(.2,.02,stage='before_endpoint_fix'),mode)
            b=folder(configuration(.2,.02),mode)
        else:
            a=folder(configuration(.1,.02,stage='pilot'),mode)
            b=folder(configuration(.1,.02,stage='endpoint_verification'),mode)
        ra,rb=load(a/'result.json'),load(b/'result.json')
        assert ra['provenance']['renderer_sha256']==old and rb['provenance']['renderer_sha256']==new
        sa,sb=load(a/'scene.json'),load(b/'scene.json')
        for s in (sa,sb): s['renderer'].pop('output_directory')
        assert sa==sb
        images=('render.pfm','samplesA.pfm','samplesB.pfm','samplesVariance.pfm')
        for name in images:
            assert digest(a/name)==ra['sha256'][name]
            assert digest(b/name)==rb['sha256'][name]
            assert ra['sha256'][name]==rb['sha256'][name],f'Non-conditional regression: {mode}/{name}'
        checks.append(dict(mode=mode,before=str(a),after=str(b),resolution=rb['resolution'],spp=rb['spp'],
            equal_images={name:rb['sha256'][name] for name in images}))
    a=folder(configuration(.1,.02),'global_conditional')
    b=folder(configuration(.1,.02,stage=BASELINE_REFRESH_STAGE),'global_conditional')
    before,after=pfm(a/'render.pfm'),pfm(b/'render.pfm')
    delta=(after-before)@WEIGHTS
    change=dict(changed_pixels=int((before!=after).any(axis=2).sum()),
        mean_luminance_difference=float(delta.mean()),luminance_rmse=float((delta**2).mean()**.5),
        maximum_absolute_luminance_difference=float(abs(delta).max()),
        relative_mean_difference=float(delta.mean()/(before@WEIGHTS).mean()))
    save(REPORT/'renderer_compatibility.json',dict(previous_renderer_sha256=old,current_renderer_sha256=new,
        nonconditional_checks=checks,conditional_baseline_before=str(a),conditional_baseline_after=str(b),
        conditional_baseline_bit_identical=digest(a/'render.pfm')==digest(b/'render.pfm'),
        conditional_baseline_change=change,
        fix='Use the renewed observation position as the conditional path endpoint, avoiding inconsistent double normalization rounding.',
        regression_test='global_conditional_endpoint_precision: reproduced the original error before the fix; both distance and shadow endpoint renewal are tested.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','figures'))
    args=parser.parse_args()
    if args.command=='run': run()
    else:
        from gpis_shaderball_sweep_figures import figures
        figures()
