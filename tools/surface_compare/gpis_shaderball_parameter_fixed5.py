"""Add matched fixed5 gradient correction to all ten existing shaderball groups.

Run with the local plot Python: prepare, pilot, run, report, or status.
Only fp_gradient and the output directory differ from each full-size baseline.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import traceback

import gpis_shaderball_parameter_comparison as s

MODE = 'global_conditional_fixed5'
LABEL = 'global_conditional + fixed5'
SETTINGS = dict(enabled=True, bridge_correction=True, anchor_selection='fixed5_snr')
PAIRS = tuple((MODE, ref) for ref in ('global_conditional', 'classic_global', 'gpis_renewal_plus', 'gpis_global'))
HISTORY = 'history/before_fixed5'
ERRORS = re.compile(s.ERRORS.pattern+r'|invalid FP|NumericError|NeedHigherPrecision', re.I)


def scene_for(args, ell, sigma, pilot=False):
    baseline = s.folder(args, ell, sigma, 'global_conditional')
    scene = copy.deepcopy(s.load(baseline/'scene.json'))
    assert scene['media'][0]['type'] == 'global_conditional'
    assert scene['media'][0]['non_gp_surface_policy'] == 'retain_endpoint'
    assert scene['media'][0]['fp_gradient'] == dict(enabled=False)
    scene['media'][0]['fp_gradient'] = copy.deepcopy(SETTINGS)
    scene['renderer']['output_directory'] = str(s.folder(args, ell, sigma, MODE, pilot))
    if pilot:
        scene['camera']['resolution'] = [32,32]
        scene['renderer'].update(spp=4, spp_step=4, enable_resume_render=False)
    else:
        check = copy.deepcopy(scene)
        check['media'][0]['fp_gradient'] = dict(enabled=False)
        check['renderer']['output_directory'] = str(baseline)
        assert check == s.load(baseline/'scene.json'), 'Unexpected change beyond fp_gradient and output'
    return scene


def identity(args, ell, sigma, pilot=False):
    manifest = s.load(args.output/'manifest.json')
    return dict(scene=scene_for(args,ell,sigma,pilot), seed=s.SEED,
                renderer_sha256=manifest['runtime_sha256']['tungsten.exe'])


def verified(args, ell, sigma, pilot=False):
    folder = s.folder(args,ell,sigma,MODE,pilot)
    row = s.load(folder/'result.json')
    assert row['identity'] == identity(args,ell,sigma,pilot)
    assert row['fp_gradient'] == SETTINGS and row['fp_gradient_statistics']['collisions'] > 0
    for name, expected in row['sha256'].items():
        assert s.sha(folder/name) == expected, str(folder/name)
    return row


def prepare(args):
    manifest = s.load(args.output/'manifest.json')
    history = args.output/HISTORY
    old = s.load(history/'manifest.json')
    old_script = history/Path(s.__file__).name
    assert s.sha(old_script) == old['report_runner_sha256']
    previous_tree = ast.parse(old_script.read_text(encoding='utf-8'))
    current_tree = ast.parse(Path(s.__file__).read_text(encoding='utf-8'))
    for name in ('scene_for','render_one','verified','validate_scenes'):
        get = lambda tree: next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(get(previous_tree)) == ast.dump(get(current_tree)), name
    if manifest.get('fixed5_extension'):
        extension = manifest['fixed5_extension']
        assert extension['runner_sha256'] == s.sha(Path(__file__)), 'Fixed5 runner changed since setup'
        assert extension['fp_gradient'] == SETTINGS
    else:
        extension = dict(created_at=s.now(), runner=str(Path(__file__).relative_to(s.ROOT)),
            runner_sha256=s.sha(Path(__file__)), fp_gradient=SETTINGS,
            mode=MODE, expected_new_renders=len(s.GROUPS), expected_new_pairs=len(s.GROUPS)*len(PAIRS),
            baseline_metrics=f'{HISTORY}/metrics.json', baseline_metrics_sha256=s.sha(history/'metrics.json'),
            original_outputs=f'{HISTORY}/original_output_sha256.json',
            original_outputs_sha256=s.sha(history/'original_output_sha256.json'))
    manifest.update(fixed5_extension=extension, comparison_modes=list(s.MODES)+( [MODE] ),
        comparison_pairs=[list(p) for p in s.PAIRS+PAIRS],
        report_runner_sha256=s.sha(Path(s.__file__)))
    s.save(args.output/'manifest.json',manifest)
    s.prepare(args)
    for ell,sigma in s.GROUPS:
        s.verified(args,ell,sigma,'global_conditional')
        folder = s.folder(args,ell,sigma,MODE)
        scene = scene_for(args,ell,sigma)
        if (folder/'scene.json').exists():
            assert s.load(folder/'scene.json') == scene
        else:
            s.save(folder/'scene.json',scene)
    return manifest


def render_one(args,ell,sigma,pilot=False):
    folder = s.folder(args,ell,sigma,MODE,pilot)
    expected = identity(args,ell,sigma,pilot)
    if (folder/'result.json').exists():
        return verified(args,ell,sigma,pilot)
    if (folder/'job.json').exists():
        assert s.load(folder/'job.json')['identity'] == expected
    if (folder/'scene.json').exists():
        assert s.load(folder/'scene.json') == expected['scene']
    if (folder/'render.log').exists():
        attempts=folder/'attempts'
        attempts.mkdir(exist_ok=True)
        shutil.copy2(folder/'render.log',attempts/f'{time.time_ns()}.log')
    s.save(folder/'scene.json',expected['scene'])
    command=[str(args.output/'renderer/tungsten.exe'),'-t',str(args.threads),'-s',str(s.SEED),
             '-d',str(folder),str(folder/'scene.json')]
    started=time.monotonic()
    print('START',s.key(ell,sigma),MODE,'pilot' if pilot else '256x256 128spp',flush=True)
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0)
    with (folder/'render.log').open('wb') as log:
        process=subprocess.Popen(command,cwd=s.ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
        s.save(folder/'job.json',dict(identity=expected,process_id=process.pid,command=command,started_at=s.now()))
        try:
            while process.poll() is None:
                if not pilot:
                    s.status(args,active=str(folder),failures=args.failures)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            process.terminate()
            process.wait()
            raise
    text=(folder/'render.log').read_text(encoding='utf-8',errors='replace')
    spp=expected['scene']['renderer']['spp']
    assert process.returncode==0 and 'Finished render.' in text,str(folder/'render.log')
    assert re.search(rf'Completed\s+{spp}/{spp}\s+spp',text),str(folder/'render.log')
    assert not ERRORS.search(text),str(folder/'render.log')
    match=re.search(r'GPIS conditional FP-gradient statistics: (\{[^\n]+\})',text)
    assert match,'Missing fixed5 execution counters'
    counters=json.loads(match[1])
    assert counters['collisions']>0 and counters['sampling_fallbacks']>=0
    discarded=re.search(r'GPIS conditional discarded surface births: (\{[^\n]+\})',text)
    assert discarded and json.loads(discarded[1])==dict(paths=0,shadow_connections=0)
    assert s.sha(args.output/'renderer/tungsten.exe')==expected['renderer_sha256']
    rgb,a,b,variance=[s.pfm(folder/name) for name in ('render.pfm','samplesA.pfm','samplesB.pfm','samplesVariance.pfm')]
    size=expected['scene']['camera']['resolution'][0]
    assert rgb.shape==variance.shape==(size,size,3) and rgb.max()>0
    assert s.np.allclose((a+b)/2,rgb,rtol=4e-6,atol=4e-7)
    with s.Image.open(folder/'render.png') as image:
        assert image.size==(size,size)
    baseline=s.folder(args,ell,sigma,'global_conditional')
    row=dict(identity=expected,seconds=time.monotonic()-started,finished_at=s.now(),command=command,
        resolution=[size,size],spp=spp,seed=s.SEED,threads=args.threads,
        fp_gradient=copy.deepcopy(SETTINGS),fp_gradient_statistics=counters,
        discarded_surface_births=json.loads(discarded[1]),
        baseline_scene_sha256=s.sha(baseline/'scene.json'),baseline_result_sha256=s.sha(baseline/'result.json'),
        mean_luminance=float((rgb@s.Y).mean()),sha256={name:s.sha(folder/name) for name in s.FILES})
    s.save(folder/'result.json',row)
    print('DONE',s.key(ell,sigma),round(row['seconds'],1),'sec; corrected collisions',counters['collisions'],flush=True)
    return verified(args,ell,sigma,pilot)


def report(args):
    s.report(args)
    history=args.output/HISTORY
    before=s.load(history/'metrics.json')
    metrics=s.load(args.output/'metrics.json')
    lookup={(r['group'],r['test'],r['reference']):r for r in metrics['comparisons']}
    for row in before['comparisons']:
        assert lookup[(row['group'],row['test'],row['reference'])]==row,'Existing comparison changed'
    original=s.load(history/'original_output_sha256.json')
    for name,expected in original.items():
        assert s.sha(args.output/name)==expected,'Original output changed: '+name
    fixed5_rows=[]
    for ell,sigma in s.GROUPS:
        folder=s.folder(args,ell,sigma,MODE)
        if (folder/'result.json').exists():
            row=verified(args,ell,sigma)
            fixed5_rows.append(dict(group=s.key(ell,sigma),fp_gradient_statistics=row['fp_gradient_statistics']))
    metrics.update(fixed5_results=fixed5_rows,prior_pairs_preserved=len(before['comparisons']))
    s.save(args.output/'metrics.json',metrics)
    verification=s.load(args.output/'verification.json')
    verification.update(prior_pairs_preserved=len(before['comparisons']),
        native_flip_records=len(list((args.output/'flip').glob('*/*/*/result.json'))),
        original_outputs_preserved=len(original),original_output_hashes_verified=True,
        fixed5_renders=len(fixed5_rows),fixed5_runner_sha256=s.sha(Path(__file__)),
        report_runner_sha256=s.sha(Path(s.__file__)),metrics_sha256=s.sha(args.output/'metrics.json'),
        figure_sha256={p.relative_to(args.output).as_posix():s.sha(p) for p in (args.output/'figures').glob('*/*.png')})
    s.save(args.output/'verification.json',verification)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','pilot','run','report','status'))
    parser.add_argument('--output',type=Path,default=s.DEFAULT_OUT)
    parser.add_argument('--threads',type=int,default=32)
    parser.add_argument('--flip',type=Path,default=s.DEFAULT_FLIP)
    args=parser.parse_args()
    args.output=args.output.resolve()
    args.failures=[]
    assert args.threads>0
    if args.command=='status':
        print(json.dumps(s.load(args.output/'progress.json'),ensure_ascii=False,indent=2))
        return
    prepare(args)
    if args.command in ('prepare','report'):
        s.status(args,complete=all((s.folder(args,e,v,MODE)/'result.json').exists() for e,v in s.GROUPS))
        report(args)
        return
    lock=args.output/'runner.lock'
    descriptor=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(descriptor,'w') as stream:
        json.dump(dict(pid=os.getpid(),started_at=s.now(),command='fixed5 '+args.command),stream)
    try:
        pilots=[]
        for ell,sigma in ((.2,.02),(.05,.08),(.4,.01)):
            row=render_one(args,ell,sigma,True)
            fixed=s.pfm(s.folder(args,ell,sigma,MODE,True)/'render.pfm')
            baseline=s.pfm(s.folder(args,ell,sigma,'global_conditional',True)/'render.pfm')
            changed=int((fixed!=baseline).any(axis=2).sum())
            assert changed>0,'fixed5 did not change the pilot image'
            pilots.append(dict(ell=ell,sigma=sigma,result=row,changed_pixels=changed))
        s.save(args.output/'fixed5_pilot_validation.json',dict(passed=True,sources=pilots,
            scope='32x32 4 spp at baseline and extreme ratios; verify active correction and matched configuration.'))
        if args.command=='pilot':
            return
        s.status(args)
        report(args)
        for ell,sigma in s.GROUPS:
            try:
                render_one(args,ell,sigma)
            except Exception as exc:
                args.failures.append(dict(group=s.key(ell,sigma),mode=MODE,error=str(exc),traceback=traceback.format_exc()))
                print('FAILED',args.failures[-1],flush=True)
            s.save(args.output/'fixed5_failures.json',args.failures)
            s.save(args.output/'failures.json',args.failures)
            s.status(args,failures=args.failures)
            report(args)
        s.status(args,failures=args.failures,complete=not args.failures)
        report(args)
        if args.failures:
            raise RuntimeError(f'{len(args.failures)} fixed5 renders failed')
        print('COMPLETE: 50 renders, 100 FLIP pairs, 200 native FLIP records; original 60 pairs preserved.',flush=True)
    finally:
        lock.unlink()


if __name__=='__main__':
    main()
