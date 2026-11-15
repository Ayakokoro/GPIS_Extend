"""Add a matched global_conditional run to the Adaptive GPIS small-light report."""
import argparse
import copy
from pathlib import Path
import re
import subprocess
import time

from render_adaptive_shaderball_small_light import ROOT, FATAL, metrics, report, read, save, sha, differences

SOURCE = ROOT/'outputs/gpis_shaderball_small_light_20261114/final/global_conditional/seed20261024'
OUTPUT = ROOT/'outputs/adaptive_gpis_mis_20261115/small_light_256_128spp'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    output=args.output.resolve()
    manifest=read(output/'manifest.json')
    adaptive=read(output/'adaptive/result.json')
    assert manifest['setup']=='isolated-sphere'
    exe=Path(manifest['executable'])
    for name,digest in manifest['runtime_sha256'].items():
        assert sha(exe.parent/name)==digest,'Runtime differs from the Adaptive run'
    source=read(SOURCE/'scene.json')
    assert sha(SOURCE/'scene.json')==read(SOURCE/'result.json')['sha256']['scene.json']
    scene=copy.deepcopy(adaptive['identity']['scene'])
    old=scene['media'][0]
    medium=source['media'][0]
    assert medium['type']=='global_conditional'
    assert medium['gaussian_process']==old['gaussian_process']
    assert medium['material']==old['phase_function']['bsdf']
    assert medium['max_bounces']==old['max_bounces']
    for key in ('bsdfs','primitives','integrator'):
        assert scene[key]==source[key],key
    camera=copy.deepcopy(source['camera']);camera['resolution']=scene['camera']['resolution']
    assert camera==scene['camera']
    scene['media']=[copy.deepcopy(medium)]
    folder=output/'global_conditional'
    identity=dict(scene=scene,manifest=manifest,source=str(SOURCE/'scene.json'),source_sha256=sha(SOURCE/'scene.json'))
    if (folder/'result.json').exists():
        row=read(folder/'result.json');assert row['identity']==identity
    else:
        save(folder/'source.json',source)
        save(folder/'scene.json',scene)
        save(folder/'changes.json',differences(source,scene))
        command=[str(exe),'-t',str(manifest['threads']),'-s',str(manifest['seed']),'-d',str(folder),str(folder/'scene.json')]
        save(folder/'command.json',command)
        print('RENDER global_conditional',manifest['size'],manifest['spp'],flush=True)
        begin=time.perf_counter()
        with (folder/'render.log').open('wb') as log:
            completed=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                timeout=1800,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        wall=time.perf_counter()-begin
        assert sha(exe)==manifest['runtime_sha256'][exe.name]
        log=(folder/'render.log').read_text(encoding='utf-8',errors='replace')
        if completed.returncode or FATAL.search(log) or f'Completed {manifest["spp"]}/{manifest["spp"]} spp' not in log:
            raise RuntimeError(f'Render failed: {folder / "render.log"}')
        row=dict(identity=identity,mode='global_conditional',wall_seconds=wall,
            render_seconds=float(re.findall(r'Completed \d+/\d+ spp, ([\d.]+)/',log)[-1]),
            **metrics(folder,manifest['size']),
            warnings=dict(zero_length_path_terminations=log.count('Path tracing bounces ended with farT == 0'),
                          nee_sanity_checks=log.count('1D NEE sanity check failed')),
            sha256={name:sha(folder/name) for name in ('scene.json','source.json','changes.json','command.json',
                'render.png','render.pfm','samplesA.pfm','samplesB.pfm','samplesVariance.pfm','render.log')})
        save(folder/'result.json',row)
        print('DONE global_conditional',row['render_seconds'],'s',flush=True)
    for name,digest in row['sha256'].items():assert sha(folder/name)==digest,name
    report(output,manifest,adaptive)
    print(output/'comparison.html',flush=True)


if __name__=='__main__':
    main()
