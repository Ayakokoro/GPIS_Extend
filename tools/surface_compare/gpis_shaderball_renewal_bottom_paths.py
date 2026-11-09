"""Selected-pixel replay of 3D Renewal+/global at the base, with step refinement.

Uses the established production observer and its input/output checks. This is
not a full-image convergence test; the seven base pixels are fixed geometrically.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import gpis_shaderball_step_audit as replay
from gpis_shaderball_renewal_3d import (
    REPORT, GROUPS, GLOBAL, MODE, SEED, destination,
)
from run import load, save, sha

OUT = REPORT/'bottom_audit'/'paths'
PIXELS = [('left_upper',(62,190)), ('left_tip',(74,202)), ('left_inner',(88,205)),
          ('right_inner',(168,205)), ('right_tip',(183,198)), ('right_upper',(197,185)),
          ('base_center',(128,193)), ('center_control',(126,82))]
STEPS = (.01, .0025)
SAMPLES = 512


def folder(group, mode, step):
    return OUT/group/mode/str(step).replace('.', 'p')


def trace_statistics(group, mode, step):
    doc = load(folder(group,mode,step)/'paths.json')
    by_pixel = {}
    for label,xy in PIXELS:
        paths = sorted((p for p in doc['paths'] if p['pixel_label']==label),key=lambda p:p['sample'])
        orders = np.array([p['medium_scatters'] for p in paths])
        ys = np.array([p['radiance'] for p in paths]) @ replay.WEIGHTS
        first = [next((e for e in p['events'] if e['type']=='medium_scatter'),None) for p in paths]
        row = dict(label=label,pixel=xy,mean_y=float(ys.mean()),
            se_y=float(ys.std(ddof=1)/np.sqrt(SAMPLES)),
            mean_gpis_scatters=float(orders.mean()),
            multiple_gpis_scatter_fraction=float((orders>=2).mean()),
            no_gpis_scatter_fraction=float((orders==0).mean()),
            terminations=dict(Counter(p['termination'] for p in paths)))
        by_pixel[label] = dict(row=row,first=first,paths=paths)
    return by_pixel


def main():
    # Reuse the observer harness; only its experiment inputs and output root change.
    replay.OUT=OUT
    replay.MODE=MODE
    replay.destination=lambda group,unused_stage: destination(group,'renewal_3d')
    replay.PIXELS=PIXELS
    replay.SAMPLES=SAMPLES
    replay.STEPS=STEPS
    jobs=[(g,m,s) for g in GROUPS for m in (GLOBAL,MODE) for s in STEPS]
    save(OUT/'plan.json',dict(groups=GROUPS,modes=[GLOBAL,MODE],steps=STEPS,
        samples=SAMPLES,seed=SEED,pixels=[dict(label=l,xy=xy) for l,xy in PIXELS],
        selection='Geometric base pixels fixed before replay; one center control.',
        harness_sha256=sha(Path(replay.__file__))))
    with ThreadPoolExecutor(max_workers=4) as pool:
        data=dict(pool.map(replay.run,jobs))
    stats={job:trace_statistics(*job) for job in jobs}
    regions={'base_selected':list(range(7)),'left_selected':[0,1,2],
        'right_selected':[3,4,5],'base_center':[6],'center_control':[7]}
    comparisons=[]
    for g in GROUPS:
        for name,ids in regions.items():
            basegap=data[g,MODE,.01]['y'][ids]-data[g,GLOBAL,.01]['y'][ids]
            refinedgap=data[g,MODE,.0025]['y'][ids]-data[g,GLOBAL,.0025]['y'][ids]
            comparisons.append(dict(group=g,region=name,
                baseline_gap=replay.estimate(basegap),
                refined_gap=replay.estimate(refinedgap),
                change_in_gap=replay.estimate(refinedgap-basegap),
                renewal_step_change=replay.estimate(data[g,MODE,.0025]['y'][ids]-data[g,MODE,.01]['y'][ids]),
                global_step_change=replay.estimate(data[g,GLOBAL,.0025]['y'][ids]-data[g,GLOBAL,.01]['y'][ids])))
    first_checks=[]
    for g in GROUPS:
        for step in STEPS:
            for label,xy in PIXELS:
                a=stats[g,GLOBAL,step][label]
                b=stats[g,MODE,step][label]
                first_checks.append(dict(group=g,step=step,label=label,
                    exact_first_scatter_matches=sum(x==y for x,y in zip(a['first'],b['first'])),
                    both_have_first_scatter=sum(x is not None and y is not None for x,y in zip(a['first'],b['first'])),
                    matched_present_first_scatter=sum(x is not None and x==y for x,y in zip(a['first'],b['first'])),
                    both_have_no_gpis_scatter=sum(x is None and y is None for x,y in zip(a['first'],b['first'])),
                    samples=SAMPLES,
                    identical_radiance_samples=sum(x['radiance']==y['radiance'] for x,y in zip(a['paths'],b['paths']))))
    summary=dict(complete=True,plan=load(OUT/'plan.json'),
        statistics=[dict(group=g,mode=m,step=s,**r['row']) for (g,m,s),items in stats.items() for r in items.values()],
        comparisons=comparisons,first_scatter_checks=first_checks,
        validations=[dict(group=g,mode=m,step=s,**v['validation']) for (g,m,s),v in data.items()],
        limits='Selected pixels only; SE estimates within-pixel Monte Carlo uncertainty. Step refinement changes geometry, RNG consumption and later paths. Counts are GPIS scatters on the continuation path, not a radiance decomposition by contributing path length.')
    save(OUT/'summary.json',summary)
    for c in comparisons:
        if c['region'] in ('base_selected','center_control'):
            print(c,flush=True)
    print('Exact first-scatter matches',sum(c['exact_first_scatter_matches'] for c in first_checks),
          '/',sum(c['samples'] for c in first_checks),flush=True)


if __name__=='__main__':
    main()
