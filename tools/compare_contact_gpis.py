#!/usr/bin/env python3
"""Compare matched contact scenes with sparse-convolution GPIS ensemble renders.

The 3D/global mode retains one world-space noise realization along each path.
The 1D/renewal+ mode resamples each segment and keeps the last contact only.
All inputs and generated artifacts default to this project's outputs directory.
"""
from __future__ import annotations
import argparse, array, copy, csv, html, json, math, os, re, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ('classic_local', 'classic_global', 'global_conditional', 'neural',
         'sparse_3d_global', 'sparse_1d_renewal')
COLORS = dict(zip(MODES, ('#387dc4', '#606b7c', '#da7628', '#288c62', '#a337a3', '#bd4051')))
ERRORS = re.compile(r'Caught an internal error|unrecoverable error|Unable to load|Wrong normal direction|terminate called|Gradient invalid|Conditioning .* error', re.I)

def load(p):
    return json.loads(p.read_text(encoding='utf-8'))

def save(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def path_arg(value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else ROOT/p).resolve()

def pfm(p):
    with p.open('rb') as f:
        if f.readline().strip() != b'PF':
            raise ValueError(f'Expected RGB PFM: {p}')
        w, h = map(int, f.readline().split())
        scale = float(f.readline()); rgb = array.array('f'); rgb.frombytes(f.read())
    if (scale < 0) != (sys.byteorder == 'little'): rgb.byteswap()
    if len(rgb) != w*h*3 or not math.isfinite(scale) or scale == 0:
        raise ValueError(f'Invalid PFM payload: {p}')
    if any(not math.isfinite(x) or x < -1e-6 for x in rgb):
        raise ValueError(f'Invalid radiance: {p}')
    pixels = [(.2126*rgb[i] + .7152*rgb[i+1] + .0722*rgb[i+2])*abs(scale) for i in range(0,len(rgb),3)]
    if max(pixels) <= 0: raise ValueError(f'Black image: {p}')
    return w, h, pixels

def measure(w, h, pixels):
    roi = [pixels[y*w+x] for y in range(h//4,3*h//4) for x in range(w//3,2*w//3)]
    return dict(width=w, height=h, mean=statistics.fmean(pixels), roi_mean=statistics.fmean(roi), maximum=max(pixels))

def make_scene(mode, args, folder):
    template = 'global_conditional' if mode.startswith('sparse_') else mode
    scene = load(args.base/'configs'/f'{template}.json')
    if mode.startswith('sparse_'):
        m = scene['media'][0]; material = m.pop('material')
        one_d = mode == 'sparse_1d_renewal'
        m.update(type='sparse_conv_noise', phase_function={'type':'brdf','bsdf':material},
                 sigma_a=0, sigma_s=1, density=1, step_size=args.step_size, min_step=8,
                 seed=17429, impulse_density=args.impulse_density, single_realization=False,
                 correlation_context='renewal+' if one_d else 'global',
                 intersect_method='gp_discrete', normal_method='conditioned_gaussian')
        m.update({'1D_sampling':one_d, '1D_sampling_scheme':'uni',
                  '1D_gradient_correlationXY':False, 'isotropic_3D_sampling':False})
    scene['camera']['resolution'] = [args.size,args.size]
    scene['renderer'].update(spp=args.spp, spp_step=max(1,args.spp//8), output_directory=folder.as_posix())
    return scene

def matching(scene, old):
    a,b = copy.deepcopy(scene),copy.deepcopy(old)
    for s in (a,b):
        for key in ('output_directory','spp_step'): s['renderer'].pop(key,None)
    return a==b

def verified(folder, scene, spp):
    try:
        log=(folder/'render.log').read_text(encoding='utf-8',errors='replace')
        return (matching(scene,load(folder/'scene.json')) and
                (folder/'render.png').is_file() and (folder/'render.pfm').is_file() and
                bool(re.search(rf'Completed\s+{spp}/{spp}\s+spp',log)) and
                'Finished render.' in log and not ERRORS.search(log))
    except (OSError,ValueError): return False

def run(mode, seed, args, run_dir):
    folder=run_dir/mode/f'seed{seed}'
    scene=make_scene(mode,args,folder)
    # Reuse only a completed render with an identical scene, resolution, spp and seed.
    candidates=[folder]
    candidates += [args.base/'selected/ell0p1'/mode/f'seed{seed}',
                   args.base/'seed_check/ell0p1'/mode/f'seed{seed}']
    reused=next((p for p in candidates if verified(p,scene,args.spp)),None)
    elapsed=None
    if reused:
        folder=reused;scene=load(folder/'scene.json')
        if (folder/'result.json').exists(): elapsed=load(folder/'result.json').get('seconds')
        print(f'Reuse {mode} seed {seed}: {folder}',flush=True)
    else:
        if not args.render:
            # Never relabel an old image by overwriting the scene it was rendered with.
            save(run_dir/'configs'/f'{mode}_seed{seed}.json',scene)
            return None
        save(folder/'scene.json',scene)
        if args.render:
            print(f'Render {mode} seed {seed}',flush=True);start=time.monotonic()
            command=[str(args.tungsten),'-t',str(args.threads),'-s',str(seed),'-d',str(folder),str(folder/'scene.json')]
            with (folder/'render.log').open('w',encoding='utf-8') as log:
                result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode or not verified(folder,scene,args.spp):
                raise RuntimeError(f'Render failed; inspect {folder / "render.log"}')
            elapsed=time.monotonic()-start
    save(run_dir/'configs'/f'{mode}_seed{seed}.json',scene)
    if not verified(folder,scene,args.spp): return None
    w,h,pixels=pfm(folder/'render.pfm')
    if (w,h)!=(args.size,args.size): raise ValueError('Unexpected rendered dimensions')
    row=dict(mode=mode,seed=seed,spp=args.spp,seconds=elapsed,scene=str(folder/'scene.json'),
             **measure(w,h,pixels))
    if not reused: save(folder/'result.json',row)
    return row

def relative(p, base): return Path(os.path.relpath(p,base)).as_posix()

def report(rows, args, run_dir):
    groups={m:[r for r in rows if r['mode']==m] for m in args.modes}
    groups={m:rs for m,rs in groups.items() if rs}
    if not groups: return
    profiles={};stats={};cards=[];scenes=[]
    for mode, rs in groups.items():
        images=[]
        for r in rs:
            scene=load(Path(r['scene']));scenes.append(scene)
            w,h,pixels=pfm(Path(r['scene']).parent/'render.pfm');images.append(pixels)
        average=[statistics.fmean(values) for values in zip(*images)]
        profiles[mode]=[statistics.fmean(average[y*w+x] for y in range(h//4,3*h//4)) for x in range(w)]
        s=measure(w,h,average);s.update(seeds=[r['seed'] for r in rs],spp_per_seed=args.spp,
                                     roi_seed_values=[r['roi_mean'] for r in rs],
                                     roi_seed_stdev=statistics.stdev(r['roi_mean'] for r in rs) if len(rs)>1 else None)
        stats[mode]=s
        r=rs[0];folder=Path(r['scene']).parent;link=relative(folder,run_dir)
        cards.append(f'<article><h3 style="color:{COLORS[mode]}">{mode}</h3>'
          f'<a href="{link}/render.png"><img src="{link}/render.png"></a>'
          f'<p>Image: seed {r["seed"]}, {args.spp} spp<br>'
          f'<a href="configs/{mode}_seed{r["seed"]}.json">Scene JSON</a> / '
          f'<a href="{link}/render.pfm">Linear PFM</a> / <a href="{link}/render.log">Log</a></p></article>')
    for s in scenes[1:]:
        for key in ('camera','primitives','integrator'):
            if s[key]!=scenes[0][key]: raise ValueError(f'Unmatched {key}')
    for s in scenes:
        m=s['media'][0]
        material=m.get('material',m.get('phase_function',{}).get('bsdf'))
        if material != {'type':'mirror','albedo':.85}: raise ValueError('Material mismatch')
        cov=m.get('covariance',m.get('gaussian_process',{}).get('covariance'))
        if cov != {'type':'squared_exponential','sigma':.02,'lengthScale':.1}: raise ValueError('Covariance mismatch')
    ref=stats.get('sparse_3d_global')
    if ref:
        for mode,s in stats.items():
            s['roi_relative_to_sparse_3d']=s['roi_mean']/ref['roi_mean']-1
            s['image_mean_relative_to_sparse_3d']=s['mean']/ref['mean']-1
            # Eight horizontal bins reduce pixel noise; residual Monte Carlo variance remains.
            binned=lambda p:[statistics.fmean(p[i*w//8:(i+1)*w//8]) for i in range(8)]
            a,b=binned(profiles[mode]),binned(profiles['sparse_3d_global'])
            s['binned_profile_relative_rmse']=math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/sum(y*y for y in b))
    summary=dict(resolution=[args.size,args.size],spp_per_seed=args.spp,rows=rows,statistics=stats,
                 impulse_density=args.impulse_density,step_size=args.step_size,
                 roi={'x_begin':w//3,'x_end':2*w//3,'y_begin':h//4,'y_end':3*h//4},
                 shared_scene_verified=True,
                 reference='3D world-space sparse convolution, global memory, independent realization per camera path; finite-density approximation, not an exact GP.',
                 renewal='1D renewal+ with correlationXY=false; last-contact approximation, not full path memory.',
                 metrics='Linear luminance. Seed standard deviation is run-to-run spread, not a confidence interval. Profile error still includes Monte Carlo noise.')
    save(run_dir/'summary.json',summary)
    with (run_dir/'horizontal_profile.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['column',*groups])
        for i in range(w): writer.writerow([i,*[profiles[m][i] for m in groups]])
    ymax=max(max(v) for v in profiles.values())*1.1
    svg=['<svg viewBox="0 0 860 320" aria-label="Linear horizontal radiance profile">']
    for i in range(5):
        y=270-240*i/4
        svg.append(f'<path d="M60,{y} H840" stroke="#ddd"/><text x="52" y="{y+4}" text-anchor="end">{ymax*i/4:.2f}</text>')
    for m,p in profiles.items():
        points=' '.join(f'{60+780*(i+.5)/w:.2f},{270-240*v/ymax:.2f}' for i,v in enumerate(p))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{COLORS[m]}" stroke-width="2"/>')
    svg.append('<text x="450" y="310" text-anchor="middle">Horizontal image position; groove bottom at center</text></svg>')
    table=[]
    for m,s in stats.items():
        delta=f'{100*s["roi_relative_to_sparse_3d"]:+.1f}%' if ref else 'n/a'
        err=f'{100*s["binned_profile_relative_rmse"]:.1f}%' if ref else 'n/a'
        spread=f'{s["roi_seed_stdev"]:.5f}' if s['roi_seed_stdev'] is not None else 'one seed'
        table.append(f'<tr><td>{m}</td><td>{s["roi_mean"]:.5f}</td><td>{delta}</td><td>{s["mean"]:.5f}</td><td>{err}</td><td>{spread}</td></tr>')
    legend=' '.join(f'<span style="color:{COLORS[m]};margin-right:14px">&#9632; {m}</span>' for m in groups)
    page='''<!doctype html><meta charset="utf-8"><title>Sparse convolution GPIS comparison</title>
<style>body{font:15px/1.6 system-ui;max-width:1500px;margin:28px;background:#f5f7fa;color:#263345}.cards{display:grid;grid-template-columns:repeat(3,minmax(200px,1fr));gap:18px}article,section{background:white;padding:18px;border:1px solid #dde2e8;border-radius:8px;margin-bottom:18px}img{width:100%;aspect-ratio:1}td,th{padding:8px 12px;border:1px solid #d5dbe3;text-align:left}table{border-collapse:collapse}a{color:#175fa5}svg{width:100%;max-width:1000px;font:12px system-ui}pre{white-space:pre-wrap}@media(max-width:900px){.cards{grid-template-columns:repeat(2,1fr)}}</style>
<h1>Sparse convolution GPIS / conditional / neural</h1>
<p>90-degree V groove; sigma=0.02; lengthScale=0.1. Identical mean geometry, camera, light, mirror albedo 0.85 and bounce cap 128. All images use the same filmic mapping.</p>
<p><b>sparse_3d_global</b>: 3D world-space noise; the same realization is retained for every bounce of one camera path, and different camera samples draw different realizations. <b>sparse_1d_renewal</b>: 1D renewal+, last-contact conditioning, correlationXY=false. Neither finite-density reference is an exact Gaussian process.</p>'''
    page+=f'<p>{args.size} x {args.size}; {args.spp} spp per seed; requested seeds: {html.escape(str(args.seeds))}. Sparse impulse_density={args.impulse_density}; marching step={args.step_size}.</p>'
    page+='<div class="cards">'+''.join(cards)+'</div><section><h2>Linear radiance measurements</h2>'
    page+='<table><tr><th>Mode</th><th>Groove ROI mean</th><th>vs 3D sparse</th><th>Image mean</th><th>8-bin profile relative RMSE</th><th>ROI seed stdev</th></tr>'+''.join(table)+'</table>'
    page+='<p>The ROI is the middle third of width and middle half of height, fixed before this comparison. Statistics and curves average all listed seeds; displayed PNGs show the first seed. Profile errors still include sampling noise. Seed standard deviation is not a confidence interval. <a href="summary.json">Full JSON</a> / <a href="horizontal_profile.csv">Profile CSV</a></p></section>'
    page+='<section><h2>Horizontal brightness profile</h2><p>'+legend+'</p>'+''.join(svg)+'</section>'
    if args.name == 'selected':
        evidence=[]
        validation=args.output/'validation/summary.json'
        if validation.exists():
            v=load(validation)
            for mode,info in v['statistics'].items():
                if 'roi_relative_to_sparse_3d' not in info: continue
                spread=info['roi_seed_stdev']
                sd=f"{spread:.5f}" if spread is not None else 'one seed'
                evidence.append(f"<tr><td>{mode}</td><td>{info['roi_mean']:.5f}</td><td>{100*info['roi_relative_to_sparse_3d']:+.1f}%</td><td>{sd}</td></tr>")
            dimensions=' x '.join(map(str,v['resolution']))
            seed_sets={tuple(info['seeds']) for info in v['statistics'].values()}
            seed_text='; '.join(str(list(seeds)) for seeds in sorted(seed_sets))
            page+='<section><h2>Independent-seed check</h2><p><a href="../validation/comparison.html">Open the multi-seed comparison</a>. '+dimensions+f"; {v['spp_per_seed']} spp per seed. Seed groups: {seed_text}. This check has its own ROI pixel bounds.</p><table><tr><th>Mode</th><th>ROI average</th><th>vs 3D sparse</th><th>Across-seed stdev</th></tr>"+''.join(evidence)+'</table></section>'
        controls=[]
        baseline_path=args.output/'step_baseline/summary.json'
        if baseline_path.exists():
            baseline=load(baseline_path)['statistics']['sparse_3d_global']['roi_mean']
            for name in ('step_baseline','step_check','density_check'):
                f=args.output/name/'summary.json'
                if not f.exists(): continue
                v=load(f);r=v['statistics']['sparse_3d_global']
                controls.append(f"<tr><td><a href='../{name}/comparison.html'>{name}</a></td><td>{v['step_size']}</td><td>{v['impulse_density']}</td><td>{r['roi_mean']:.5f}</td><td>{100*(r['roi_mean']/baseline-1):+.1f}%</td></tr>")
            page+='<section><h2>Reference parameter checks</h2><p>32 x 32, 128 spp, seed 20261023 for every row. Changing impulse density also changes the random field realization; a single pair is a sensitivity check, not a convergence proof.</p><table><tr><th>Check</th><th>Marching step</th><th>Impulse density</th><th>ROI mean</th><th>vs baseline</th></tr>'+''.join(controls)+'</table></section>'
    page+='<section><h2>Reproduce</h2><pre>python tools/compare_contact_gpis.py --render --name '+html.escape(args.name)+f' --size {args.size} --spp {args.spp} --seeds '+ ' '.join(map(str,args.seeds))+' --modes '+' '.join(args.modes)+f' --impulse-density {args.impulse_density} --step-size {args.step_size}</pre></section>'
    (run_dir/'comparison.html').write_text(page,encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False),flush=True)
    print('Report:',run_dir/'comparison.html',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=path_arg,default=ROOT/'outputs/contact_contrast')
    p.add_argument('--output',type=path_arg,default=ROOT/'outputs/contact_contrast/sparse_convolution')
    p.add_argument('--name',default='selected')
    p.add_argument('--tungsten',type=path_arg,default=ROOT/'build/Release/tungsten.exe')
    p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    p.add_argument('--size',type=int,default=64);p.add_argument('--spp',type=int,default=128)
    p.add_argument('--seeds',nargs='+',type=int,default=[20261023]);p.add_argument('--threads',type=int,default=8)
    p.add_argument('--step-size',type=float,default=.005);p.add_argument('--impulse-density',type=float,default=10)
    p.add_argument('--render',action='store_true',help='Render missing or changed configurations; otherwise generate configs and report completed renders.')
    a=p.parse_args()
    if a.size<8 or min(a.spp,a.threads)<1 or not all(math.isfinite(v) and v>0 for v in (a.step_size,a.impulse_density)):
        p.error('Image size must be at least 8; sample counts, thread count and sparse parameters must be positive.')
    a.seeds=list(dict.fromkeys(a.seeds));a.modes=list(dict.fromkeys(a.modes))
    run_dir=(a.output/a.name).resolve()
    if not run_dir.is_relative_to(a.output.resolve()) or run_dir==a.output.resolve(): p.error('--name must be a subdirectory of --output')
    run_dir.mkdir(parents=True,exist_ok=True);rows=[]
    for mode in a.modes:
        for seed in a.seeds:
            r=run(mode,seed,a,run_dir)
            if r: rows.append(r);save(run_dir/'render_rows.json',rows)
    report(rows,a,run_dir)
if __name__=='__main__':main()
