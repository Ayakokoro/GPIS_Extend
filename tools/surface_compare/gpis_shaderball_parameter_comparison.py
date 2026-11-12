"""Matched four-mode shaderball sweep with frozen runtime and native FLIP.

Use the existing plot Python. Commands: prepare, pilot, run, report, status.
The run command resumes validated completions and updates comparison.html.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import html
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import traceback

import numpy as np
from figures import Image
from PIL import ImageDraw, ImageFont
from run import ROOT, load, pfm, sha
from gpis_shaderball_flip import DEFAULT_FLIP, evaluate, verify_links

DEFAULT_OUT = ROOT/'outputs/gpis_shaderball_parameter_comparison_20261111'
BASE = ROOT/'outputs/gpis_shaderball_compare/ell0p2_sigma0p02/final'
SEED = 20261024
GROUPS = ((.2, .02),) + tuple(itertools.product((.05, .2, .4), (.01, .04, .08)))
MODES = ('global_conditional', 'classic_global', 'gpis_renewal_plus', 'gpis_global')
LABELS = dict(global_conditional='global_conditional', classic_global='classic_global',
              gpis_renewal_plus='GPIS renewal+ (1D MIS)', gpis_global='GPIS global (3D)')
PAIRS = tuple((m, r) for r in ('gpis_global', 'gpis_renewal_plus')
              for m in MODES[:2]) + (('gpis_renewal_plus', 'gpis_global'),
                                     ('global_conditional', 'classic_global'))
BOUNDS = (44, 164, 213, 216)
Y = np.array([.2126, .7152, .0722])
FILES = ('scene.json', 'render.png', 'render.pfm', 'samplesA.pfm',
         'samplesB.pfm', 'samplesVariance.pfm', 'render.log')
ERRORS = re.compile(r'Caught an internal error|unrecoverable error|Unable to load|'
    r'Gradient invalid|Wrong normal direction|Failed to sample gradient|Gradient zero|'
    r'Conditioning .* error|sanity check failed|GPContext does not exist|'
    r'Clamp one (?:inf|nan) sample|terminate called', re.I)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def key(ell, sigma):
    tag = lambda v: format(v, '.7g').replace('.', 'p')
    return f'ell{tag(ell)}_sigma{tag(sigma)}'


def folder(args, ell, sigma, mode, pilot=False):
    return args.output/('pilot' if pilot else 'renders')/key(ell, sigma)/mode/f'seed{SEED}'


def scene_for(args, ell, sigma, mode, pilot=False):
    scene = load(args.output/'sources/classic_global.json')
    analytic = scene['media'][0]
    gp = copy.deepcopy(analytic['gaussian_process'])
    gp['covariance'] = dict(type='squared_exponential', lengthScale=ell, sigma=sigma)
    if mode in MODES[:2]:
        analytic.update(type=mode, gaussian_process=gp)
        if mode == 'global_conditional':
            analytic.update(non_gp_surface_policy='retain_endpoint',
                fp_gradient=dict(enabled=False), tracking_step_scale=.5,
                tracking_age_scale=.25, tracking_max_expected_events=64.)
    else:
        medium = load(args.output/'sources/gpis_global.json')['media'][0]
        medium.update(gaussian_process=gp, step_size=.0025, min_step=8, seed=17429,
            impulse_density=10, single_realization=False, isotropic_3D_sampling=False,
            correlation_context='global' if mode == 'gpis_global' else 'renewal+',
            intersect_method='gp_discrete', normal_method='conditioned_gaussian',
            surf_vol_phase_separate=False)
        one_d = mode == 'gpis_renewal_plus'
        medium.update({'1D_sampling':one_d, '1D_sampling_scheme':'mis' if one_d else 'uni',
                       '1D_gradient_correlationXY':one_d})
        medium['phase_function']['bsdf'] = copy.deepcopy(analytic['material'])
        scene['media'] = [medium]
    size, spp = (32, 4) if pilot else (256, 128)
    scene['camera']['resolution'] = [size, size]
    scene['integrator'].update(use_cuda_gpis=False, independent_path_sampling=True, wavefront_gpis=False)
    scene['renderer'].update(output_directory=str(folder(args, ell, sigma, mode, pilot)),
        output_file='render.png', hdr_output_file='render.pfm', spp=spp, spp_step=4 if pilot else 8,
        resume_render_file='resume.dat', enable_resume_render=not pilot, checkpoint_interval='300',
        adaptive_sampling=False, stratified_sampler=False, overwrite_output_files=True,
        timeout='0', output_buffers=[dict(type='color', hdr_output_file='samples.pfm',
            sample_variance=True, two_buffer_variance=True)])
    return scene


def validate_scenes(args):
    for ell, sigma in GROUPS:
        scenes = [scene_for(args, ell, sigma, mode) for mode in MODES]
        base = scenes[0]
        for mode, scene in zip(MODES, scenes):
            for section in ('bsdfs', 'primitives', 'camera', 'integrator'):
                assert scene[section] == base[section], (mode, section)
            a, b = copy.deepcopy(scene['renderer']), copy.deepcopy(base['renderer'])
            a.pop('output_directory'); b.pop('output_directory')
            assert a == b
            m = scene['media'][0]
            assert m['gaussian_process'] == base['media'][0]['gaussian_process']
            material = m['material'] if mode in MODES[:2] else m['phase_function']['bsdf']
            assert material == base['media'][0]['material']
            if mode in MODES[2:]:
                assert m['step_size'] == .0025 and m['impulse_density'] == 10
                assert m['1D_sampling'] == (mode == 'gpis_renewal_plus')
                assert m['type'] == 'sparse_conv_noise'
        a, b = (copy.deepcopy(s['media'][0]) for s in scenes[2:])
        for name in ('correlation_context', '1D_sampling', '1D_sampling_scheme', '1D_gradient_correlationXY'):
            a.pop(name); b.pop(name)
        assert a == b


def prepare(args):
    manifest_file = args.output/'manifest.json'
    if manifest_file.exists():
        manifest = load(manifest_file)
        assert manifest['groups'] == [list(g) for g in GROUPS]
        assert manifest['modes'] == list(MODES) and manifest['seed'] == SEED
        assert manifest['flip']['ppd'] == 67 and manifest['flip']['sha256'] == sha(args.flip)
        expected_runner = manifest.get('report_runner_sha256', manifest['runner_sha256'])
        assert expected_runner == sha(Path(__file__)), 'Use a new output after changing the runner'
        if 'report_runner_sha256' in manifest:
            assert sha(args.output/manifest['original_runner']) == manifest['runner_sha256']
        for name, digest in manifest['runtime_sha256'].items():
            assert sha(args.output/'renderer'/name) == digest
        for row in manifest['sources'].values():
            assert sha(args.output/row['snapshot']) == row['sha256']
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        runtime = args.output/'renderer'
        runtime.mkdir(exist_ok=True)
        hashes = {}
        for path in [ROOT/'build/Release/tungsten.exe', *sorted((ROOT/'build/Release').glob('*.dll'))]:
            shutil.copy2(path, runtime/path.name)
            hashes[path.name] = sha(runtime/path.name)
        sources = {}
        for mode in ('classic_global', 'gpis_global'):
            internal = 'sparse_3d_global' if mode == 'gpis_global' else mode
            source = BASE/internal/f'seed{SEED}'/'scene.json'
            result = load(source.parent/'result.json')
            assert sha(source) == result['sha256']['scene.json']
            target = args.output/'sources'/f'{mode}.json'
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(source, target)
            sources[mode] = dict(original=str(source), snapshot=target.relative_to(args.output).as_posix(),
                                 sha256=sha(target), original_result_sha256=sha(source.parent/'result.json'))
        manifest = dict(created_at=now(), groups=[list(g) for g in GROUPS], modes=list(MODES),
            resolution=[256, 256], spp=128, seed=SEED, gpis_step=.0025, impulse_density=10,
            sources=sources, runtime_sha256=hashes, runner_sha256=sha(Path(__file__)),
            flip=dict(executable=str(args.flip), sha256=sha(args.flip), ppd=67),
            bottom_bounds=list(BOUNDS), pairs=[list(p) for p in PAIRS],
            notes=['Original Renewal+ handling of ordinary surfaces; no retained-surface variant.',
                'global_conditional retains the endpoint value/gradient; fp_gradient is disabled.',
                'Renewal+: native 1D sparse convolution, MIS, transverse gradient correlation enabled.',
                'Global: world-coordinate 3D sparse convolution, one field per camera path.',
                'A 1D Global toggle does not preserve a fixed 3D field across ray direction changes.',
                'Full and cropped-input FLIP use the original filmic PNGs at 67 PPD.',
                'Finite-spp, finite-step approximate references; sampling noise contributes to FLIP.'])
        save(manifest_file, manifest)
    validate_scenes(args)
    for ell, sigma in GROUPS:
        for mode in MODES:
            path = folder(args, ell, sigma, mode)/'scene.json'
            scene = scene_for(args, ell, sigma, mode)
            if path.exists():
                assert load(path) == scene
            else:
                save(path, scene)
    return manifest


def verified(args, ell, sigma, mode, pilot=False):
    d = folder(args, ell, sigma, mode, pilot)
    row = load(d/'result.json')
    assert row['identity'] == dict(scene=scene_for(args, ell, sigma, mode, pilot), seed=SEED,
        renderer_sha256=load(args.output/'manifest.json')['runtime_sha256']['tungsten.exe'])
    for name, digest in row['sha256'].items():
        assert sha(d/name) == digest, str(d/name)
    return row


def status(args, active=None, failures=None, complete=False):
    rows = []
    manifest = load(args.output/'manifest.json')
    comparison_modes = manifest.get('comparison_modes', MODES)
    for ell, sigma in GROUPS:
        for mode in comparison_modes:
            d = folder(args, ell, sigma, mode)
            text = (d/'render.log').read_text(encoding='utf-8', errors='replace') if (d/'render.log').exists() else ''
            matches = re.findall(r'Completed (\d+)/(\d+) spp[^\r\n]*', text)
            warnings = load(d/'result.json').get('warnings', []) if (d/'result.json').exists() else []
            rows.append(dict(group=key(ell, sigma), mode=mode, warnings=warnings,
                state='complete' if (d/'result.json').exists() else ('running' if active == str(d) else 'pending'),
                spp=int(matches[-1][0]) if matches else 0,
                last_progress=matches[-1][0]+'/128 spp' if matches else '0/128 spp'))
    result = dict(updated_at=now(), complete=complete and all(r['state']=='complete' for r in rows),
                  completed=sum(r['state']=='complete' for r in rows),
                  total=len(rows), active=active, failures=failures or [], jobs=rows,
                  included_with_warnings=sum(bool(r['warnings']) for r in rows))
    save(args.output/'progress.json', result)
    return result


def render_one(args, ell, sigma, mode, pilot=False):
    d = folder(args, ell, sigma, mode, pilot)
    if (d/'result.json').exists():
        return verified(args, ell, sigma, mode, pilot)
    scene = scene_for(args, ell, sigma, mode, pilot)
    identity = dict(scene=scene, seed=SEED,
        renderer_sha256=load(args.output/'manifest.json')['runtime_sha256']['tungsten.exe'])
    if (d/'job.json').exists():
        assert load(d/'job.json')['identity'] == identity
    if (d/'render.log').exists():
        archive = d/'attempts'
        archive.mkdir(exist_ok=True)
        shutil.copy2(d/'render.log', archive/f'{time.time_ns()}.log')
    save(d/'scene.json', scene)
    command = [str(args.output/'renderer/tungsten.exe'), '-t', str(args.threads), '-s', str(SEED),
               '-d', str(d), str(d/'scene.json')]
    print('START', key(ell, sigma), mode, 'pilot' if pilot else '256x256 128spp', flush=True)
    started = time.monotonic()
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0) | getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0)
    with (d/'render.log').open('wb') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        save(d/'job.json', dict(identity=identity, process_id=process.pid, command=command, started_at=now()))
        try:
            while process.poll() is None:
                if not pilot:
                    status(args, active=str(d))
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            process.terminate()
            process.wait()
            raise
    log_text = (d/'render.log').read_text(encoding='utf-8', errors='replace')
    spp = scene['renderer']['spp']
    assert process.returncode == 0 and 'Finished render.' in log_text, str(d/'render.log')
    assert re.search(rf'Completed\s+{spp}/{spp}\s+spp', log_text), str(d/'render.log')
    assert not ERRORS.search(log_text), str(d/'render.log')
    assert sha(args.output/'renderer/tungsten.exe') == identity['renderer_sha256']
    rgb, a, b, var = (pfm(d/name) for name in ('render.pfm', 'samplesA.pfm', 'samplesB.pfm', 'samplesVariance.pfm'))
    size = scene['camera']['resolution'][0]
    assert rgb.shape == var.shape == (size, size, 3) and rgb.max() > 0
    assert np.allclose((a+b)/2, rgb, rtol=4e-6, atol=4e-7)
    with Image.open(d/'render.png') as im:
        assert im.size == (size, size)
    row = dict(identity=identity, seconds=time.monotonic()-started, finished_at=now(), command=command,
        resolution=[size, size], spp=spp, seed=SEED, threads=args.threads,
        mean_luminance=float((rgb@Y).mean()), sha256={name:sha(d/name) for name in FILES})
    save(d/'result.json', row)
    print('DONE', key(ell, sigma), mode, round(row['seconds'], 1), 'sec', flush=True)
    return verified(args, ell, sigma, mode, pilot)


def grid(paths, labels, target, bottom=False, columns=4):
    width = 320
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    measure = ImageDraw.Draw(Image.new('RGB', (1,1)))
    wrapped = []
    for label in labels:
        lines = []
        for paragraph in label.split('\n'):
            line = ''
            for word in paragraph.split():
                candidate = line+' '+word if line else word
                if line and measure.textlength(candidate, font=font) > width-20:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        wrapped.append('\n'.join(lines))
    header = max(58, max(measure.multiline_textbbox((0,0), label, font=font, spacing=4)[3]
                         for label in wrapped)+16)
    height = (180 if bottom else 356)+header-58
    rows = (len(paths)+columns-1)//columns
    canvas = Image.new('RGB', (columns*width, rows*height), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (path, label) in enumerate(zip(paths, wrapped)):
        x, y = i % columns*width, i//columns*height
        draw.multiline_text((x+10, y+7), label, fill='#253145', font=font, spacing=4)
        with Image.open(path) as im:
            im = im.convert('RGB')
            if bottom:
                im = im.crop(BOUNDS)
            im.thumbnail((300, height-header-2), Image.Resampling.NEAREST)
            canvas.paste(im, (x+(width-im.width)//2, y+header))
    canvas.save(target)


def render_duration(seconds):
    hours, remainder = divmod(int(seconds+.5), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f'{hours}h {minutes:02}m {secs:02}s'
    if minutes:
        return f'{minutes}m {secs:02}s'
    return f'{secs}s'


def report(args):
    manifest = load(args.output/'manifest.json')
    all_modes, pairs, labels = MODES, PAIRS, dict(LABELS)
    extra_verifier, extension_note = None, ''
    if manifest.get('fixed5_extension'):
        import gpis_shaderball_parameter_fixed5 as fixed5
        all_modes = MODES+(fixed5.MODE,)
        pairs = PAIRS+fixed5.PAIRS
        labels[fixed5.MODE] = fixed5.LABEL
        extra_verifier = fixed5.verified
    display_names = args.output/'display_names.json'
    if display_names.exists():
        aliases = load(display_names)
        assert set(aliases).issubset(all_modes)
        assert all(isinstance(value,str) and value.strip() for value in aliases.values())
        labels.update(aliases)
    if manifest.get('fixed5_extension'):
        extension_note = (f'<p><b>{html.escape(labels[fixed5.MODE])}</b>：开启梯度修正，'
            'bridge_correction=true，anchor_selection=fixed5_snr；'
            '与同参数未修正版本相比仅修改 fp_gradient 配置。</p>')
    expected_renders = len(GROUPS)*len(all_modes)
    expected_pairs = len(GROUPS)*len(pairs)
    display_pairs, display_columns = pairs, 3
    display_file = args.output/'display_comparisons.json'
    if display_file.exists():
        selection = load(display_file)
        display_pairs = tuple(tuple(pair) for pair in selection['pairs'])
        display_columns = selection.get('columns', 3)
        assert len(display_pairs) == len(set(display_pairs))
        assert set(display_pairs).issubset(pairs)
        assert isinstance(display_columns,int) and 1 <= display_columns <= 5
    expected_display_pairs = len(GROUPS)*len(display_pairs)
    displayed_comparisons = []
    sections, comparisons, sources, warning_sources, render_times = [], [], [], [], []
    for ell, sigma in GROUPS:
        tag = key(ell, sigma)
        available, render_warnings, group_times = {}, {}, {}
        for mode in all_modes:
            d = folder(args, ell, sigma, mode)
            if (d/'result.json').exists():
                row = (verified(args, ell, sigma, mode) if mode in MODES
                       else extra_verifier(args, ell, sigma))
                available[mode] = d
                log_text = (d/'render.log').read_text(encoding='utf-8', errors='replace')
                elapsed = re.findall(r'Completed\s+(\d+)/(\d+)\s+spp,\s*([\d.]+)/[\d.]+\s+sec', log_text)
                assert elapsed and elapsed[-1][:2] == (str(row['spp']), str(row['spp']))
                render_seconds = float(elapsed[-1][2])
                assert np.isfinite(render_seconds) and render_seconds > 0
                timing = dict(group=tag, ell=ell, sigma=sigma, mode=mode, label=labels[mode],
                    render_seconds=render_seconds, duration=render_duration(render_seconds),
                    recorded_seconds=row['seconds'],
                    recorded_timing_source=row.get('timing_source','runner_wall_clock'),
                    threads=row['threads'], log=(d/'render.log').relative_to(args.output).as_posix(),
                    result=(d/'result.json').relative_to(args.output).as_posix())
                group_times[mode] = timing
                render_times.append(timing)
                if row.get('warnings'):
                    render_warnings[mode] = row['warnings']
                    warning_sources.append(dict(group=tag, mode=mode, warnings=row['warnings']))
                sources.append(dict(group=tag, mode=mode, result=d.relative_to(args.output).as_posix()+'/result.json',
                    result_sha256=sha(d/'result.json'), seconds=row['seconds'],
                    warnings=row.get('warnings', [])))
        dest = args.output/'figures'/tag
        dest.mkdir(parents=True, exist_ok=True)
        heading = f'<h2 id="{tag}">ℓ={ell:g}，σ={sigma:g}，σ/ℓ={sigma/ell:g}</h2>'
        for mode, warnings in render_warnings.items():
            log_url = (available[mode]/'render.log').relative_to(args.output).as_posix()
            messages = '<br>'.join(html.escape(w['message']) for w in warnings)
            heading += (f'<div class="warning"><b>{labels[mode]}：含 NEE 一致性告警</b>。'
                f'该图已完成 128 spp，按用户要求纳入渲染与 FLIP 比较。'
                f'<br><code>{messages}</code><br><a href="{log_url}">原始渲染日志</a></div>')
        images = ''
        if available:
            modes = [m for m in all_modes if m in available]
            for bottom in (False, True):
                name = 'appearance_bottom.png' if bottom else 'appearance.png'
                grid([available[m]/'render.png' for m in modes],
                    [labels[m]+(' [NEE warning]' if m in render_warnings else '')+
                     f'\nRender: {group_times[m]["duration"]}' for m in modes],
                    dest/name, bottom, columns=len(all_modes))
                url = (dest/name).relative_to(args.output).as_posix()
                images += f'<a href="{url}"><img loading="lazy" src="{url}"></a>'
        maps, map_labels, table, group_comparisons = [], [], '', {}
        for test, reference in pairs:
            if test not in available or reference not in available:
                continue
            fd = args.output/'flip'/tag/f'{test}_vs_{reference}'
            full = evaluate(args.flip, available[reference]/'render.png', available[test]/'render.png', fd/'full', 67.)
            inputs = args.output/'inputs'/tag
            inputs.mkdir(parents=True, exist_ok=True)
            for mode in (test, reference):
                with Image.open(available[mode]/'render.png') as im:
                    im.crop(BOUNDS).save(inputs/f'{mode}_bottom.png')
            bottom = evaluate(args.flip, inputs/f'{reference}_bottom.png', inputs/f'{test}_bottom.png', fd/'bottom_input', 67.)
            rgb, ref = pfm(available[test]/'render.pfm'), pfm(available[reference]/'render.pfm')
            stats = dict(group=tag, ell=ell, sigma=sigma, test=test, reference=reference,
                full_flip=full['statistics'], bottom_input_flip=bottom['statistics'],
                linear_rgb_rmse=float(np.sqrt(np.mean((rgb-ref)**2))),
                map=(fd/'full'/full['error_map']).relative_to(args.output).as_posix(),
                native_record=(fd/'full/result.json').relative_to(args.output).as_posix(),
                bottom_record=(fd/'bottom_input/result.json').relative_to(args.output).as_posix())
            affected = [mode for mode in (test, reference) if mode in render_warnings]
            if affected:
                stats['warning_sources'] = affected
            comparisons.append(stats)
            group_comparisons[(test,reference)] = stats
        for test, reference in display_pairs:
            if (test,reference) not in group_comparisons:
                continue
            stats = group_comparisons[(test,reference)]
            displayed_comparisons.append(stats)
            maps.append(args.output/stats['map'])
            marker = ' *' if stats.get('warning_sources') else ''
            map_labels.append(f'{labels[test]}\nvs {labels[reference]}\nFLIP {stats["full_flip"]["mean"]:.6f}{marker}')
            test_label = html.escape(labels[test])+('（含告警图）' if test in render_warnings else '')
            reference_label = html.escape(labels[reference])+('（含告警图）' if reference in render_warnings else '')
            table += f'<tr><td>{test_label}</td><td>{reference_label}</td><td>{stats["full_flip"]["mean"]:.6f}</td><td>{stats["bottom_input_flip"]["mean"]:.6f}</td><td><a href="{stats["map"]}">误差图</a> · <a href="{stats["native_record"]}">原生统计</a></td></tr>'
        if maps:
            for bottom in (False, True):
                name = 'flip_bottom.png' if bottom else 'flip.png'
                grid(maps, map_labels, dest/name, bottom, display_columns)
                url = (dest/name).relative_to(args.output).as_posix()
                images += f'<a href="{url}"><img loading="lazy" src="{url}"></a>'
        links = ' · '.join(f'<a href="{d.relative_to(args.output).as_posix()}/scene.json">{html.escape(labels[mode])} 参数</a>' for mode, d in available.items())
        sections.append(heading+f'<p>已完成 {len(available)}/{len(all_modes)}；{links}</p>'+images+
            '<table><tr><th>测试</th><th>参考</th><th>全图 FLIP</th><th>底部输入 FLIP</th><th>文件</th></tr>'+table+'</table>')
    done = len(sources)
    save(args.output/'metrics.json', dict(complete=done==expected_renders,
        completed_renders=done, total_renders=expected_renders,
        sources=sources, comparisons=comparisons, included_with_warnings=warning_sources,
        displayed_comparison_pairs=len(displayed_comparisons), display_pairs=[list(p) for p in display_pairs],
        updated_at=now(), manifest_sha256=sha(args.output/'manifest.json')))
    import csv
    save(args.output/'render_times.json', dict(
        timing_source='Final Completed spp progress line in each original render.log; renderer elapsed wall time in seconds.',
        rows=render_times))
    with (args.output/'render_times.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=('group','ell','sigma','mode','label',
            'render_seconds','duration','recorded_seconds','recorded_timing_source','threads','log','result'))
        writer.writeheader()
        writer.writerows(render_times)
    with (args.output/'metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['group', 'ell', 'sigma', 'test', 'reference', 'mean_flip', 'mean_flip_bottom_input', 'linear_rgb_rmse'])
        for row in comparisons:
            writer.writerow([row[n] for n in ('group','ell','sigma','test','reference')]+[
                row['full_flip']['mean'], row['bottom_input_flip']['mean'], row['linear_rgb_rmse']])
    nav = ' · '.join(f'<a href="#{key(e,s)}">ℓ={e:g}, σ={s:g}</a>' for e,s in GROUPS)
    warning_link = '<a href="warnings.json">告警与纳入记录</a>' if (args.output/'warnings.json').exists() else ''
    display_json = json.dumps(labels, ensure_ascii=True).replace('<', '\\u003c')
    time_lookup = {(row['group'],row['mode']):row for row in render_times}
    time_header = '<th>参数组</th>'+''.join(f'<th>{html.escape(labels[m])}</th>' for m in all_modes)
    time_rows = ''
    for ell, sigma in GROUPS:
        tag = key(ell,sigma)
        cells = ''
        for mode in all_modes:
            timing = time_lookup.get((tag,mode))
            cells += (f'<td><a href="{timing["log"]}">{timing["duration"]}</a>'
                      f'<br><small>{timing["render_seconds"]:.2f} s · {timing["threads"]} 线程</small></td>'
                      if timing else '<td>待完成</td>')
        time_rows += f'<tr><td><a href="#{tag}">ℓ={ell:g}，σ={sigma:g}</a></td>{cells}</tr>'
    timing_section = (f'<h2 id="render-times">渲染时间</h2>'
        '<p>采用各图 render.log 最后一条 Completed spp 记录的累计渲染秒数；'
        '点击耗时打开原始日志。h、m、s 分别表示小时、分钟、秒。'
        '<a href="render_times.csv">耗时 CSV</a> · <a href="render_times.json">精确耗时记录</a></p>'
        f'<div style="overflow-x:auto"><table id="render-times-table"><tr>{time_header}</tr>{time_rows}</table></div>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shaderball 多参数模式比较</title><style>body{max-width:1500px;margin:30px auto;padding:0 20px;font:16px/1.6 system-ui;color:#253145;background:#f5f6f8}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;background:white}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}h2{margin-top:45px}a{color:#235e91}.note{padding:18px;background:#e7edf5;border-radius:8px}.warning{padding:14px;background:#fff0d2;border-left:4px solid #b27612;margin:12px 0}pre{white-space:pre-wrap}</style>
<h1>Shaderball：多参数模式比较</h1>'''+f'''<div class="note">256×256 · 128 spp · seed {SEED} · {len(GROUPS)} 组参数。已完成 <b>{done}/{expected_renders}</b> 张；FLIP 展示 <b>{len(displayed_comparisons)}/{expected_display_pairs}</b> 个配对。<br>
两种 GPIS 步长均为 0.0025、impulse density=10。Renewal+：1D sparse convolution + MIS + 横向梯度相关；Global：世界坐标 3D sparse convolution。<br>
{html.escape(labels['global_conditional'])} 使用 retain_endpoint，关闭 fp_gradient；Renewal+ 使用原版非 GPIS 表面流程。</div>
{extension_note}
{timing_section}
<p>每组展示 {len(display_pairs)} 个 FLIP 配对：各方法分别与 GPIS Global、GPIS Renewal+ 比较，并保留两个 GPIS 的比较。</p>
<p>含告警但按用户要求纳入比较：<b>{len(warning_sources)}</b> 张。对应参数组列出原始告警；渲染图标为 [NEE warning]，相关 FLIP 面板标为 *。{warning_link}</p>
<p>同相机、地面、光照、材质、最大反弹次数。统一 filmic 显示；本地 FLIP 1.7，67 PPD，误差色标固定 0–1。参考图仍含 128 spp 采样噪声与有限步长误差。</p>
<p>原图下方为底部放大；FLIP 放大图裁自全图误差图，底部数值则由独立裁剪输入计算（范围 {BOUNDS}）。</p>
<p><a href="manifest.json">完整配置与来源</a> · <a href="metrics.json">完整指标</a> · <a href="metrics.csv">CSV</a> · <a href="progress.json">实时进度</a></p>
<script id="display-labels" type="application/json">{display_json}</script>
<pre id="progress">读取实时进度…</pre><nav>{nav}</nav>'''+''.join(sections)+r'''
<script>const modeLabels=JSON.parse(document.getElementById('display-labels').textContent);async function update(){try{const p=await(await fetch('progress.json?t='+Date.now())).json();document.getElementById('progress').textContent='完成 '+p.completed+'/'+p.total+'；'+(p.active?'正在渲染':'空闲')+'\n'+p.jobs.filter(x=>x.state==='running').map(x=>x.group+' '+(modeLabels[x.mode]||x.mode)+' '+x.last_progress).join('\n');}catch(e){document.getElementById('progress').textContent='刷新页面查看新结果；实时 JSON 可从“实时进度”链接查看。';}}update();setInterval(update,15000);</script></html>'''
    temporary = args.output/'comparison.html.tmp'
    temporary.write_text(page, encoding='utf-8')
    temporary.replace(args.output/'comparison.html')
    count = verify_links(args.output/'comparison.html')
    save(args.output/'verification.json', dict(passed=True, complete=done==expected_renders,
        completed_renders=done, comparison_pairs=len(comparisons), local_links=count,
        displayed_comparison_pairs=len(displayed_comparisons), displayed_pairs_per_group=len(display_pairs),
        warning_free=not warning_sources, included_with_warnings=len(warning_sources),
        scene_parameters_matched=True, updated_at=now(), metrics_sha256=sha(args.output/'metrics.json'),
        render_time_records=len(render_times), render_times_sha256=sha(args.output/'render_times.json'),
        render_times_csv_sha256=sha(args.output/'render_times.csv'),
        html_sha256=sha(args.output/'comparison.html')))
    print('REPORT', done, '/', expected_renders, 'renders;', len(comparisons), '/', expected_pairs,
          'computed pairs;', len(displayed_comparisons), '/', expected_display_pairs, 'displayed pairs', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'pilot', 'run', 'report', 'status'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--flip', type=Path, default=DEFAULT_FLIP)
    args = parser.parse_args()
    args.output = args.output.resolve()
    assert args.threads > 0 and args.flip.is_file()
    if args.command == 'status':
        print(json.dumps(load(args.output/'progress.json'), ensure_ascii=False, indent=2))
        return
    prepare(args)
    if not (args.output/'progress.json').exists():
        status(args)
    if args.command in ('prepare', 'report'):
        report(args)
        return
    lock = args.output/'runner.lock'
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        raise RuntimeError(f'Queue lock exists: {lock}; check the recorded PID before removing a stale lock')
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(dict(pid=os.getpid(), started_at=now(), command=args.command), stream)
    try:
        pilot_rows = []
        for ell, sigma in ((.2, .02), (.05, .08), (.4, .01)):
            for mode in MODES:
                pilot_rows.append(dict(ell=ell, sigma=sigma, mode=mode,
                    result=render_one(args, ell, sigma, mode, True)))
        save(args.output/'pilot_validation.json', dict(passed=True, sources=pilot_rows,
            scope='32x32, 4 spp: baseline and two extreme sigma/ell ratios; configuration and finite-output checks, not convergence.'))
        if args.command == 'pilot':
            return
        failures = []
        report(args)
        # Complete fast analytic comparisons and 1D Renewal+ before long 3D references.
        for mode in MODES:
            for ell, sigma in GROUPS:
                try:
                    render_one(args, ell, sigma, mode)
                except Exception as exc:
                    failure = dict(group=key(ell,sigma), mode=mode, error=str(exc), traceback=traceback.format_exc())
                    failures.append(failure)
                    save(args.output/'failures.json', failures)
                    print('FAILED', failure, flush=True)
                status(args, failures=failures)
                report(args)
        complete = not failures
        status(args, failures=failures, complete=complete)
        report(args)
        if failures:
            raise RuntimeError(f'{len(failures)} renders failed; see failures.json')
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
