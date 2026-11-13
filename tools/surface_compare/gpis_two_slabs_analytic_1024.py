"""Render optimized native single history at 1024 spp and compare frozen references."""
from __future__ import annotations

import argparse
import copy
import csv
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import traceback
from types import SimpleNamespace

import numpy as np
from figures import plt
from run import ROOT, load, sha, pfm
from gpis_sphere_small_light import save, now, FATAL, FILES, Y
from gpis_shaderball_flip import DEFAULT_FLIP, verify_links
from gpis_two_slabs import roi_masks
from gpis_two_slabs_step_refinement import compare, picture_grid, url

BASE = ROOT / 'outputs/gpis_two_slabs_compare_20261113'
OUT = BASE / 'single_history_analytic_1024_comparison'
SEEDS = (20261024, 20261113)
MODE = 'single_history_gpis'
MODES = ('global_conditional', 'neural_continued', MODE, 'gpis_renewal_plus', 'gpis_global')
LABELS = dict(global_conditional='start_conditioning', neural_continued='Neural: continued training',
              single_history_gpis='single_history: analytic covariance',
              gpis_renewal_plus='GPIS renewal+ (1D MIS)', gpis_global='GPIS Global (3D)',
              previous='single_history: previous implementation')
BEGIN = '<!-- BEGIN TWO SLABS ANALYTIC SINGLE HISTORY 1024 -->'
END = '<!-- END TWO SLABS ANALYTIC SINGLE HISTORY 1024 -->'
ARGS = SimpleNamespace(output=OUT, base=BASE, flip=DEFAULT_FLIP)


def folder(mode, seed):
    suffix = Path('final') / (MODE if mode == 'previous' else mode) / f'seed{seed}'
    if mode == MODE:
        return OUT / suffix
    if mode == 'previous':
        return BASE / 'single_history_comparison' / suffix
    if mode == 'gpis_global':
        return BASE / 'gpis_global_comparison' / suffix
    if mode == 'neural_continued':
        return BASE / 'neural_single_history_continued/final/global_conditional_neural' / f'seed{seed}'
    return BASE / suffix


def verify_record(mode, seed):
    directory = folder(mode, seed)
    record = load(directory / 'result.json')
    assert record['seed'] == seed and record['spp'] == 1024 and record['resolution'] == [64, 64]
    assert not record['warnings']
    for name, digest in record['sha256'].items():
        assert sha(directory / name) == digest, str(directory / name)
    if mode == MODE:
        assert record['renderer_sha256'] == load(OUT / 'manifest.json')['runtime_sha256']['tungsten.exe']
    return record


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = OUT / 'renderer'
    runtime.mkdir(exist_ok=True)
    reference_hashes = {}
    for seed in SEEDS:
        for mode in (*[m for m in MODES if m != MODE], 'previous'):
            record = verify_record(mode, seed)
            directory = folder(mode, seed)
            reference_hashes[str(directory / 'result.json')] = sha(directory / 'result.json')
            reference_hashes.update({str(directory / name): digest for name, digest in record['sha256'].items()})
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        manifest = load(manifest_path)
        assert manifest['runner_sha256'] == sha(Path(__file__))
        assert manifest['reference_sha256'] == reference_hashes
        for name, digest in manifest['runtime_sha256'].items():
            assert sha(runtime / name) == digest
    else:
        for source in [ROOT / 'build/Release/tungsten.exe', *sorted((ROOT / 'build/Release').glob('*.dll'))]:
            shutil.copy2(source, runtime / source.name)
        source_dir = OUT / 'source'
        source_dir.mkdir(exist_ok=True)
        for source in [Path(__file__), ROOT / 'src/core/math/SparseHistoryGaussian.cpp',
                       ROOT / 'src/core/math/SparseHistoryGaussian.hpp']:
            shutil.copy2(source, source_dir / source.name)
        manifest = dict(created_at=now(), resolution=[64, 64], spp=1024, seeds=SEEDS, threads=16,
                        runner_sha256=sha(Path(__file__)),
                        runtime_sha256={p.name: sha(p) for p in runtime.iterdir()},
                        source_sha256={p.name: sha(p) for p in source_dir.iterdir()},
                        reference_sha256=reference_hashes,
                        flip=dict(path=str(DEFAULT_FLIP), sha256=sha(DEFAULT_FLIP), ppd=67),
                        notes=['Fresh optimized native single_history renders; references are hash-verified existing 1024 spp results.',
                               'Same scene and native single_history settings as previous experiment; only output directory changes.',
                               'Reference mode timings are historical and are not a matched performance benchmark.',
                               'Stable analytic covariance with original feature fallback; numerical probability integration unchanged.'])
        save(manifest_path, manifest)
    assert manifest['flip']['sha256'] == sha(DEFAULT_FLIP)
    for seed in SEEDS:
        scene = copy.deepcopy(load(folder('previous', seed) / 'scene.json'))
        scene['renderer']['output_directory'] = str(folder(MODE, seed))
        target = folder(MODE, seed) / 'scene.json'
        if target.exists():
            assert load(target) == scene
        else:
            save(target, scene)


def progress(active=None, failure=None):
    jobs = []
    for seed in SEEDS:
        directory = folder(MODE, seed)
        text = (directory / 'render.log').read_text(errors='replace') if (directory / 'render.log').exists() else ''
        matches = re.findall(r'Completed\s+(\d+)/1024\s+spp', text)
        jobs.append(dict(seed=seed, spp=int(matches[-1]) if matches else 0, target_spp=1024,
                         state='complete' if (directory / 'result.json').exists() else 'running' if seed == active else 'pending'))
    save(OUT / 'progress.json', dict(updated_at=now(), active_seed=active, jobs=jobs,
                                    complete=all(j['state'] == 'complete' for j in jobs), failure=failure))


def render(seed):
    directory = folder(MODE, seed)
    if (directory / 'result.json').exists():
        return verify_record(MODE, seed)
    assert not (directory / 'render.log').exists(), 'Inspect unfinished attempt before restarting'
    exe = OUT / 'renderer/tungsten.exe'
    command = [str(exe), '-t', '16', '-s', str(seed), '-d', str(directory), str(directory / 'scene.json')]
    print(f'START single_history seed={seed} 64x64 1024spp threads=16', flush=True)
    started = time.monotonic()
    with (directory / 'render.log').open('wb') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        save(directory / 'job.json', dict(pid=process.pid, command=command, started_at=now()))
        try:
            while process.poll() is None:
                progress(seed)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            process.terminate()
            process.wait()
            raise
    text = (directory / 'render.log').read_text(errors='replace')
    assert process.returncode == 0 and 'Finished render.' in text and not FATAL.search(text), str(directory)
    assert re.search(r'Completed\s+1024/1024\s+spp', text)
    warnings = [line for line in text.splitlines() if 'sanity check failed' in line]
    assert not warnings, warnings
    counters = json.loads(re.findall(r'GPIS single-history profile:\s*(\{.*\})', text)[-1])
    discarded = json.loads(re.findall(r'GPIS single-history discarded surface births:\s*(\{.*\})', text)[-1])
    assert discarded == dict(paths=0, shadow_connections=0)
    rgb, a, b, var = [pfm(directory / name) for name in ('render.pfm', 'samplesA.pfm', 'samplesB.pfm', 'samplesVariance.pfm')]
    assert all(np.isfinite(x).all() for x in (rgb, a, b, var))
    assert rgb.shape == var.shape == (64, 64, 3) and rgb.max() > 0
    assert np.allclose((a+b)/2, rgb, rtol=4e-6, atol=4e-7)
    assert np.max(np.abs(rgb*1024-np.round(rgb*1024))) < 2e-3
    assert np.allclose(var*1023, rgb*(1-rgb), rtol=2e-4, atol=2e-6)
    seconds = float(re.findall(r'Completed\s+1024/1024\s+spp,\s*([\d.]+)/', text)[-1])
    record = dict(seed=seed, spp=1024, resolution=[64, 64], threads=16, render_seconds=seconds,
                  wall_seconds=time.monotonic()-started, scene=load(directory / 'scene.json'),
                  warnings=warnings, profile=counters, discarded_surface_births=discarded,
                  renderer_sha256=sha(exe), finished_at=now(), sha256={name: sha(directory / name) for name in FILES})
    save(directory / 'result.json', record)
    progress()
    print(f'DONE seed={seed} {seconds:.2f}s', flush=True)
    return verify_record(MODE, seed)


def link_parents():
    for parent in (BASE, BASE / 'neural_single_history_continued', BASE / 'single_history_comparison'):
        page = parent / 'comparison.html'
        text = page.read_text(encoding='utf-8')
        backup = OUT / 'history' / parent.name
        backup.mkdir(parents=True, exist_ok=True)
        for name in ('comparison.html', 'verification.json'):
            if (parent / name).exists() and not (backup / name).exists():
                shutil.copy2(parent / name, backup / name)
        relative = os.path.relpath(OUT / 'comparison.html', parent).replace('\\', '/')
        block = BEGIN + '<section class="note"><h2>解析协方差版 single_history：1024 spp 已完成</h2>'
        block += f'<p>64×64、两个种子；与 start_conditioning、续训 neural、GPIS Renewal+、GPIS Global 比较。<a href="{relative}">查看新渲染、FLIP、透射率与耗时</a></p></section>' + END
        if BEGIN in text:
            start = text.index(BEGIN)
            end = text.index(END, start)+len(END)
            text = text[:start]+block+text[end:]
        else:
            index = text.index('</h1>')+len('</h1>')
            text = text[:index]+block+text[index:]
        page.write_text(text, encoding='utf-8')
        verification = load(parent / 'verification.json')
        verification.update(html_sha256=sha(page), local_links=verify_links(page),
                            analytic_single_history_1024=dict(complete=True, page=relative))
        save(parent / 'verification.json', verification)


def report():
    sources, pairs, old_checks, sections = [], [], [], []
    masks = roi_masks(64)
    for seed in SEEDS:
        images, labels, rows = [], [], []
        for mode in MODES:
            directory = folder(mode, seed)
            result = verify_record(mode, seed)
            rgb = pfm(directory / 'render.pfm') @ Y
            roi = {name: float(rgb[mask].mean()) for name, mask in masks.items()}
            sources.append(dict(mode=mode, seed=seed, roi=roi, render_seconds=result['render_seconds'], result=url(ARGS, directory / 'result.json')))
            images.append(directory / 'render.png')
            labels.append(LABELS[mode])
            rows.append(f'<tr><td>{LABELS[mode]}</td><td>{roi["overlap"]:.6f}</td><td>{roi["left_single"]:.6f}</td><td>{roi["right_single"]:.6f}</td><td>{result["render_seconds"]:.2f}</td><td><a href="{url(ARGS,directory/"render.png")}">PNG</a> · <a href="{url(ARGS,directory/"scene.json")}">配置</a> · <a href="{url(ARGS,directory/"render.log")}">日志</a></td></tr>')
        figure = OUT / 'figures' / f'seed{seed}'
        section = f'<h2>64×64 · 1024 spp · seed {seed}</h2>' + picture_grid(ARGS, images, labels, figure / 'appearance.png', 5)
        section += '<table><tr><th>模式</th><th>重叠区</th><th>左单层</th><th>右单层</th><th>秒</th><th>文件</th></tr>' + ''.join(rows) + '</table>'
        maps, map_labels, flip_rows = [], [], []
        for reference in ('gpis_global', 'gpis_renewal_plus', 'global_conditional', 'neural_continued'):
            pair = compare(ARGS, seed, folder(reference, seed), folder(MODE, seed), 'single_history_vs_'+reference)
            pair.update(reference=reference, test=MODE)
            pairs.append(pair)
            maps.append(OUT / pair['error_map'])
            map_labels.append(LABELS[reference]+f'\nFLIP {pair["flip"]["mean"]:.6f}')
            flip_rows.append(f'<tr><td>{LABELS[reference]}</td><td>{pair["flip"]["mean"]:.6f}</td><td>{pair["linear_difference_rms"]:.6f}</td></tr>')
        section += '<h3>single_history 的本地 FLIP 误差图（标题为参考模式）</h3>' + picture_grid(ARGS, maps, map_labels, figure / 'flip.png', 4)
        section += '<table><tr><th>参考</th><th>Mean FLIP</th><th>线性差值 RMS</th></tr>' + ''.join(flip_rows) + '</table>'
        fig, ax = plt.subplots(figsize=(11, 3.6))
        for mode in MODES:
            ax.plot((pfm(folder(mode, seed) / 'render.pfm') @ Y)[32], label=LABELS[mode])
        ax.set(xlabel='Pixel x', ylabel='Linear transmittance', ylim=(0, 1.03))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figure / 'profile.png', dpi=130)
        plt.close(fig)
        section += f'<img src="{url(ARGS, figure/"profile.png")}">'
        old = verify_record('previous', seed)
        delta = pfm(folder(MODE, seed) / 'render.pfm') - pfm(folder('previous', seed) / 'render.pfm')
        check = dict(seed=seed, identical=sha(folder(MODE, seed) / 'render.pfm') == old['sha256']['render.pfm'],
                     max_absolute_pixel_difference=float(np.abs(delta).max()))
        old_checks.append(check)
        section += f'<p>与旧版同种子 single_history：PFM 是否完全一致：{check["identical"]}；最大像素差 {check["max_absolute_pixel_difference"]:g}。</p>'
        sections.append(section)
    aggregate = []
    for mode in MODES:
        rows = [r for r in sources if r['mode'] == mode]
        aggregate.append(dict(mode=mode, overlap_transmittance=float(np.mean([r['roi']['overlap'] for r in rows])),
                              average_render_seconds=float(np.mean([r['render_seconds'] for r in rows]))))
    save(OUT / 'metrics.json', dict(complete=True, sources=sources, comparisons=pairs, aggregate=aggregate, previous_single_history_checks=old_checks))
    with (OUT / 'render_times.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['mode', 'seed', 'render_seconds'])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in sources)
    averages = ''.join(f'<tr><td>{LABELS[r["mode"]]}</td><td>{r["overlap_transmittance"]:.6f}</td><td>{r["average_render_seconds"]:.2f}</td></tr>' for r in aggregate)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>双平面：解析协方差 single_history 1024 spp</title><style>body{max-width:1500px;margin:30px auto;padding:0 20px;font:16px/1.65 system-ui;color:#243346;background:#f5f6f8}img{width:100%;height:auto;background:white}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}a{color:#235e91}.note{padding:16px;background:#e7edf5}</style>
<h1>双平面：解析协方差 single_history · 1024 spp</h1>
<div class="note">本次重新渲染原生 single_history：64×64、1024 spp、16 线程、两个原有种子。σ=0.02，ℓ=0.2；场景、材质、相机、光照及 history 参数沿用原实验。<br>协方差使用解析公式与稳定求值，退化时保留原特征算法。概率积分、选点策略与消光率上界不变。其他模式引用原有 1024 spp 图像；GPIS 步长均为 0.0025。<br>其他模式的时间是历史记录，不作为本次同负载速度基准。图像统一 filmic；本地 FLIP 使用原始 64×64 PNG、67 PPD。误差包含有限 spp 的采样噪声。</div>
<p><a href="metrics.json">指标</a> · <a href="manifest.json">版本和参数</a> · <a href="render_times.csv">渲染时间</a> · <a href="../local_law_analytic_optimization/validation.json">此前配对性能验证</a></p>
<table><tr><th>模式</th><th>两种子重叠区透射率均值</th><th>平均秒数</th></tr>'''+averages+'</table>'+''.join(sections)+'</html>'
    destination = OUT / 'comparison.html'
    destination.write_text(page, encoding='utf-8')
    manifest = load(OUT / 'manifest.json')
    for path, digest in manifest['reference_sha256'].items():
        assert sha(Path(path)) == digest
    save(OUT / 'verification.json', dict(passed=True, complete=True, new_render_records=2, native_flip_records=len(pairs),
                                        original_outputs_preserved=True, previous_single_history_checks=old_checks,
                                        local_links=verify_links(destination), html_sha256=sha(destination), metrics_sha256=sha(OUT / 'metrics.json')))
    link_parents()
    print(json.dumps(dict(aggregate=aggregate, previous_single_history_checks=old_checks)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('run', 'report'), default='run', nargs='?')
    args = parser.parse_args()
    prepare()
    try:
        if args.action == 'run':
            for seed in SEEDS:
                render(seed)
        report()
        progress()
    except BaseException:
        progress(failure=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
