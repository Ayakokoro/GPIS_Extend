"""Matched two-slab renders of the learned single-point extinction correction.

The data and trained model belong in OUTPUT/data and OUTPUT/model. This runner
freezes a newly built renderer separately and reuses verified existing images.
Use the local plotting Python with prepare, smoke, run, report, or status.
"""
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
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
from figures import plt
from run import ROOT, load, sha, pfm
from gpis_sphere_small_light import save, now, FATAL, FILES, Y
from gpis_shaderball_flip import DEFAULT_FLIP, verify_links
from gpis_two_slabs import roi_masks
from gpis_two_slabs_step_refinement import compare, picture_grid, url

BASE = ROOT / 'outputs/gpis_two_slabs_compare_20261113'
OUT = BASE / 'neural_single_history_comparison'
MODE = 'global_conditional_neural'
SEEDS = (20261024, 20261113)
QUALITY = dict(smoke=(32, 8), final=(64, 1024))
REFERENCES = ('global_conditional', 'single_history_gpis', 'gpis_renewal_plus', 'gpis_global')
MODES = ('global_conditional', MODE, 'single_history_gpis', 'gpis_renewal_plus', 'gpis_global')
LABELS = dict(global_conditional='start_conditioning',
              global_conditional_neural='start_conditioning + neural single history',
              single_history_gpis='native single_history',
              gpis_renewal_plus='GPIS renewal+ (1D MIS)', gpis_global='GPIS Global (3D)')
BEGIN = '<!-- BEGIN TWO SLABS NEURAL SINGLE HISTORY -->'
END = '<!-- END TWO SLABS NEURAL SINGLE HISTORY -->'


def folder(args, stage, mode, seed):
    if mode == MODE:
        root = args.output
    elif mode == 'gpis_global':
        root = args.base / 'gpis_global_comparison'
    elif mode == 'single_history_gpis':
        root = args.base / 'single_history_comparison'
    else:
        root = args.base
    return root / stage / mode / f'seed{seed}'


def jobs(stage):
    return [(stage, seed) for seed in (SEEDS if stage == 'final' else SEEDS[:1])]


def model_path(args):
    return args.output / 'model/model.json'


def scene(args, stage, seed):
    old = load(folder(args, stage, 'global_conditional', seed) / 'scene.json')
    value = copy.deepcopy(old)
    medium = value['media'][0]
    medium['type'] = MODE
    medium['single_point_neural'] = dict(model=str(model_path(args)), correction_cap=args.cap)
    assert medium['fp_gradient']['enabled'] is False
    value['renderer']['output_directory'] = str(folder(args, stage, MODE, seed))
    check = copy.deepcopy(value)
    check['media'][0]['type'] = old['media'][0]['type']
    del check['media'][0]['single_point_neural']
    check['renderer']['output_directory'] = old['renderer']['output_directory']
    assert check == old, 'Unintended physical scene difference'
    assert value['camera']['resolution'] == [QUALITY[stage][0]] * 2
    assert value['renderer']['spp'] == QUALITY[stage][1]
    return value


def preserve_references(args):
    preserved = {}
    original = load(args.base / 'manifest.json')
    for name, digest in original['runtime_sha256'].items():
        assert sha(args.base / 'renderer' / name) == digest, name
    for stage in QUALITY:
        for _, seed in jobs(stage):
            for mode in REFERENCES:
                d = folder(args, stage, mode, seed)
                result = load(d / 'result.json')
                assert result['renderer_sha256'] == original['runtime_sha256']['tungsten.exe']
                for name, digest in result['sha256'].items():
                    assert sha(d / name) == digest, str(d / name)
                    preserved[(d / name).relative_to(args.base).as_posix()] = digest
                preserved[(d / 'result.json').relative_to(args.base).as_posix()] = sha(d / 'result.json')
    for sub in ('', 'single_history_comparison/', 'gpis_global_comparison/', 'renewal_step_refinement/'):
        for name in ('manifest.json', 'metrics.json'):
            preserved[sub + name] = sha(args.base / (sub + name))
    return preserved


def check_frozen(args, manifest):
    for name, digest in manifest['runtime_sha256'].items():
        assert sha(args.output / 'renderer' / name) == digest, name
    for name, digest in manifest['training_sha256'].items():
        assert sha(args.output / name) == digest, name
    for name, digest in manifest['preserved_original_sha256'].items():
        assert sha(args.base / name) == digest, name
    assert sha(args.flip) == manifest['flip']['sha256']


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'manifest.json'
    if path.exists():
        manifest = load(path)
        assert manifest['runner_sha256'] == sha(Path(__file__)), 'Runner changed since preparation'
        assert manifest['correction_cap'] == args.cap
        assert manifest['threads'] == args.threads and manifest['parallel_jobs'] == args.parallel_jobs
        check_frozen(args, manifest)
    else:
        model = load(model_path(args))
        data = load(args.output / 'data/manifest.json')
        assert model['schema'] == 'tungsten.single_point_history_mlp.v2'
        assert data['status'] == 'complete'
        # Native scene covariance parameters pass through float32 storage.
        assert np.isclose(model['sigma'], .02, rtol=1e-7, atol=0)
        assert np.isclose(model['length_scale'], .2, rtol=1e-7, atol=0)
        assert model['deployment']['correction_cap'] == args.cap
        assert model['deployment']['applied_to_training_loss'] is False
        assert model['deployment']['applied_to_checkpoint_selection'] is False
        preserved = preserve_references(args)
        history = args.output / 'history'
        history.mkdir(exist_ok=True)
        for name in ('comparison.html', 'verification.json', 'manifest.json', 'metrics.json', 'progress.json'):
            dest = history / name
            if not dest.exists():
                shutil.copy2(args.base / name, dest)
        runtime = args.output / 'renderer'
        runtime.mkdir(exist_ok=True)
        inputs = [args.renderer] + sorted(args.renderer.parent.glob('*.dll'))
        for source in inputs:
            dest = runtime / source.name
            if dest.exists():
                assert sha(source) == sha(dest), 'Existing runtime differs: ' + str(dest)
            else:
                shutil.copy2(source, dest)
        source_dir = args.output / 'source'
        source_dir.mkdir(exist_ok=True)
        sources = [Path(__file__), ROOT / 'tools/single_point_training/model.py',
                   ROOT / 'tools/single_point_training/train.py', ROOT / 'tools/single_point_training/generate.cpp',
                   ROOT / 'src/core/media/GlobalConditionalMedium.cpp', ROOT / 'src/core/media/GlobalConditionalMedium.hpp',
                   ROOT / 'src/core/math/conditional/transport/ConditionalSegment.cpp',
                   ROOT / 'src/core/math/conditional/transport/ConditionalSegment.h']
        sources += sorted((ROOT / 'src/core/math/conditional/transport').glob('SinglePoint*'))
        sources += [ROOT / name for name in (
            'CMakeLists.txt', 'src/core/media/MediumFactory.cpp',
            'src/core/math/GPFunctions.hpp', 'src/core/math/ProceduralMeanSdf.cpp',
            'src/core/math/SdfQueries.hpp', 'src/core/math/SdfFunctions.hpp',
            'src/core/math/SdfFunctionsQueries.cpp', 'src/core/math/conditional/gpss/MeanField.h',
            'src/core/math/conditional/gpss/SdfMeanField.hpp',
            'tools/single_point_training/SceneField.hpp', 'tools/single_point_training/SceneField.cpp')]
        for source in sources:
            dest = source_dir / source.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
        training_files = ('data/manifest.json', 'data/samples.csv', 'model/model.json', 'model/best.pt',
                          'model/metrics.json', 'model/config.json', 'model/history.csv')
        manifest = dict(created_at=now(), mode=MODE, correction_cap=args.cap,
            quality=QUALITY, seeds=SEEDS, threads=args.threads, parallel_jobs=args.parallel_jobs,
            runner_sha256=sha(Path(__file__)),
            runtime_sha256={p.name: sha(p) for p in runtime.iterdir() if p.is_file()},
            source_sha256={p.relative_to(source_dir).as_posix(): sha(p) for p in source_dir.rglob('*') if p.is_file()},
            training_sha256={name: sha(args.output / name) for name in training_files},
            preserved_original_sha256=preserved, flip=dict(path=str(args.flip), sha256=sha(args.flip), ppd=67.),
            notes=[
                'New global_conditional_neural mode; original global_conditional remains available.',
                'Only extinction / distance sampling and transmittance use the learned correction. Collision gradient law is unchanged.',
                'Training labels retain full sampled (F0, gradient0) birth and one additional F(u)>0 history observation.',
                'Inference-only upper correction cap; training labels, loss and checkpoint selection are uncapped by the user cap.',
                'All proposal bounds use cap * analytic baseline majorant, including changing history anchors.',
                'No fixed neural marching step. Geometry selects anchors and a continuous query computes six features.',
                'Original camera, geometry, material, lighting, sigma=.02, length_scale=.2, quality and seeds are preserved.',
                'Old native single_history has a different exterior birth model and also changes collision gradients; it is a reference, not this model\'s exact teacher.',
                'Existing reference renders use the original frozen executable. Their runtime is not a controlled speed benchmark.'])
        save(path, manifest)
        check_frozen(args, manifest)
    for stage in QUALITY:
        for _, seed in jobs(stage):
            dest = folder(args, stage, MODE, seed) / 'scene.json'
            value = scene(args, stage, seed)
            if dest.exists():
                assert load(dest) == value, str(dest)
            else:
                save(dest, value)
    return manifest


def verified(args, stage, mode, seed):
    d = folder(args, stage, mode, seed)
    r = load(d / 'result.json')
    manifest = load(args.output / 'manifest.json')
    expected = (manifest['runtime_sha256']['tungsten.exe'] if mode == MODE else
                load(args.base / 'manifest.json')['runtime_sha256']['tungsten.exe'])
    assert r['renderer_sha256'] == expected
    if mode == MODE:
        assert r['scene'] == scene(args, stage, seed)
        assert r['model_sha256'] == sha(model_path(args))
    assert r['spp'] == QUALITY[stage][1] and r['seed'] == seed and not r['warnings']
    for name, digest in r['sha256'].items():
        assert sha(d / name) == digest, str(d / name)
    return r


def status(args, active=None, failure=None):
    active_paths = {active} if isinstance(active, str) else set(active or [])
    rows = []
    for stage in QUALITY:
        for _, seed in jobs(stage):
            d = folder(args, stage, MODE, seed)
            text = (d / 'render.log').read_text(encoding='utf-8', errors='replace') if (d / 'render.log').exists() else ''
            samples = re.findall(r'Completed (\d+)/(\d+) spp', text)
            rows.append(dict(stage=stage, seed=seed, mode=MODE,
                state='complete' if (d / 'result.json').exists() else 'running' if str(d) in active_paths else 'pending',
                spp=int(samples[-1][0]) if samples else 0, target_spp=QUALITY[stage][1]))
    final = [r for r in rows if r['stage'] == 'final']
    save(args.output / 'progress.json', dict(updated_at=now(), active=active, jobs=rows,
        complete=all(r['state'] == 'complete' for r in final),
        completed=sum(r['state'] == 'complete' for r in final), total=len(final), failure=failure))


def render(args, stage, seed, update_status=True):
    d = folder(args, stage, MODE, seed)
    if (d / 'result.json').exists():
        return verified(args, stage, MODE, seed)
    assert not (d / 'render.log').exists(), 'Unverified existing attempt: ' + str(d)
    exe = args.output / 'renderer/tungsten.exe'
    command = [str(exe), '-t', str(args.threads), '-s', str(seed), '-d', str(d), str(d / 'scene.json')]
    print('START', stage, MODE, seed, flush=True)
    started = time.monotonic()
    with (d / 'render.log').open('wb') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) | getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
        save(d / 'job.json', dict(pid=process.pid, command=command, started_at=now()))
        try:
            while process.poll() is None:
                if update_status:
                    status(args, str(d))
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            process.terminate()
            process.wait()
            raise
    text = (d / 'render.log').read_text(encoding='utf-8', errors='replace')
    size, spp = QUALITY[stage]
    assert process.returncode == 0 and 'Finished render.' in text and not FATAL.search(text), str(d / 'render.log')
    assert re.search(rf'Completed\s+{spp}/{spp}\s+spp', text)
    warnings = [line for line in text.splitlines() if 'sanity check failed' in line]
    assert not warnings, warnings
    # Keep all native JSON diagnostics without depending on unrelated profiles.
    diagnostics = {}
    for line in text.splitlines():
        match = re.search(r'([^\r\n]+?):\s*(\{.*\})\s*$', line)
        if match:
            try:
                diagnostics[match[1]] = json.loads(match[2])
            except json.JSONDecodeError:
                pass
    neural = diagnostics.get('GPIS single-point neural statistics')
    assert neural and neural['network_queries'] > 0, 'New neural mode did not record network evaluations'
    assert neural['collision_gradient_correction'] is False
    assert neural['correction_cap'] == args.cap
    assert neural['outside_domain'] == 0, 'Matched two-slab rays unexpectedly left the training domain'
    discarded = diagnostics.get('GPIS conditional discarded surface births')
    assert discarded == dict(paths=0, shadow_connections=0), discarded
    rgb, a, b, variance = [pfm(d / name) for name in ('render.pfm', 'samplesA.pfm', 'samplesB.pfm', 'samplesVariance.pfm')]
    assert rgb.shape == variance.shape == (size, size, 3) and rgb.max() > 0
    assert all(np.isfinite(x).all() for x in (rgb, a, b, variance))
    assert np.allclose((a + b) / 2, rgb, rtol=4e-6, atol=4e-7)
    assert np.max(np.abs(rgb * spp - np.round(rgb * spp))) < 2e-3, 'Pure absorption must be Bernoulli transmission'
    assert np.allclose(variance * (spp - 1), rgb * (1 - rgb), rtol=2e-4, atol=2e-6)
    times = re.findall(r'Completed\s+\d+/\d+\s+spp,\s*([\d.]+)/[\d.]+\s+sec', text)
    assert times, 'Missing render duration'
    r = dict(scene=scene(args, stage, seed), seed=seed, spp=spp, resolution=[size, size], threads=args.threads,
        renderer_sha256=sha(exe), model_sha256=sha(model_path(args)), correction_cap=args.cap,
        render_seconds=float(times[-1]), wall_seconds=time.monotonic() - started,
        warnings=warnings, native_diagnostics=diagnostics, gradient_correction_enabled=False,
        mean_transmittance=float(rgb.mean()), finished_at=now(), sha256={name: sha(d / name) for name in FILES})
    save(d / 'result.json', r)
    print('DONE', stage, seed, r['render_seconds'], 's; T', r['mean_transmittance'], flush=True)
    return verified(args, stage, MODE, seed)


def link_parent(args, complete, aggregate):
    path = args.base / 'comparison.html'
    text = path.read_text(encoding='utf-8')
    link = os.path.relpath(args.output / 'comparison.html', args.base).replace('\\', '/')
    state = '已完成' if complete else '进行中'
    block = (BEGIN + f'<section id="neural-single-history-comparison"><h2>单点神经网络透射率修正：{state}</h2>'
        f'<div class="note">独立模式 global_conditional_neural；保持原碰撞点梯度分布。64×64、1024 spp、两个种子，'
        f'修正上限 C={args.cap:g} 仅在推理时使用。<br><a href="{link}">查看训练指标、渲染图、透射率和 FLIP</a></div>')
    if aggregate:
        block += '<table><tr><th>模式</th><th>重叠区透射率</th></tr>' + ''.join(
            f'<tr><td>{LABELS[r["mode"]]}</td><td>{r["overlap_transmittance"]:.6f}</td></tr>' for r in aggregate) + '</table>'
    image = args.output / 'figures' / f'seed{SEEDS[0]}' / 'appearance.png'
    if image.exists():
        relative = os.path.relpath(image, args.base).replace('\\', '/')
        block += f'<a href="{link}"><img src="{relative}"></a>'
    block += '</section>' + END
    if BEGIN in text:
        a = text.index(BEGIN)
        b = text.index(END, a) + len(END)
        text = text[:a] + block + text[b:]
    else:
        a = text.index('</h1>') + len('</h1>')
        text = text[:a] + block + text[a:]
    path.write_text(text, encoding='utf-8')
    verification = load(args.base / 'verification.json')
    verification.update(html_sha256=sha(path), local_links=verify_links(path), neural_single_history_comparison=dict(
        complete=complete, page=link, verification=link.replace('comparison.html', 'verification.json')))
    save(args.base / 'verification.json', verification)


def training_section(args):
    metrics = load(args.output / 'model/metrics.json')
    manifest = load(args.output / 'data/manifest.json')
    model = load(model_path(args))
    width = model['architecture']['width']
    transform = model.get('feature_transform', 'identity')
    rows = ''
    for name, m in metrics['partitions'].items():
        capped = m['render_capped']
        rows += (f'<tr><td>{html.escape(name)}</td><td>{m["count"]}</td><td>{m["baseline"]["log_mae"]:.6f}</td>'
                 f'<td>{m["log_mae"]:.6f}</td><td>{m["log_p95"]:.6f}</td><td>{capped["log_mae"]:.6f}</td>'
                 f'<td>{capped["prediction_clipped_fraction"]:.3%}</td><td>{capped["teacher_above_cap_count"]}</td></tr>')
    counts = manifest.get('split_counts', [])
    section = (f'<h2>单点修正网络训练</h2><p>6→{width}→{width}→1 MLP，特征变换 {html.escape(transform)}。'
        f'{manifest["rows"]:,} 条标签，训练/验证/测试 {counts}；'
        f'{metrics["parameter_count"]:,} 个参数，完成 {metrics["epochs_completed"]} 个 epoch，选中 epoch {metrics["best_epoch"]}。'
        '按完整视角划分，训练与验证不共享同一条射线。标签由数值单点历史条件化计算，未使用渲染图片作为监督。</p>'
        '<table><tr><th>划分</th><th>标签数</th><th>原始 log R MAE</th><th>网络 log R MAE</th>'
        '<th>绝对误差 P95</th><th>加推理上限后 MAE</th><th>预测触顶比例</th><th>教师超过上限数</th></tr>' + rows + '</table>'
        '<p>上限后的误差仍对原始、未截断的教师计算。保留完整起点 F 和梯度，只增加一个历史正值约束；'
        '这是单点近似，训练误差小不等于完整 GPIS Global 无偏。其他场景和参数需要重新生成数据与训练。</p>'
        '<p><a href="data/manifest.json">数据来源与采样</a> · <a href="model/config.json">训练设置</a> · '
        '<a href="model/metrics.json">完整训练指标</a> · <a href="model/history.csv">训练曲线 CSV</a> · '
        '<a href="model/model.json">导出模型</a></p>')
    history = np.genfromtxt(args.output / 'model/history.csv', delimiter=',', names=True)
    history = np.atleast_1d(history)
    names = history.dtype.names or ()
    if len(history) and 'epoch' in names:
        choices = [name for name in names if name in ('validation_log_mae', 'train_loss', 'training_loss', 'validation_loss')]
        if choices:
            fig, ax = plt.subplots(figsize=(9, 3))
            for name in choices:
                ax.plot(history['epoch'], history[name], label=name)
            ax.set(xlabel='Epoch', ylabel='Loss / log correction error')
            ax.legend()
            fig.tight_layout()
            destination = args.output / 'figures/training.png'
            destination.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(destination, dpi=130)
            plt.close(fig)
            section += '<img src="figures/training.png">'
    return section, metrics


def report(args):
    manifest = load(args.output / 'manifest.json')
    check_frozen(args, manifest)
    sources, pairs, sections = [], [], []
    for stage in ('final', 'smoke'):
        size, spp = QUALITY[stage]
        masks = roi_masks(size)
        for _, seed in jobs(stage):
            available = [mode for mode in MODES if (folder(args, stage, mode, seed) / 'result.json').exists()]
            fd = args.output / 'figures' / (f'seed{seed}' if stage == 'final' else f'smoke_seed{seed}')
            images, labels, rows = [], [], ''
            for mode in available:
                d = folder(args, stage, mode, seed)
                result = verified(args, stage, mode, seed)
                rgb = pfm(d / 'render.pfm') @ Y
                rois = {name: float(rgb[mask].mean()) for name, mask in masks.items()}
                sources.append(dict(stage=stage, mode=mode, seed=seed, roi=rois,
                    render_seconds=result['render_seconds'], result=url(args, d / 'result.json')))
                images.append(d / 'render.png')
                labels.append(LABELS[mode] + f'\n{result["render_seconds"]:.2f} s')
                rows += (f'<tr><td>{LABELS[mode]}</td><td>{rois["overlap"]:.6f}</td>'
                    f'<td>{rois["left_single"]:.6f}</td><td>{rois["right_single"]:.6f}</td>'
                    f'<td>{result["render_seconds"]:.2f} s</td><td><a href="{url(args,d/"render.png")}">PNG</a> · '
                    f'<a href="{url(args,d/"scene.json")}">配置</a> · <a href="{url(args,d/"render.log")}">日志</a></td></tr>')
            section = f'<h2>{stage} · {size}×{size} · {spp} spp · seed {seed}</h2>'
            section += picture_grid(args, images, labels, fd / 'appearance.png', 5)
            section += ('<table><tr><th>模式</th><th>重叠区透射率</th><th>左侧单层区</th><th>右侧单层区</th>'
                        '<th>渲染时间</th><th>文件</th></tr>' + rows + '</table>')
            if MODE in available:
                maps, map_labels, flip_rows = [], [], ''
                for ref in REFERENCES:
                    pair = compare(args, seed, folder(args, stage, ref, seed), folder(args, stage, MODE, seed),
                                   stage + '_neural_single_history_vs_' + ref)
                    pair.update(stage=stage, reference=ref, test=MODE)
                    pairs.append(pair)
                    maps.append(args.output / pair['error_map'])
                    map_labels.append('Neural vs ' + LABELS[ref] + f'\nFLIP {pair["flip"]["mean"]:.6f}')
                    flip_rows += (f'<tr><td>{LABELS[ref]}</td><td>{pair["flip"]["mean"]:.6f}</td>'
                        f'<td>{pair["linear_difference_rms"]:.6f}</td><td>{pair["estimated_difference_noise_rms"]:.6f}</td></tr>')
                section += '<h3>新模式与四个已有参考的 FLIP</h3>' + picture_grid(args, maps, map_labels, fd / 'flip.png', 4)
                section += ('<table><tr><th>参考</th><th>平均 FLIP</th><th>线性差值 RMS</th>'
                            '<th>差值噪声 RMS 估计</th></tr>' + flip_rows + '</table>')
            fig, ax = plt.subplots(figsize=(10, 3.6))
            for mode in available:
                ax.plot((pfm(folder(args, stage, mode, seed) / 'render.pfm') @ Y)[size // 2], label=LABELS[mode])
            ax.set(xlabel='Pixel x', ylabel='Linear transmittance', ylim=(0, 1.03), title='Center-row profile')
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(fd / 'profile.png', dpi=130)
            plt.close(fig)
            section += f'<img src="{url(args,fd/"profile.png")}">'
            sections.append(section)
    aggregate = []
    for mode in MODES:
        rows = [r for r in sources if r['stage'] == 'final' and r['mode'] == mode]
        if len(rows) == len(SEEDS):
            aggregate.append(dict(mode=mode,
                overlap_transmittance=float(np.mean([r['roi']['overlap'] for r in rows])),
                single_transmittance=float(np.mean([r['roi'][key] for r in rows for key in ('left_single', 'right_single')])),
                average_render_seconds=float(np.mean([r['render_seconds'] for r in rows]))))
    complete = sum(r['stage'] == 'final' and r['mode'] == MODE for r in sources) == len(SEEDS)
    training_html, training_metrics = training_section(args)
    save(args.output / 'metrics.json', dict(complete=complete, sources=sources, comparisons=pairs, aggregate=aggregate,
        training=dict(best_epoch=training_metrics['best_epoch'], partitions=training_metrics['partitions']),
        correction_cap=args.cap, collision_gradient_correction=False))
    with (args.output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['mode', 'overlap_transmittance', 'single_transmittance', 'average_render_seconds'])
        writer.writeheader()
        writer.writerows(aggregate)
    rows = ''.join(f'<tr><td>{LABELS[r["mode"]]}</td><td>{r["overlap_transmittance"]:.6f}</td>'
                   f'<td>{r["single_transmittance"]:.6f}</td><td>{r["average_render_seconds"]:.2f} s</td></tr>' for r in aggregate)
    page = ('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>双薄板：单点神经网络透射率修正</title>
<style>body{max-width:1500px;margin:30px auto;padding:0 20px;font:16px/1.65 system-ui;color:#243346;background:#f5f6f8}img{width:100%;height:auto;background:white}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}a{color:#235e91}.note{padding:16px;background:#e7edf5}h2{margin-top:32px}</style>
<h1>双薄板：start_conditioning + 单点神经网络修正</h1>
<div class="note">独立新模式 <code>global_conditional_neural</code>。只修正距离采样、消光率和透射率；碰撞点梯度分布保留原 <code>global_conditional</code> 的实现。<br>
64×64、1024 spp、两个原有种子；σ=0.02、ℓ=0.2、纯吸收材质、相同相机与背光。原始参考图和程序均保留。</div>
''' + f'<p>小型 MLP 仅估计解析起点条件化消光率的单点历史修正。推理修正上限 C={args.cap:g}，不影响训练标签、损失或 checkpoint 选择。'
        '网络不要求固定步长；GPIS Global 和 Renewal+ 引用原有步长 0.0025 的结果。原生 single_history 的外部起点只保留 F&gt;0，'
        '且同时改变梯度分布，因此它与新网络的教师不是同一种条件模型。</p>'
        '<p>线性 PFM 用于透射率与 RMS，所有显示图共用 filmic，不降噪、不单独调曝光。'
        'FLIP 使用本地可执行程序在原始分辨率、67 PPD 下计算；展示仅最近邻放大。差值噪声由两半样本估计，不能单独证明无偏。</p>'
        '<table><tr><th>模式</th><th>重叠区透射率均值</th><th>两侧单层区均值</th><th>平均渲染时间</th></tr>' + rows + '</table>'
        f'<p>新模式每任务 {args.threads} 线程，最多 {args.parallel_jobs} 个任务并行。旧参考来自原有程序与执行记录，耗时仅供参考。</p>'
        f'<p><a href="{url(args,args.base/"comparison.html")}">原比较页</a> · <a href="manifest.json">版本与文件校验</a> · '
        '<a href="metrics.json">渲染指标</a> · <a href="summary.csv">CSV</a> · <a href="progress.json">进度</a></p>'
        + training_html + ''.join(sections) + '</html>')
    (args.output / 'comparison.html').write_text(page, encoding='utf-8')
    save(args.output / 'verification.json', dict(passed=True, complete=complete, render_records=len(sources),
        new_render_records=sum(r['mode'] == MODE for r in sources), native_flip_records=len(pairs),
        shared_physical_scene=True, original_outputs_preserved=True, collision_gradient_correction=False,
        local_links=verify_links(args.output / 'comparison.html'), html_sha256=sha(args.output / 'comparison.html'),
        metrics_sha256=sha(args.output / 'metrics.json')))
    link_parent(args, complete, aggregate)
    print('REPORT', len(sources), 'images;', len(pairs), 'FLIP records', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'smoke', 'run', 'report', 'status'))
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--renderer', type=Path, default=ROOT / 'build/Release/tungsten.exe')
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--parallel-jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--cap', type=float, default=2.)
    parser.add_argument('--flip', type=Path, default=DEFAULT_FLIP)
    args = parser.parse_args()
    for name in ('base', 'output', 'renderer', 'flip'):
        setattr(args, name, getattr(args, name).resolve())
    assert args.threads > 0 and args.threads * args.parallel_jobs <= (os.cpu_count() or 1)
    assert np.isfinite(args.cap) and args.cap >= 1 and args.flip.is_file()
    if args.command == 'status':
        print(json.dumps(load(args.output / 'progress.json'), indent=2))
        return
    prepare(args)
    if args.command in ('prepare', 'report'):
        status(args)
        report(args)
        return
    lock = args.output / 'runner.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(dict(pid=os.getpid(), started_at=now()), stream)
    try:
        for stage in (('smoke',) if args.command == 'smoke' else ('smoke', 'final')):
            if len(jobs(stage)) == 1 or args.parallel_jobs == 1:
                for job in jobs(stage):
                    render(args, *job)
                    status(args)
                    report(args)
            else:
                with ThreadPoolExecutor(max_workers=args.parallel_jobs) as pool:
                    pending = {pool.submit(render, args, *job, False): job for job in jobs(stage)}
                    while pending:
                        status(args, active=[str(folder(args, job[0], MODE, job[1])) for job in pending.values()])
                        done, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                        for future in done:
                            future.result()
                            pending.pop(future)
                            report(args)
                status(args)
        print('COMPLETE', args.command, flush=True)
    except Exception as exc:
        failure = dict(error=str(exc), traceback=traceback.format_exc())
        save(args.output / 'failure.json', failure)
        status(args, failure=failure)
        raise
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
