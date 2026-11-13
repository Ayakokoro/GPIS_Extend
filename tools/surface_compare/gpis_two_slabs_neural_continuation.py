"""Compare continued single-point training without replacing the first experiment.

Reuses the already validated render helper and frozen native renderer. The data
remain in the first experiment; only the continued checkpoint and new images
belong here. Run after OUTPUT/model has completed training.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np

import gpis_two_slabs_neural_single_history as first
from figures import plt
from run import ROOT, load, sha, pfm
from gpis_sphere_small_light import save, now, Y
from gpis_shaderball_flip import DEFAULT_FLIP, verify_links
from gpis_two_slabs import roi_masks
from gpis_two_slabs_step_refinement import compare, picture_grid, url

BASE = first.BASE
PREVIOUS = BASE / 'neural_single_history_comparison'
OUT = BASE / 'neural_single_history_continued'
MODE = first.MODE
SEEDS, QUALITY = first.SEEDS, first.QUALITY
REFERENCES = ('neural_previous', 'global_conditional', 'gpis_global', 'gpis_renewal_plus')
MODES = ('global_conditional', 'neural_previous', MODE, 'gpis_renewal_plus', 'gpis_global')
LABELS = dict(global_conditional='start_conditioning', neural_previous='Neural: initial training',
              global_conditional_neural='Neural: continued training',
              gpis_renewal_plus='GPIS renewal+ (1D MIS)', gpis_global='GPIS Global (3D)')
BEGIN = '<!-- BEGIN TWO SLABS NEURAL CONTINUATION -->'
END = '<!-- END TWO SLABS NEURAL CONTINUATION -->'


def folder(args, stage, mode, seed):
    if mode == 'neural_previous':
        return args.previous / stage / MODE / f'seed{seed}'
    return first.folder(args, stage, mode, seed)


def reference_records(args):
    records = {}
    expected_old = load(args.base / 'manifest.json')['runtime_sha256']['tungsten.exe']
    expected_nn = load(args.previous / 'manifest.json')['runtime_sha256']['tungsten.exe']
    for stage in QUALITY:
        for _, seed in first.jobs(stage):
            for mode in REFERENCES:
                d = folder(args, stage, mode, seed)
                record = load(d / 'result.json')
                assert record['renderer_sha256'] == (expected_nn if mode == 'neural_previous' else expected_old)
                for name, digest in record['sha256'].items():
                    assert sha(d / name) == digest, str(d / name)
                    records[str(d / name)] = digest
                records[str(d / 'result.json')] = sha(d / 'result.json')
    for name in ('manifest.json', 'metrics.json', 'model/model.json', 'model/best.pt',
                 'model/metrics.json', 'model/config.json', 'model/history.csv'):
        records[str(args.previous / name)] = sha(args.previous / name)
    return records


def check_frozen(args, manifest):
    for name, digest in manifest['runtime_sha256'].items():
        assert sha(args.output / 'renderer' / name) == digest, name
        assert sha(args.previous / 'renderer' / name) == digest, name
    for name, digest in manifest['training_sha256'].items():
        assert sha(args.output / name) == digest, name
    for path, digest in manifest['reference_sha256'].items():
        assert sha(Path(path)) == digest, path
    for path, digest in manifest['dataset_sha256'].items():
        assert sha(Path(path)) == digest, path
    assert sha(args.flip) == manifest['flip']['sha256']


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / 'manifest.json'
    if destination.exists():
        manifest = load(destination)
        assert manifest['runner_sha256'] == sha(Path(__file__)), 'Runner changed since preparation'
        assert manifest['helper_sha256'] == sha(Path(first.__file__)), 'Render helper changed since preparation'
        assert manifest['correction_cap'] == args.cap
        assert manifest['threads'] == args.threads and manifest['parallel_jobs'] == args.parallel_jobs
        assert manifest['previous'] == str(args.previous) and manifest['data'] == str(args.data)
        check_frozen(args, manifest)
    else:
        previous = load(args.previous / 'manifest.json')
        model = load(first.model_path(args))
        config = load(args.output / 'model/config.json')
        old_config = load(args.previous / 'model/config.json')
        data = load(args.data / 'manifest.json')
        assert data['status'] == 'complete'
        assert config['dataset_sha256'] == old_config['dataset_sha256'] == sha(args.data / 'samples.csv')
        assert model['architecture'] == load(args.previous / 'model/model.json')['architecture']
        for key in ('width', 'feature_transform', 'loss_weighting', 'weight_decay'):
            assert config[key] == old_config[key], f'Unintended training change: {key}'
        assert model['schema'] == 'tungsten.single_point_history_mlp.v2'
        assert model['deployment']['correction_cap'] == args.cap
        assert model['deployment']['applied_to_training_loss'] is False
        assert np.isclose(model['sigma'], .02, rtol=1e-7, atol=0)
        assert np.isclose(model['length_scale'], .2, rtol=1e-7, atol=0)
        preserved = reference_records(args)
        runtime = args.output / 'renderer'
        runtime.mkdir(exist_ok=True)
        for name, digest in previous['runtime_sha256'].items():
            source, target = args.previous / 'renderer' / name, runtime / name
            assert sha(source) == digest, str(source)
            if target.exists():
                assert sha(target) == digest, str(target)
            else:
                shutil.copy2(source, target)
        history = args.output / 'history'
        for label, parent in (('original_neural', args.previous), ('original_scene', args.base)):
            for name in ('comparison.html', 'verification.json'):
                target = history / label / name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(parent / name, target)
        source_dir = args.output / 'source'
        sources = [Path(__file__), Path(first.__file__), ROOT / 'tools/single_point_training/train.py',
                   ROOT / 'tools/single_point_training/model.py', ROOT / 'tools/single_point_training/verify_runtime.py']
        for source in sources:
            target = source_dir / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        training_files = ('model/model.json', 'model/best.pt', 'model/metrics.json', 'model/config.json', 'model/history.csv')
        manifest = dict(created_at=now(), mode=MODE, previous=str(args.previous), data=str(args.data),
            correction_cap=args.cap, quality=QUALITY, seeds=SEEDS, threads=args.threads,
            parallel_jobs=args.parallel_jobs, runner_sha256=sha(Path(__file__)),
            helper_sha256=sha(Path(first.__file__)), runtime_sha256=previous['runtime_sha256'],
            dataset_sha256={str(args.data / name): sha(args.data / name) for name in ('manifest.json', 'samples.csv')},
            training_sha256={name: sha(args.output / name) for name in training_files}, reference_sha256=preserved,
            source_sha256={p.relative_to(source_dir).as_posix(): sha(p) for p in source_dir.rglob('*') if p.is_file()},
            flip=dict(path=str(args.flip), sha256=sha(args.flip), ppd=67.),
            notes=['Same dataset, network architecture, feature transform, loss and inference cap as initial training.',
                   'Optimizer/RNG continuation semantics are recorded in model/config.json; no claim of exact resume is inferred here.',
                   'The native renderer is copied byte-for-byte from the first neural experiment.',
                   'Only model path and output directory change relative to the first neural render.',
                   'Collision gradient distributions are unchanged; only distance and transmittance use the learned correction.',
                   'All old checkpoint, render and FLIP artifacts are preserved. Parent HTML receives only a linked extension.'])
        save(destination, manifest)
        check_frozen(args, manifest)
    for stage in QUALITY:
        for _, seed in first.jobs(stage):
            value = first.scene(args, stage, seed)
            old = load(folder(args, stage, 'neural_previous', seed) / 'scene.json')
            matched = json.loads(json.dumps(value))
            matched['media'][0]['single_point_neural']['model'] = old['media'][0]['single_point_neural']['model']
            matched['renderer']['output_directory'] = old['renderer']['output_directory']
            assert matched == old, 'Only model path and output directory may change'
            target = folder(args, stage, MODE, seed) / 'scene.json'
            if target.exists():
                assert load(target) == value
            else:
                save(target, value)
    return manifest


def verified(args, stage, mode, seed):
    if mode == MODE:
        return first.verified(args, stage, mode, seed)
    directory = folder(args, stage, mode, seed)
    record = load(directory / 'result.json')
    assert record['seed'] == seed and record['spp'] == QUALITY[stage][1] and not record['warnings']
    for name, digest in record['sha256'].items():
        assert sha(directory / name) == digest, str(directory / name)
    return record


def training_section(args):
    old = load(args.previous / 'model/metrics.json')
    new = load(args.output / 'model/metrics.json')
    config = load(args.output / 'model/config.json')
    rows = ''
    for split in ('train', 'validation', 'test'):
        a, b = old['partitions'][split], new['partitions'][split]
        rows += (f'<tr><td>{split}</td><td>{a["count"]}</td><td>{a["log_mae"]:.6f}</td>'
                 f'<td>{b["log_mae"]:.6f}</td><td>{a["render_capped"]["log_mae"]:.6f}</td>'
                 f'<td>{b["render_capped"]["log_mae"]:.6f}</td><td>{b["render_capped"]["prediction_clipped_fraction"]:.2%}</td></tr>')
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for directory, label in ((args.previous, 'Initial'), (args.output, 'Continued')):
        values = np.atleast_1d(np.genfromtxt(directory / 'model/history.csv', delimiter=',', names=True))
        ax.plot(values['epoch'], values['validation_log_mae'], label=label + ' validation log MAE')
    starting_epoch = new.get('starting_epoch', old['best_epoch'])
    ax.axvline(starting_epoch, color='gray', linestyle=':', label='Starting checkpoint')
    ax.set(xlabel='Epoch (recorded checkpoint numbering)', ylabel='Uncapped log correction MAE')
    ax.legend(); fig.tight_layout()
    path = args.output / 'figures/training.png'
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=130); plt.close(fig)
    continuation = new.get('continuation') or config.get('continuation') or {}
    semantics = ('旧 checkpoint 没有保存优化器和调度器状态；本次恢复最佳权重，重新建立优化器和调度器。'
                 if continuation.get('kind') == 'warm_start' else '恢复状态的具体范围见续训设置。')
    text = (f'<h2>同一数据上的续训</h2><p>原训练完成 300 轮并选择 epoch {old["best_epoch"]}；'
        f'本次从 epoch {starting_epoch} 的权重继续，新增 {new["epochs_completed"]} 轮，'
        f'结束于 epoch {new.get("final_epoch", starting_epoch + new["epochs_completed"])}，选择 epoch {new["best_epoch"]}。'
        '固定 6→32→32→1 MLP、asinh 特征变换、等权损失和推理上限 C=2。'
        f'训练标签、视角划分和渲染参数不变。续训初始学习率 {config["learning_rate"]:g}。{semantics}</p>'
        f'<p><a href="{url(args,args.data/"manifest.json")}">复用的数据集</a> · <a href="model/config.json">续训设置</a> · '
        '<a href="model/metrics.json">续训指标</a> · <a href="model/history.csv">续训曲线</a></p>'
        '<table><tr><th>划分</th><th>数量</th><th>原 log MAE</th><th>续训 log MAE</th><th>原截断后 MAE</th>'
        '<th>续训截断后 MAE</th><th>续训预测触顶比例</th></tr>' + rows + '</table><img src="figures/training.png">')
    for name, label in (('runtime_parity/verification.json', 'C++ / PyTorch 一致性'),
                        ('diagnostic_comparison.json', '关键区域训练误差诊断')):
        if (args.output / name).exists():
            text += f'<p><a href="{name}">{label}</a></p>'
    return text, dict(previous=old, continued=new, continuation=continuation)


def link_parents(args, complete):
    for parent in (args.previous, args.base):
        page = parent / 'comparison.html'
        text = page.read_text(encoding='utf-8')
        relative = os.path.relpath(args.output / 'comparison.html', parent).replace('\\', '/')
        block = (BEGIN + '<section><h2>单点网络续训对比：' + ('已完成' if complete else '进行中') + '</h2>'
                 '<p>保持数据、网络结构、损失与修正上限不变；原结果保留。'
                 f'<a href="{relative}">查看续训前后、GPIS Global 和 Renewal+ 的图像与 FLIP</a></p></section>' + END)
        if BEGIN in text:
            a, b = text.index(BEGIN), text.index(END) + len(END)
            text = text[:a] + block + text[b:]
        else:
            a = text.index('</h1>') + len('</h1>')
            text = text[:a] + block + text[a:]
        page.write_text(text, encoding='utf-8')
        verification = load(parent / 'verification.json')
        verification.update(html_sha256=sha(page), local_links=verify_links(page), neural_single_history_continued=dict(
            complete=complete, page=relative, verification=relative.replace('comparison.html', 'verification.json')))
        save(parent / 'verification.json', verification)


def report(args):
    manifest = load(args.output / 'manifest.json'); check_frozen(args, manifest)
    sources, pairs, sections = [], [], []
    for stage in ('final', 'smoke'):
        size, spp = QUALITY[stage]; masks = roi_masks(size)
        for _, seed in first.jobs(stage):
            available = [mode for mode in MODES if (folder(args, stage, mode, seed) / 'result.json').exists()]
            fd = args.output / 'figures' / (f'seed{seed}' if stage == 'final' else f'smoke_seed{seed}')
            images, labels, rows = [], [], ''
            for mode in available:
                d = folder(args, stage, mode, seed); result = verified(args, stage, mode, seed)
                rgb = pfm(d / 'render.pfm') @ Y
                rois = {name: float(rgb[mask].mean()) for name, mask in masks.items()}
                sources.append(dict(stage=stage, mode=mode, seed=seed, roi=rois, render_seconds=result['render_seconds']))
                images.append(d / 'render.png'); labels.append(LABELS[mode] + f'\n{result["render_seconds"]:.2f} s')
                rows += (f'<tr><td>{LABELS[mode]}</td><td>{rois["overlap"]:.6f}</td><td>{rois["left_single"]:.6f}</td>'
                         f'<td>{rois["right_single"]:.6f}</td><td>{result["render_seconds"]:.2f}</td>'
                         f'<td><a href="{url(args,d/"render.png")}">PNG</a> · <a href="{url(args,d/"scene.json")}">配置</a></td></tr>')
            section = f'<h2>{stage} · {size}×{size} · {spp} spp · seed {seed}</h2>'
            section += picture_grid(args, images, labels, fd / 'appearance.png', 5)
            section += '<table><tr><th>模式</th><th>重叠区</th><th>左单层</th><th>右单层</th><th>秒</th><th>文件</th></tr>' + rows + '</table>'
            if MODE in available:
                maps, labels, rows = [], [], ''
                for ref in REFERENCES:
                    pair = compare(args, seed, folder(args, stage, ref, seed), folder(args, stage, MODE, seed),
                                   stage + '_continued_vs_' + ref)
                    pair.update(stage=stage, reference=ref, test=MODE); pairs.append(pair)
                    maps.append(args.output / pair['error_map'])
                    labels.append('Continued vs ' + LABELS[ref] + f'\nFLIP {pair["flip"]["mean"]:.6f}')
                    rows += (f'<tr><td>{LABELS[ref]}</td><td>{pair["flip"]["mean"]:.6f}</td>'
                             f'<td>{pair["linear_difference_rms"]:.6f}</td><td>{pair["estimated_difference_noise_rms"]:.6f}</td></tr>')
                section += '<h3>续训结果的 FLIP</h3>' + picture_grid(args, maps, labels, fd / 'flip.png', 4)
                section += '<table><tr><th>参考</th><th>FLIP</th><th>线性差值 RMS</th><th>差值噪声 RMS 估计</th></tr>' + rows + '</table>'
            fig, ax = plt.subplots(figsize=(10, 3.6))
            for mode in available:
                ax.plot((pfm(folder(args, stage, mode, seed) / 'render.pfm') @ Y)[size//2], label=LABELS[mode])
            ax.set(xlabel='Pixel x', ylabel='Linear transmittance', ylim=(0,1.03), title='Center-row profile')
            ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(fd / 'profile.png', dpi=130); plt.close(fig)
            section += f'<img src="{url(args,fd/"profile.png")}">'; sections.append(section)
    aggregate = []
    for mode in MODES:
        rows = [r for r in sources if r['stage']=='final' and r['mode']==mode]
        if len(rows)==len(SEEDS):
            aggregate.append(dict(mode=mode, overlap_transmittance=float(np.mean([r['roi']['overlap'] for r in rows])),
                single_transmittance=float(np.mean([r['roi'][key] for r in rows for key in ('left_single','right_single')])),
                average_render_seconds=float(np.mean([r['render_seconds'] for r in rows]))))
    complete = sum(r['stage']=='final' and r['mode']==MODE for r in sources)==len(SEEDS)
    training_html, training = training_section(args)
    old_pairs = load(args.previous / 'metrics.json')['comparisons']
    flip_progress = []
    for ref in ('gpis_global', 'gpis_renewal_plus', 'global_conditional'):
        old = [p['flip']['mean'] for p in old_pairs if p['stage']=='final' and p['reference']==ref]
        new = [p['flip']['mean'] for p in pairs if p['stage']=='final' and p['reference']==ref]
        if len(old)==len(SEEDS) and len(new)==len(SEEDS):
            flip_progress.append(dict(reference=ref, previous=float(np.mean(old)), continued=float(np.mean(new))))
    save(args.output / 'metrics.json', dict(complete=complete, sources=sources, comparisons=pairs, aggregate=aggregate,
        flip_progress=flip_progress, training=training, correction_cap=args.cap, collision_gradient_correction=False))
    with (args.output / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['mode','overlap_transmittance','single_transmittance','average_render_seconds'])
        writer.writeheader(); writer.writerows(aggregate)
    rows = ''.join(f'<tr><td>{LABELS[r["mode"]]}</td><td>{r["overlap_transmittance"]:.6f}</td>'
                   f'<td>{r["single_transmittance"]:.6f}</td><td>{r["average_render_seconds"]:.2f}</td></tr>' for r in aggregate)
    flip_rows = ''.join(f'<tr><td>{LABELS[r["reference"]]}</td><td>{r["previous"]:.6f}</td><td>{r["continued"]:.6f}</td></tr>' for r in flip_progress)
    page = ('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>双平面：单点网络续训对比</title><style>body{max-width:1500px;margin:30px auto;padding:0 20px;font:16px/1.65 system-ui;color:#243346;background:#f5f6f8}img{width:100%;height:auto;background:white}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}a{color:#235e91}.note{padding:16px;background:#e7edf5}h2{margin-top:32px}</style>
<h1>双平面：单点神经网络继续训练</h1><div class="note">同一数据、同一网络结构、同一数值教师和修正上限 C=2。只更换模型权重，保留碰撞点梯度分布。64×64、1024 spp、原有两个种子；σ=0.02、ℓ=0.2。</div>
''' + '<p>续训与初次神经网络使用同一冻结渲染器。PFM 用于线性透射率；显示图使用相同 filmic，不降噪、不单独调曝光。'
        'FLIP 使用本地可执行程序、原始分辨率及 67 PPD。旧 GPIS 参考耗时来自先前运行，仅供参考。</p>'
        '<table><tr><th>模式</th><th>重叠区均值</th><th>单层区均值</th><th>平均渲染秒数</th></tr>' + rows + '</table>'
        '<h2>两种子平均 FLIP：续训前后</h2><table><tr><th>参考</th><th>初次训练</th><th>继续训练</th></tr>' + flip_rows + '</table>'
        f'<p><a href="{url(args,args.previous/"comparison.html")}">初次训练结果</a> · <a href="manifest.json">版本校验</a> · '
        '<a href="metrics.json">全部指标</a> · <a href="summary.csv">CSV</a> · <a href="progress.json">进度</a></p>'
        + training_html + ''.join(sections) + '</html>')
    (args.output / 'comparison.html').write_text(page, encoding='utf-8')
    save(args.output / 'verification.json', dict(passed=True, complete=complete, render_records=len(sources),
        new_render_records=sum(r['mode']==MODE for r in sources), native_flip_records=len(pairs),
        same_native_renderer_as_previous=True, same_dataset=True, shared_physical_scene=True,
        original_outputs_preserved=True, collision_gradient_correction=False,
        local_links=verify_links(args.output / 'comparison.html'), html_sha256=sha(args.output / 'comparison.html'),
        metrics_sha256=sha(args.output / 'metrics.json')))
    link_parents(args, complete)
    print('REPORT', len(sources), 'images;', len(pairs), 'FLIP records', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','smoke','run','report','status'))
    parser.add_argument('--base', type=Path, default=BASE)
    parser.add_argument('--previous', type=Path, default=PREVIOUS)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--parallel-jobs', type=int, choices=(1,2), default=2)
    parser.add_argument('--cap', type=float, default=2.)
    parser.add_argument('--flip', type=Path, default=DEFAULT_FLIP)
    args = parser.parse_args()
    for name in ('base','previous','output','flip'):
        setattr(args, name, getattr(args,name).resolve())
    args.data = (args.data or args.previous / 'data').resolve()
    assert args.threads>0 and args.threads*args.parallel_jobs <= (os.cpu_count() or 1)
    assert args.cap>=1 and np.isfinite(args.cap) and args.flip.is_file()
    if args.command=='status':
        print(json.dumps(load(args.output / 'progress.json'),indent=2)); return
    prepare(args)
    if args.command in ('prepare','report'):
        first.status(args); report(args); return
    lock = args.output / 'runner.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor,'w') as stream:
        json.dump(dict(pid=os.getpid(),started_at=now()),stream)
    try:
        for stage in (('smoke',) if args.command=='smoke' else ('smoke','final')):
            if len(first.jobs(stage))==1 or args.parallel_jobs==1:
                for job in first.jobs(stage):
                    first.render(args,*job); first.status(args); report(args)
            else:
                with ThreadPoolExecutor(max_workers=args.parallel_jobs) as pool:
                    pending={pool.submit(first.render,args,*job,False):job for job in first.jobs(stage)}
                    while pending:
                        first.status(args,active=[str(folder(args,job[0],MODE,job[1])) for job in pending.values()])
                        done,_=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
                        for future in done:
                            future.result();pending.pop(future);report(args)
                first.status(args)
        print('COMPLETE',args.command,flush=True)
    except Exception as error:
        failure=dict(error=str(error),traceback=traceback.format_exc())
        save(args.output/'failure.json',failure);first.status(args,failure=failure);raise
    finally:
        lock.unlink()


if __name__=='__main__':
    main()
