"""Append matched global_conditional renders to a frozen shaderball experiment."""
from __future__ import annotations

import argparse
import copy
import csv
import html
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from compare_shaderball_fixed_ratio import (
    ROOT, block_mean, build_report, filmic, load, read_pfm, render_rows,
    roi_stats, save, signed_color, tag, verified, write_pfm, write_png,
)
from render_shaderball_neural import sha256

MODE = 'global_conditional'
OFFSET = 3000000
TRACKING = ('tracking_step', 'tracking_step_scale', 'tracking_age_scale',
            'tracking_max_expected_events')


def retry_key(ell, seed):
    return f'ell{tag(ell)}_seed{seed}'


def planned_seed(ell, seed, retries):
    return retries.get(retry_key(ell, seed), {}).get('render_seed', seed + OFFSET)


def failed_attempts(output, reference, retries):
    attempts = []
    for ell in reference['lengths']:
        for seed in reference['seed_ids']:
            folders = [output / f'ell{tag(ell)}' / f'seed{seed}']
            retry = retries.get(retry_key(ell, seed))
            if retry:
                folders.append(output / f'ell{tag(ell)}' / f'seed{seed}_rng{retry["render_seed"]}')
            for folder in folders:
                if not (folder / 'result.json').exists():
                    continue
                record = load(folder / 'result.json')
                if record.get('status') != 'failed':
                    continue
                log = (folder / 'render.log').read_text(encoding='utf-8')
                checkpoints = re.findall(r'Completed\s+(\d+)/(\d+)\s+spp', log)
                error = (log.split('unrecoverable error:', 1)[-1].strip()
                         if 'unrecoverable error:' in log else record['error'])
                attempts.append(dict(ell=ell, seed_id=seed, render_seed=record['render_seed'],
                                     folder=str(folder), status='failed', error=error,
                                     requested_spp=reference['spp_per_seed'],
                                     last_checkpoint_spp=int(checkpoints[-1][0]) if checkpoints else 0))
    return attempts


def active_revision(main):
    path = main / MODE / 'active_revision.json'
    return load(path)['revision'] if path.exists() else None


def run_output(main, revision):
    return main / MODE / revision if revision else main / MODE


def write_status(main, output, data):
    save(output / 'batch_status.json', data)
    if output != main / MODE:
        save(main / MODE / 'batch_status.json', dict(data, revision=output.name))


def prepare(main, revision=None):
    reference = load(main / 'manifest.json')
    executable = ROOT / 'build/Release/tungsten.exe'
    template_path = ROOT / 'gpis-scenes/shaderball/shaderball-stationary-27-global-conditional.json'
    template = load(template_path)
    settings = {k: template['media'][0][k] for k in TRACKING}
    scenes = {}
    for ell in reference['lengths']:
        base = load(main / 'configs' / f'ell{tag(ell)}_classic_global.json')
        scene = copy.deepcopy(base)
        scene['media'][0].update(type=MODE, **settings)
        # The renderer's existing adaptive majorant is also its default.
        restored = copy.deepcopy(scene)
        restored['media'][0]['type'] = 'classic_global'
        for k in TRACKING:
            if k in base['media'][0]:
                restored['media'][0][k] = base['media'][0][k]
            else:
                restored['media'][0].pop(k)
        assert restored == base
        expected_medium = copy.deepcopy(template['media'][0])
        expected_medium['gaussian_process']['covariance'].update(
            sigma=ell * reference['ratio'], lengthScale=ell)
        assert scene['media'][0] == expected_medium
        for k in ['primitives', 'bsdfs']:
            assert scene[k] == template[k]
        assert scene['camera']['resolution'] == reference['resolution']
        assert scene['renderer']['spp'] == reference['spp_per_seed']
        scenes[ell] = scene
    manifest = dict(mode=MODE, seed_offset=OFFSET, reference_manifest=reference,
                    source_scene=str(template_path), source_scene_sha256=sha256(template_path),
                    tracking_majorant='adaptive', tracking_settings=settings,
                    executable_sha256=reference['executable_sha256'],
                    note='Existing latest-observation conditional transport, unchanged evaluator. '
                         'Matched procedural mean, covariance, material, camera, lighting, integrator and sampling settings.')
    if revision:
        assert revision == 'endpoint_sync'
        executable = ROOT / 'build/Release/tungsten-endpoint-sync.exe'
        backup = ROOT / 'build/Release/tungsten-before-endpoint-sync.exe'
        assert sha256(backup) == reference['executable_sha256']
        manifest.update(revision=revision, executable=str(executable), executable_sha256=sha256(executable),
                        reference_executable_backup=str(backup),
                        correction='Reuse the renewed observation position for both geometry-endpoint path states. '
                                   'The conditional evaluator and direction-change guard are unchanged.',
                        corrected_source_sha256=sha256(ROOT / 'src/core/media/GlobalConditionalMedium.cpp'))
    return scenes, manifest


def image_card(title, png, pfm=None, block=None):
    block = block or png.replace('.png', '_block2.png')
    download = f' · <a href="{pfm}">线性 PFM</a>' if pfm else ''
    return (f'<figure><figcaption>{title}{download}</figcaption><a href="{png}">'
            f'<img class="switchable" src="{png}" data-raw="{png}" data-block="{block}" '
            f'alt="{title}"></a></figure>')


def report(main):
    revision = active_revision(main)
    output = run_output(main, revision)
    relative_output = output.relative_to(main)
    scenes, manifest = prepare(main, revision)
    assert load(output / 'manifest.json') == manifest, 'Conditional provenance differs.'
    reference = manifest['reference_manifest']
    retry_path = output / 'retry_plan.json'
    retries = load(retry_path) if retry_path.exists() else {}
    failures = failed_attempts(output, reference, retries)
    historical_failures = []
    if revision:
        old_retry_path = main / MODE / 'retry_plan.json'
        old_retries = load(old_retry_path) if old_retry_path.exists() else {}
        historical_failures = failed_attempts(main / MODE, reference, old_retries)
    rows = load(output / 'render_rows.json') if (output / 'render_rows.json').exists() else []
    keys = [(r['ell'], r['seed_id']) for r in rows]
    assert len(keys) == len(set(keys))
    assert set(keys) <= {(ell, seed) for ell in reference['lengths'] for seed in reference['seed_ids']}
    expected = len(reference['lengths']) * len(reference['seed_ids'])
    summary = dict(manifest=manifest, retry_plan=retries, failed_attempts=failures,
                   previous_failed_attempts=historical_failures,
                   render_rows=(relative_output / 'render_rows.json').as_posix(),
                   completed_renders=len(rows), expected_renders=expected, cases=[],
                   statistics='Linear Rec.709 luminance. Independent mode seeds. '
                              'A single seed does not estimate a confidence interval.',
                   display='Linear averaging before the common filmic map; no denoising.', difference_scale=.15)
    reference_rows = render_rows(main)
    sections, table, csv_rows = [], [], []
    for ell in reference['lengths']:
        case = dict(ell=ell, sigma=ell * reference['ratio'], modes={}, comparisons={})
        stacks, images, differences = {}, [], []
        selected = {m: [r for r in reference_rows if r['ell'] == ell and r['mode'] == m]
                    for m in ['classic_global', 'gpis_global']}
        selected[MODE] = [r for r in rows if r['ell'] == ell]
        if ell == .1 and (main / 'neural_ell0p1/render_rows.json').exists():
            selected['neural'] = load(main / 'neural_ell0p1/render_rows.json')
        for mode in ['classic_global', MODE, 'gpis_global', 'neural']:
            if mode not in selected:
                continue
            entries = selected[mode]
            if not entries:
                failed = [entry for entry in failures if entry['ell'] == ell] if mode == MODE else []
                if failed:
                    attempts = []
                    for entry in failed:
                        log = (Path(entry['folder']) / 'render.log').relative_to(main).as_posix()
                        attempts.append(f'<p>种子 {entry["render_seed"]}：最近检查点 '
                                        f'{entry["last_checkpoint_spp"]}/{entry["requested_spp"]} spp '
                                        f'· <a href="{log}">错误日志</a></p>')
                    images.append(f'<figure><figcaption>{mode} · 未完成</figcaption>'
                                  '<p>渲染器报错，未生成完整图像。</p>' + ''.join(attempts) +
                                  f'<p><code>{html.escape(failed[-1]["error"])}</code></p></figure>')
                else:
                    images.append(f'<figure><figcaption>{mode}</figcaption><p>等待渲染</p></figure>')
                continue
            if mode != MODE:
                assert sorted(r['seed_id'] for r in entries) == sorted(reference['seed_ids'])
            scene = scenes[ell] if mode == MODE else load(main / 'configs' / f'ell{tag(ell)}_{mode}.json')
            offset = {'classic_global': 0, 'gpis_global': 1000000, 'neural': 2000000, MODE: OFFSET}[mode]
            for row in entries:
                assert row['spp'] == reference['spp_per_seed']
                render_seed = planned_seed(ell, row['seed_id'], retries) if mode == MODE else row['seed_id'] + offset
                executable_hash = manifest['executable_sha256'] if mode == MODE else reference['executable_sha256']
                assert verified(Path(row['folder']), scene, render_seed, executable_hash)
            stack = np.stack([read_pfm(Path(r['folder']) / 'render.pfm') for r in entries])
            assert stack.shape[1:] == (*reference['resolution'][::-1], 3)
            stacks[mode] = stack
            if mode == MODE:
                rel = relative_output / 'averages' / f'ell{tag(ell)}' / MODE
                mean = stack.mean(axis=0)
                write_pfm(main / rel.with_suffix('.pfm'), mean)
                write_png(main / rel.with_suffix('.png'), filmic(mean))
                write_png(main / rel.with_name(MODE + '_block2.png'), filmic(block_mean(mean)))
            elif mode == 'neural':
                rel = Path('neural_ell0p1/averages/neural')
            else:
                rel = Path('averages') / f'ell{tag(ell)}' / mode
            png, pfm = rel.with_suffix('.png').as_posix(), rel.with_suffix('.pfm').as_posix()
            spp = sum(r['spp'] for r in entries)
            case['modes'][mode] = dict(seed_ids=[r['seed_id'] for r in entries], spp_total=spp,
                                       render_seeds=[r['render_seed'] for r in entries],
                                       image=png, pfm=pfm, seconds=sum(r['seconds'] for r in entries))
            images.append(image_card(f'{mode} · {spp} spp', png, pfm))
        if MODE in stacks:
            for other in ['classic_global', 'gpis_global', 'neural']:
                if other not in stacks:
                    continue
                rois = {}
                for name, roi in {'full': [0, 0, 1, 1], **reference['rois']}.items():
                    stats = roi_stats(stacks[other], stacks[MODE], roi)
                    stats = {k.replace('classic', 'reference').replace('gpis', 'conditional'): v for k, v in stats.items()}
                    rois[name] = stats
                    ci = stats.get('difference_ci95')
                    confidence = f'[{ci[0]:+.5f}, {ci[1]:+.5f}]' if ci else '单个 seed；未估计'
                    if name != 'full':
                        table.append(f'<tr><td>{ell:g}</td><td>{other}</td><td>{name}</td>'
                                     f'<td>{stats["reference_mean"]:.5f}</td><td>{stats["conditional_mean"]:.5f}</td>'
                                     f'<td>{stats["relative_difference"]:+.2%}</td><td>{confidence}</td></tr>')
                    csv_rows.append([ell, other, name, stats['reference_mean'], stats['conditional_mean'],
                                     stats['relative_difference'], stats['difference'], *(ci or ['', ''])])
                case['comparisons'][other] = rois
                delta = (stacks[MODE].mean(axis=0) - stacks[other].mean(axis=0)) @ np.array([.2126, .7152, .0722])
                rel = relative_output / 'averages' / f'ell{tag(ell)}' / f'conditional_minus_{other}'
                write_png(main / rel.with_suffix('.png'), signed_color(delta))
                write_png(main / rel.with_name(rel.name + '_block2.png'), signed_color(block_mean(delta[..., None])[..., 0]))
                differences.append(image_card(f'global_conditional − {other}', rel.with_suffix('.png').as_posix()))
        original_diff = f'averages/ell{tag(ell)}/difference.png'
        differences.append(image_card('GPIS − classic_global', original_diff))
        section_id = 'neural-ell0p1' if ell == .1 and 'neural' in selected else f'conditional-ell{tag(ell)}'
        sections.append(f'<section id="{section_id}"><h2>ℓ = {ell:g} · σ = {case["sigma"]:g}</h2>'
                        f'<div class="images conditional-images" style="--columns:{len(images)}">' + ''.join(images) + '</div>'
                        '<details><summary>线性亮度差值 · 共用 ±0.15 色标</summary><div class="images conditional-differences">'
                        + ''.join(differences) + '</div></details></section>')
        summary['cases'].append(case)
    save(output / 'summary.json', summary)
    if revision:
        save(main / MODE / 'summary.json', summary)
    with (output / 'roi_statistics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ell', 'reference', 'roi', 'reference_mean', 'conditional_mean', 'relative_difference',
                         'difference', 'difference_ci95_low', 'difference_ci95_high'])
        writer.writerows(csv_rows)
    retry_note = ''
    if retries:
        attempts = []
        for entry in retries.values():
            log = (Path(entry['failed_folder']) / 'render.log').relative_to(main).as_posix()
            attempts.append(f'ℓ=σ={entry["ell"]:g}：原种子 {entry["failed_render_seed"]} 未完成，'
                            f'重试种子 {entry["render_seed"]}（<a href="{log}">原错误日志</a>）')
        retry_note = '<p>重试记录：' + '；'.join(attempts) + f'。<a href="{relative_output.as_posix()}/retry_plan.json">种子记录</a>。未完成尝试不计入图像。</p>'
    revision_note = ''
    if revision:
        logs = []
        for entry in historical_failures:
            log = (Path(entry['folder']) / 'render.log').relative_to(main).as_posix()
            logs.append(f'<a href="{log}">种子 {entry["render_seed"]} 的原错误日志</a>')
        revision_note = ('<p>端点坐标同步已修复：路径状态复用更新后的观测坐标，求值公式和方向检查保持不变。'
                         '三个尺度均重新渲染，使用原计划种子 23261027。修复前的结果与失败记录保留；'
                         + ' · '.join(logs) + f' · <a href="{relative_output.as_posix()}/previous_run.json">修复前记录</a>。</p>')
    fragment = (f'<section id="global-conditional"><h2>追加 global_conditional</h2><p>已完成 {len(rows)} / {expected} 张；'
                f'{reference["resolution"][0]}×{reference["resolution"][1]}，每张 {reference["spp_per_seed"]} spp。'
                '采用项目现有 latest-observation 条件输运与 adaptive tracking；相同 procedural knob、协方差、材质、相机和光照。'
                '每次 GP 碰撞和几何端点更新最近一次值与梯度观测，各模式使用独立随机种子。</p>'
                '<p>neural 沿用现有模型及 dx=0.01 的三线性 SDF；其余模式使用 procedural mean。'
                '<a href="neural_ell0p1/comparison.html">原 neural 对照与说明</a> · '
                f'<a href="{relative_output.as_posix()}/manifest.json">新增配置清单</a> · <a href="{relative_output.as_posix()}/summary.json">新增统计</a> · '
                f'<a href="{relative_output.as_posix()}/roi_statistics.csv">新增区域统计 CSV</a></p>' + revision_note + retry_note +
                (f'<p>共 {len(failures)} 次失败尝试；失败样本未计入完成数、图像或统计。'
                 '下方保留各次检查点与错误日志。</p>' if failures else '') + '</section>' + ''.join(sections) +
                '<section><h2>Global conditional 区域亮度</h2><p>百分比为 global_conditional / reference − 1。'
                '单个 seed 不估计重复采样置信区间；图像未做降噪。</p><table><thead><tr><th>ℓ</th><th>reference</th>'
                '<th>区域</th><th>reference</th><th>global_conditional</th><th>相对差</th><th>差值 95% 区间</th>'
                '</tr></thead><tbody>' + ''.join(table) + '</tbody></table></section>'
                '<style>.images.conditional-images{grid-template-columns:repeat(var(--columns),minmax(0,1fr))}'
                'details{margin-top:16px}summary{cursor:pointer}.conditional-differences{margin-top:12px}'
                '@media(max-width:850px){.images.conditional-images{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>')
    (output / 'section.html').write_text(fragment, encoding='utf-8')
    if revision:
        (main / MODE / 'section.html').write_text(fragment, encoding='utf-8')
    build_report(main)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', type=Path, default=ROOT / 'outputs/shaderball_fixed_ratio/res256_spp256')
    parser.add_argument('--threads', type=int, default=28)
    parser.add_argument('--revision', choices=['endpoint_sync'], help='Start or resume all scales with the endpoint-position repair; retain the previous run.')
    parser.add_argument('--lengths', type=float, nargs='+', help='Render only these planned scales; preserve the complete comparison.')
    parser.add_argument('--retry-seed', type=int, help='One explicit replacement seed for a failed single-scale, single-seed render; retain the failed attempt.')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    main_path = args.main.resolve()
    revision = args.revision or active_revision(main_path)
    output = run_output(main_path, revision)
    scenes, manifest = prepare(main_path, revision)
    if (output / 'manifest.json').exists():
        assert load(output / 'manifest.json') == manifest, 'Existing conditional inputs differ.'
    save(output / 'manifest.json', manifest)
    if revision:
        if not (output / 'previous_run.json').exists():
            previous = {}
            for name in ['manifest', 'render_rows', 'retry_plan', 'summary', 'batch_status']:
                path = main_path / MODE / f'{name}.json'
                if path.exists():
                    previous[name] = load(path)
            save(output / 'previous_run.json', previous)
        save(main_path / MODE / 'active_revision.json', dict(revision=revision))
    for ell, scene in scenes.items():
        save(main_path / 'configs' / f'ell{tag(ell)}_{MODE}.json', scene)
    if args.report_only or not args.render:
        report(main_path)
        return
    reference = manifest['reference_manifest']
    executable = Path(manifest.get('executable', ROOT / 'build/Release/tungsten.exe'))
    assert sha256(executable) == manifest['executable_sha256'], 'Renderer differs from the frozen run.'
    lengths = reference['lengths'] if args.lengths is None else args.lengths
    assert lengths and set(lengths) <= set(scenes), 'Requested scale is absent from the frozen experiment.'
    retry_path = output / 'retry_plan.json'
    retries = load(retry_path) if retry_path.exists() else {}
    if args.retry_seed is not None:
        assert len(lengths) == len(reference['seed_ids']) == 1
        ell, seed = lengths[0], reference['seed_ids'][0]
        assert 0 <= args.retry_seed < 2**32 and args.retry_seed != seed + OFFSET
        failed_folder = output / f'ell{tag(ell)}' / f'seed{seed}'
        assert not verified(failed_folder, scenes[ell], seed + OFFSET, manifest['executable_sha256'])
        log = (failed_folder / 'render.log').read_text(encoding='utf-8')
        assert 'unrecoverable error' in log and 'Finished render.' not in log
        entry = dict(ell=ell, seed_id=seed, render_seed=args.retry_seed, failed_render_seed=seed + OFFSET,
                     failed_folder=str(failed_folder), reason='Existing renderer aborted; explicit retry without evaluator or scene changes.')
        key = retry_key(ell, seed)
        assert key not in retries or retries[key] == entry, 'A different retry is already recorded.'
        retries[key] = entry
        save(retry_path, retries)
        failed_record = load(failed_folder / 'result.json')
        failed_record.update(status='failed', error=log.split('unrecoverable error:', 1)[-1].strip())
        save(failed_folder / 'result.json', failed_record)
    rows = load(output / 'render_rows.json') if (output / 'render_rows.json').exists() else []
    report(main_path)
    try:
        for ell in sorted(set(lengths), reverse=True):
            for seed in reference['seed_ids']:
                scene = scenes[ell]
                render_seed = planned_seed(ell, seed, retries)
                suffix = '' if render_seed == seed + OFFSET else f'_rng{render_seed}'
                folder = output / f'ell{tag(ell)}' / f'seed{seed}{suffix}'
                write_status(main_path, output, dict(status='running', current=dict(ell=ell, mode=MODE),
                                                    completed=len(rows), log=str(folder / 'render.log')))
                if verified(folder, scene, render_seed, manifest['executable_sha256']):
                    record = load(folder / 'result.json')
                else:
                    save(folder / 'scene.json', scene)
                    assert sha256(executable) == manifest['executable_sha256']
                    command = [str(executable), '-t', str(args.threads),
                               '-s', str(render_seed), '-d', str(folder), str(folder / 'scene.json')]
                    print(f'Render {MODE} ell=sigma={ell:g}, {reference["resolution"]}, {reference["spp_per_seed"]} spp', flush=True)
                    start = time.monotonic()
                    with (folder / 'render.log').open('w', encoding='utf-8') as log:
                        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    if process.returncode:
                        raise RuntimeError(f'Render failed: {folder / "render.log"}')
                    assert sha256(executable) == manifest['executable_sha256']
                    record = dict(ell=ell, sigma=ell * reference['ratio'], mode=MODE, seed_id=seed,
                                  render_seed=render_seed, spp=reference['spp_per_seed'], seconds=time.monotonic()-start,
                                  folder=str(folder), command=command, executable_sha256=manifest['executable_sha256'])
                    save(folder / 'result.json', record)
                    if not verified(folder, scene, render_seed, manifest['executable_sha256']):
                        record.update(status='failed', error='Missing valid complete render; see render.log.')
                        save(folder / 'result.json', record)
                        raise RuntimeError(f'Invalid render: {folder / "render.log"}')
                rows = [r for r in rows if (r['ell'], r['seed_id']) != (ell, seed)] + [record]
                save(output / 'render_rows.json', rows)
                report(main_path)
                print(f'Completed ell={ell:g} in {record["seconds"]:.2f}s', flush=True)
        complete = len(rows) == len(scenes) * len(reference['seed_ids'])
        write_status(main_path, output, dict(status='complete' if complete else 'partial', completed=len(rows)))
        print(f'Completed {len(rows)} / {len(scenes) * len(reference["seed_ids"])} global_conditional renders.', flush=True)
    except Exception as error:
        write_status(main_path, output, dict(status='failed', completed=len(rows), error=str(error)))
        raise


if __name__ == '__main__':
    main()
