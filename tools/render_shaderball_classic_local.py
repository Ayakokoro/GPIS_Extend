"""Append three classic_local renders to the native shaderball scale sweep.

The existing exporter, 3-sigma density band, grid placement and renderer are
retained. Only the formal 256 x 256 / 256 spp experiment is updated.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time
from urllib.parse import unquote

import numpy as np

import compare_shaderball_fixed_ratio as b
from render_shaderball_neural import sha256

ROOT = b.ROOT
MAIN = ROOT/'outputs/shaderball_fixed_ratio/res256_spp256'
OUT = MAIN/'classic_local'
SCENES = ROOT/'gpis-scenes/shaderball'
SOURCE_FIELD = SCENES/'knob-l0.02-s0.02-dx0.01.nvdb'
SOURCE_BAKE = SCENES/'knob-l0.02-s0.02-dx0.01-bake-source.json'
TEMPLATE = SCENES/'knob-classic-local-l0.02-s0.02.json'
EXE = ROOT/'build/Release/tungsten-before-endpoint-sync.exe'
BAKER = ROOT/'build/Release/obj2json_macrofacet.exe'
OFFSET, THREADS = 4000000, 8
MODES = ('classic_global', 'classic_local', 'global_conditional', 'gpis_global', 'neural')
LABELS = dict(classic_global='classic_global', classic_local='classic_local',
              global_conditional='global_conditional', gpis_global='GPIS global', neural='neural')


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(text, encoding='utf-8')
    for attempt in range(20):
        try: os.replace(temp, path); return
        except PermissionError:
            if attempt==19: raise
            time.sleep(.25)


def save(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False)+'\n')


def inspect_field(path):
    """Read v32 uncompressed metadata and hash complete stored grid blocks."""
    result = []
    with path.open('rb') as stream:
        while header := stream.read(16):
            if len(header)!=16 or header[:8] not in (b'NanoVDB0', b'NanoVDB2'):
                raise ValueError('Unexpected NanoVDB file header')
            _, version, count, codec = struct.unpack('<QIHH', header)
            if version>>21!=32 or codec!=0: raise ValueError('Unsupported NanoVDB encoding')
            metadata = []
            for _ in range(count):
                data = stream.read(176)
                grid_size, file_size, _, voxels = struct.unpack_from('<4Q', data)
                name_size, = struct.unpack_from('<I', data, 136)
                name = stream.read(name_size).rstrip(b'\0').decode()
                grid_type, = struct.unpack_from('<I', data, 32)
                if grid_type!=1 or grid_size!=file_size: raise ValueError('Expected uncompressed float grid')
                metadata.append(dict(name=name, grid_bytes=grid_size, active_voxels=voxels,
                    index_bounds=list(struct.unpack_from('<6i', data, 88)),
                    world_bounds=list(struct.unpack_from('<6d', data, 40)),
                    voxel_size=list(struct.unpack_from('<3d', data, 112))))
            for grid in metadata:
                start = stream.tell()
                data = stream.read(736)
                if data[:8] not in (b'NanoVDB0', b'NanoVDB1') or data[40:296].split(b'\0')[0].decode()!=grid['name']:
                    raise ValueError('Unexpected NanoVDB grid header')
                grid['index_to_world_matrix'] = list(struct.unpack_from('<9d', data, 384))
                grid['index_to_world_translation'] = list(struct.unpack_from('<3d', data, 528))
                root_offset, = struct.unpack_from('<q', data, 696)
                stream.seek(start+672+root_offset+28)
                grid['background'], grid['minimum'], grid['maximum'] = struct.unpack('<3f', stream.read(12))
                if not all(math.isfinite(grid[k]) for k in ('background', 'minimum', 'maximum')):
                    raise ValueError('Nonfinite grid metadata')
                stream.seek(start)
                digest, remaining = hashlib.sha256(), grid['grid_bytes']
                while remaining:
                    chunk = stream.read(min(remaining, 1024*1024))
                    if not chunk: raise ValueError('Truncated grid')
                    digest.update(chunk); remaining -= len(chunk)
                grid['sha256'] = digest.hexdigest()
                result.append(grid)
    return {grid['name']:grid for grid in result}


def field_path(ell):
    return SOURCE_FIELD if ell==.02 else OUT/'fields'/f'ell{b.tag(ell)}'/'knob.nvdb'


def bake_field(ell):
    path = field_path(ell)
    source = b.load(SOURCE_BAKE)
    source['media'][0]['gaussian_process']['covariance'].update(sigma=ell, lengthScale=ell)
    folder = path.parent
    if (folder/'bake_record.json').exists():
        record = b.load(folder/'bake_record.json')
        if record['field_sha256']!=sha256(path) or b.load(folder/'source.json')!=source:
            raise RuntimeError('Existing bake differs')
        return record
    if path.exists(): raise RuntimeError('Unrecorded bake retained: '+str(path))
    save(folder/'source.json', source)
    command = [str(BAKER), '--format', 'nvdb', '--voxel-size', '.01', '--padding-sigma', '3',
               '--include-density', '--verbose', str(folder/'source.json'), str(path.with_suffix(''))]
    print(f'BAKE ell=sigma={ell:g}', flush=True)
    start = time.monotonic()
    with (folder/'bake.log').open('wb') as log:
        proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    if proc.returncode: raise RuntimeError('Bake failed: '+str(folder/'bake.log'))
    record = dict(ell=ell, sigma=ell, command=command, seconds=time.monotonic()-start,
                  returncode=proc.returncode, baker_sha256=sha256(BAKER), field_sha256=sha256(path))
    save(folder/'bake_record.json', record)
    print(f'BAKED ell={ell:g} seconds={record["seconds"]:.1f}', flush=True)
    return record


def make_scene(ell):
    scene = b.load(MAIN/'configs'/f'ell{b.tag(ell)}_classic_global.json')
    original = scene['media'][0]
    medium = copy.deepcopy(b.load(TEMPLATE)['media'][0])
    medium.update(name=original['name'], max_bounces=original['max_bounces'],
                  covariance=copy.deepcopy(original['gaussian_process']['covariance']),
                  material=copy.deepcopy(original['material']))
    medium['field']['file'] = field_path(ell).as_posix()
    scene['media'] = [medium]
    return scene


def references():
    rows = [(row['mode'], row) for row in b.render_rows(MAIN)]
    revision = b.load(MAIN/'global_conditional/active_revision.json')['revision']
    rows += [('global_conditional', row) for row in b.load(MAIN/'global_conditional'/revision/'render_rows.json')]
    rows += [('neural', row) for row in b.load(MAIN/'neural_ell0p1/render_rows.json')]
    return rows


def prepare():
    reference = b.load(MAIN/'manifest.json')
    if reference['resolution']!=[256,256] or reference['spp_per_seed']!=256 or reference['ratio']!=1:
        raise RuntimeError('Unexpected formal batch configuration')
    if sha256(EXE)!=reference['executable_sha256']: raise RuntimeError('Frozen renderer differs')
    sidecar = b.load(SOURCE_FIELD.with_suffix('.nvdb.json'))
    if sha256(SOURCE_FIELD)!=sidecar['nvdb_sha256']: raise RuntimeError('Existing field hash differs')
    original_grids = inspect_field(SOURCE_FIELD)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(bake_field, ell) for ell in reference['lengths'] if ell!=.02]:
            future.result()
    fields = {}
    for ell in reference['lengths']:
        path = field_path(ell)
        grids = original_grids if ell==.02 else inspect_field(path)
        if set(grids)!={'sdf','density','alpha'}: raise RuntimeError('Unexpected grids')
        for name, grid in grids.items():
            for key in ('active_voxels','index_bounds','voxel_size','index_to_world_matrix','index_to_world_translation'):
                if grid[key]!=original_grids[name][key]: raise RuntimeError('Grid layout mismatch: '+key)
        if grids['sdf']['sha256']!=original_grids['sdf']['sha256']:
            raise RuntimeError('Baked SDF differs across scales')
        if grids['density']['background']!=0 or grids['density']['minimum']<0:
            raise RuntimeError('Invalid density grid')
        metadata = dict(sigma=ell, length_scale=ell, dx=.01, field=str(path),
            field_sha256=sha256(path), grids=grids, sdf_identical_to_source=True,
            nvdb_to_scene_transform=sidecar['nvdb_to_scene_transform'],
            density_rule='Existing exporter: phi(d/sigma)/(sigma*max(Phi(d/sigma),1e-8)); zero for abs(d)>3*sigma.')
        fields[b.tag(ell)] = metadata
        if ell!=.02: save(path.with_suffix('.nvdb.json'), metadata)
        base = b.load(MAIN/'configs'/f'ell{b.tag(ell)}_classic_global.json')
        if b.load(SOURCE_BAKE)['media'][0]['gaussian_process']['mean']!=base['media'][0]['gaussian_process']['mean']:
            raise RuntimeError('Different procedural mean')
    manifest_path = OUT/'manifest.json'
    if manifest_path.exists():
        manifest = b.load(manifest_path)
        if manifest['reference_manifest']!=reference or manifest['fields']!=fields or manifest['runner_sha256']!=sha256(Path(__file__)):
            raise RuntimeError('Existing local batch differs')
    else:
        preserved = {}
        for _, row in references():
            folder = Path(row['folder'])
            for name in ('render.png','render.pfm','render.log','result.json','scene.json'):
                path = folder/name
                preserved[str(path)] = sha256(path)
        for path in [MAIN/'manifest.json', MAIN.parent/'main/manifest.json',
                     MAIN.parent/'main/comparison.html', MAIN.parent/'main/summary.json']:
            preserved[str(path)] = sha256(path)
        for name in ('comparison.html','summary.json'):
            shutil.copy2(MAIN/name, OUT/('before_classic_local_'+name))
        manifest = dict(mode='classic_local', reference_manifest=reference, fields=fields,
            executable=str(EXE), executable_sha256=sha256(EXE), baker_sha256=sha256(BAKER),
            source_scene=str(TEMPLATE), source_scene_sha256=sha256(TEMPLATE),
            runner_sha256=sha256(Path(__file__)), seed_offset=OFFSET, threads=THREADS,
            preserved_files=preserved, expected_renders=3,
            note='Existing classic_local preset and exporter. Same camera, lighting, material and covariance. All SDF grid bytes match the existing shaderball grid. Density is interpolated from the existing 3-sigma truncated bake rule; this differs from the procedural reference representation.')
        save(manifest_path, manifest)
        (OUT/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    for ell in reference['lengths']:
        path = MAIN/'configs'/f'ell{b.tag(ell)}_classic_local.json'
        scene = make_scene(ell)
        if path.exists() and b.load(path)!=scene: raise RuntimeError('Local configuration differs')
        if not path.exists(): save(path, scene)
    return manifest


def render(ell, seed, manifest):
    folder = OUT/f'ell{b.tag(ell)}'/f'seed{seed}'
    scene, render_seed = make_scene(ell), seed+OFFSET
    if (folder/'result.json').exists():
        if not b.verified(folder, scene, render_seed, manifest['executable_sha256']):
            raise RuntimeError('Incomplete previous attempt retained: '+str(folder))
        return b.load(folder/'result.json')
    save(folder/'scene.json', scene)
    command = [str(EXE), '-t', str(THREADS), '-s', str(render_seed), '-d', str(folder), str(folder/'scene.json')]
    environment = os.environ.copy()
    for key in ('TUNGSTEN_GPIS_CONDITIONAL_PROFILE','TUNGSTEN_GPIS_PROFILE','TUNGSTEN_GPIS_FS_PROFILE'):
        environment.pop(key, None)
    print(f'START classic_local ell={ell:g} 256x256 256spp', flush=True)
    start = time.monotonic()
    with (folder/'render.log').open('wb') as log:
        proc = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=environment, timeout=43200)
    row = dict(ell=ell, sigma=ell, mode='classic_local', seed_id=seed, render_seed=render_seed, spp=256,
        seconds=time.monotonic()-start, folder=str(folder), command=command,
        executable_sha256=manifest['executable_sha256'], returncode=proc.returncode,
        scene_sha256=sha256(folder/'scene.json'))
    save(folder/'result.json', row)
    if proc.returncode or not b.verified(folder, scene, render_seed, manifest['executable_sha256']):
        row.update(status='failed', error='Incomplete or invalid render; see render.log')
        save(folder/'result.json', row)
        raise RuntimeError('Invalid render: '+str(folder))
    print(f'COMPLETE ell={ell:g} seconds={row["seconds"]:.1f}', flush=True)
    return row


def card(label, png, pfm=None):
    block = png.replace('.png','_block2.png')
    return f'<figure><figcaption>{html.escape(label)}'+(f' · <a href="{pfm}">PFM</a>' if pfm else '')+f'</figcaption><a href="{png}"><img class="switchable" src="{png}" data-raw="{png}" data-block="{block}" alt="{html.escape(label)}"></a></figure>'


def report(manifest):
    reference = manifest['reference_manifest']
    rows = b.load(OUT/'render_rows.json') if (OUT/'render_rows.json').exists() else []
    selected_rows = references()+[('classic_local',row) for row in rows]
    sections, comparisons, table = [], [], []
    for ell in reference['lengths']:
        images, stacks, cases = [], {}, {}
        for mode in MODES:
            entries = [row for label,row in selected_rows if label==mode and row['ell']==ell]
            if not entries:
                if mode=='classic_local': images.append('<figure><figcaption>classic_local</figcaption><p>渲染中</p></figure>')
                continue
            arrays = np.stack([b.read_pfm(Path(row['folder'])/'render.pfm') for row in entries])
            if arrays.shape[1:]!=(256,256,3): raise RuntimeError('Resolution mismatch')
            stacks[mode] = arrays
            if mode=='classic_local':
                relative = Path('classic_local/averages')/f'ell{b.tag(ell)}'/mode
                mean = arrays.mean(axis=0)
                b.write_pfm(MAIN/relative.with_suffix('.pfm'), mean)
                b.write_png(MAIN/relative.with_suffix('.png'), b.filmic(mean))
                b.write_png(MAIN/relative.with_name(mode+'_block2.png'), b.filmic(b.block_mean(mean)))
            elif mode=='global_conditional':
                revision = b.load(MAIN/'global_conditional/active_revision.json')['revision']
                relative = Path('global_conditional')/revision/'averages'/f'ell{b.tag(ell)}'/mode
            elif mode=='neural': relative = Path('neural_ell0p1/averages/neural')
            else: relative = Path('averages')/f'ell{b.tag(ell)}'/mode
            images.append(card(LABELS[mode]+' · 256 spp', relative.with_suffix('.png').as_posix(), relative.with_suffix('.pfm').as_posix()))
        diffs = []
        if 'classic_local' in stacks:
            for mode in MODES:
                if mode=='classic_local' or mode not in stacks: continue
                stats = {name:b.roi_stats(stacks[mode],stacks['classic_local'],roi)
                         for name,roi in {'full':[0,0,1,1],**reference['rois']}.items()}
                cases[mode] = stats
                for name in reference['rois']:
                    s = stats[name]
                    table.append(f'<tr><td>{ell:g}</td><td>{mode}</td><td>{name}</td><td>{s["classic_mean"]:.5f}</td><td>{s["gpis_mean"]:.5f}</td><td>{s["relative_difference"]:+.2%}</td></tr>')
                delta_rgb = stacks['classic_local'].mean(axis=0)-stacks[mode].mean(axis=0)
                delta = delta_rgb @ np.array([.2126,.7152,.0722])
                relative = Path('classic_local/averages')/f'ell{b.tag(ell)}'/('local_minus_'+mode)
                b.write_pfm(MAIN/relative.with_suffix('.pfm'), delta_rgb)
                b.write_png(MAIN/relative.with_suffix('.png'), b.signed_color(delta))
                b.write_png(MAIN/relative.with_name(relative.name+'_block2.png'), b.signed_color(b.block_mean(delta[...,None])[...,0]))
                diffs.append(card('classic_local − '+LABELS[mode],relative.with_suffix('.png').as_posix(),relative.with_suffix('.pfm').as_posix()))
        sections.append(f'<section><h2>ℓ = σ = {ell:g}</h2><div class="images local-images" style="--columns:{len(images)}">'+''.join(images)+'</div>'+
            '<details><summary>classic_local 差分 · 统一 ±0.15 线性亮度色标</summary><div class="images">'+''.join(diffs)+'</div></details></section>')
        comparisons.append(dict(ell=ell,sigma=ell,comparisons=cases))
    summary = dict(manifest=manifest, completed_renders=len(rows), expected_renders=3, cases=comparisons,
        statistics='Linear Rec.709 luminance. roi_stats classic_* fields refer to the comparison mode and gpis_* fields to classic_local. Single seed: no repeat-based confidence interval.',
        display='Same filmic map after linear averaging; optional 2x2 linear averaging; no denoising.', difference_scale=.15)
    save(OUT/'summary.json',summary)
    fragment = f'''<section id="classic-local"><h2>追加 classic_local</h2><p>已完成 {len(rows)} / 3 张；每张原生 256×256、256 spp。相同 σ/ℓ=1、相机、灯光、绿色镜面材质及反射上限；各模式使用独立 seed。</p>
<p>classic_local 沿用现有 dx=0.01 的三线性 SDF、单位均值梯度及密度烘焙规则：仅在 |SDF|≤3σ 内保留密度。0.05 与 0.1 分别烘焙对应 σ；三组的 SDF 网格字节完全相同，使用已有坐标变换恢复位置。该网格插值和密度截断与 procedural mean 对照存在表示差异。</p>
<p>GPIS 为原有 global 模式；global_conditional 使用已完成的 endpoint_sync 修复版。neural 保留原有模型与说明。<a href="classic_local/manifest.json">classic_local 配置</a> · <a href="classic_local/summary.json">区域统计</a> · <a href="classic_local/before_classic_local_comparison.html">此前的 10 张结果与条件模式说明</a> · <a href="neural_ell0p1/comparison.html">neural 说明</a></p></section>'''
    # The historical page uses paths relative to the experiment directory.
    fragment = fragment.replace('classic_local/before_classic_local_comparison.html','comparison_before_classic_local.html')
    fragment += ''.join(sections)+'<section><h2>classic_local 区域亮度</h2><p>相对差为 classic_local / reference − 1；单 seed 无重复试验置信区间。</p><table><tr><th>ℓ</th><th>reference</th><th>区域</th><th>reference</th><th>classic_local</th><th>相对差</th></tr>'+''.join(table)+'</table></section>'
    fragment += '<style>.images.local-images{grid-template-columns:repeat(var(--columns),minmax(0,1fr))}details{margin-top:16px}summary{cursor:pointer}@media(max-width:850px){.images.local-images{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>'
    atomic_text(OUT/'section.html', fragment)
    for attempt in range(20):
        try: b.build_report(MAIN); break
        except PermissionError:
            if attempt==19: raise
            time.sleep(.25)
    save(OUT/'batch_status.json',dict(status='complete' if len(rows)==3 else 'running', completed=len(rows), total=3,
                                   updated_utc=datetime.now(timezone.utc).isoformat()))


class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links=[]
    def handle_starttag(self, tag, attrs):
        self.links.extend(v for k,v in attrs if k in ('href','src','data-raw','data-block') and v)


def validate(manifest):
    for path,digest in manifest['preserved_files'].items():
        if sha256(Path(path))!=digest: raise RuntimeError('Preserved artifact changed: '+path)
    if sha256(EXE)!=manifest['executable_sha256']: raise RuntimeError('Renderer changed')
    for field in manifest['fields'].values():
        if sha256(Path(field['field']))!=field['field_sha256']: raise RuntimeError('Field changed')
    rows = b.load(OUT/'render_rows.json')
    if len(rows)!=3 or len({r['ell'] for r in rows})!=3: raise RuntimeError('Incomplete scales')
    for row in rows:
        folder = Path(row['folder'])
        if not b.verified(folder,make_scene(row['ell']),row['seed_id']+OFFSET,manifest['executable_sha256']):
            raise RuntimeError('Render verification failed')
        png = (folder/'render.png').read_bytes()
        if png[:8]!=b'\x89PNG\r\n\x1a\n' or struct.unpack('>II',png[16:24])!=(256,256):
            raise RuntimeError('PNG dimensions mismatch')
    page = MAIN/'comparison.html'
    links = Links(); links.feed(page.read_text(encoding='utf-8'))
    local = [v for v in links.links if not v.startswith(('http:','https:','data:','#'))]
    if any(not (page.parent/unquote(v.split('#')[0])).exists() for v in local): raise RuntimeError('Broken report link')
    summary = b.load(MAIN/'summary.json')
    if summary['total_completed_renders']!=13 or summary['total_expected_renders']!=13:
        raise RuntimeError('Combined render count mismatch')
    save(OUT/'validation.json',dict(state='passed',new_renders=3,total_formal_renders=13,
        preserved_files_checked=len(manifest['preserved_files']),local_links_checked=len(local),
        checks=['256/256 spp, native PNG/PFM and finite nonnegative radiance',
                'matched scene settings and frozen executable', 'byte-identical SDF across three density grids',
                'all original results and main version unchanged', 'report links and total counts']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args = parser.parse_args()
    b.save = save
    OUT.mkdir(parents=True,exist_ok=True)
    if not (MAIN/'comparison_before_classic_local.html').exists():
        shutil.copy2(MAIN/'comparison.html',MAIN/'comparison_before_classic_local.html')
    manifest = prepare()
    report(manifest)
    if args.prepare_only: return
    rows = b.load(OUT/'render_rows.json') if (OUT/'render_rows.json').exists() else []
    if args.report_only:
        validate(manifest); return
    jobs = [(ell,seed) for ell in manifest['reference_manifest']['lengths'] for seed in manifest['reference_manifest']['seed_ids']]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(render,ell,seed,manifest):(ell,seed) for ell,seed in jobs}
        while futures:
            done,_ = wait(futures,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                row=future.result(); del futures[future]
                rows=[r for r in rows if (r['ell'],r['seed_id'])!=(row['ell'],row['seed_id'])]+[row]
                save(OUT/'render_rows.json',rows)
            if done: report(manifest)
            print(f'PROGRESS classic_local {len(rows)}/3',flush=True)
    report(manifest)
    validate(manifest)
    print('COMPLETE: 13 formal images, including 3 classic_local; validation passed',flush=True)


if __name__=='__main__': main()
