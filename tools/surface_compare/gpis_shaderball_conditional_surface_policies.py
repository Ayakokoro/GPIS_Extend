"""Matched shaderball renders for two global_conditional surface policies.

Run with the plot Python: this_script.py all
The original Renewal+ and Global images are immutable reference inputs.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from run import load, save, sha, pfm
from gpis_shaderball_renewal import (
    ROOT, BASE, REPORT, GROUPS, GLOBAL, SEED, EXE, FILES, WEIGHTS, original, verified,
)
from gpis_shaderball_renewal_3d import destination as renewal_destination

OUT = REPORT/'conditional_surface_policies'
POLICIES = ('retain_endpoint', 'restart_exterior')
KEY = 'non_gp_surface_policy'
BASELINE_EXE = ROOT/'build/Release/tungsten_before_shadow_condition_ablation.exe'


def destination(group, policy):
    assert policy in POLICIES
    return BASE/group/'conditional_surface_policies'/('global_conditional_'+policy)/f'seed{SEED}'


def normalized_scene(scene):
    scene = copy.deepcopy(scene)
    scene['renderer'].pop('output_directory')
    scene['media'][0].pop(KEY, None)
    return scene


def policy_scene(group, policy, folder, size=256, spp=128):
    old = verified(original(group, 'global_conditional'))
    scene = copy.deepcopy(old['scene'])
    assert scene['media'][0]['type'] == 'global_conditional'
    assert KEY not in scene['media'][0]
    scene['media'][0][KEY] = policy
    scene['camera']['resolution'] = [size, size]
    scene['renderer'].update(spp=spp, spp_step=min(16, spp), output_directory=str(folder))
    if (size, spp) == (256, 128):
        assert normalized_scene(scene) == normalized_scene(old['scene'])
    return scene, old['source']['result']


def render_one(exe, folder, scene, old, threads, changes):
    exe = Path(exe).resolve()
    exe_hash = sha(exe)
    provenance = dict(old['provenance'], renderer_sha256=exe_hash)
    if (folder/'result.json').exists():
        data = verified(folder)
        assert data['scene'] == scene
        row = data['source']['result']
        assert row['provenance'] == provenance and row['changes'] == changes
        return row
    assert not (folder/'render.pfm').exists(), 'Unvalidated pre-existing render: '+str(folder)
    save(folder/'scene.json', scene)
    begin = time.monotonic()
    with (folder/'render.log').open('w', encoding='utf-8') as log:
        completed = subprocess.run([str(exe), '-t', str(threads), '-s', str(SEED),
            '-d', str(folder), str(folder/'scene.json')], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
    assert completed.returncode == 0, str(folder/'render.log')
    assert sha(exe) == exe_hash, 'Renderer changed during rendering'
    rgb = pfm(folder/'render.pfm')
    log_text = (folder/'render.log').read_text(encoding='utf-8',errors='replace')
    discarded = re.search(r'discarded surface births: (\{.*\})',log_text)
    if scene['media'][0]['type']=='global_conditional':
        assert discarded, 'Missing conditional discard diagnostics'
    row = dict(seconds=time.monotonic()-begin, resolution=scene['camera']['resolution'],
        spp=scene['renderer']['spp'], seed=SEED, threads=threads, provenance=provenance,
        source_scene_sha256=old['sha256']['scene.json'], changes=changes,
        discarded_surface_births=json.loads(discarded[1]) if discarded else None,
        mean_luminance=float((rgb@WEIGHTS).mean()),
        sha256={name:sha(folder/name) for name in FILES})
    save(folder/'result.json', row)
    verified(folder)
    return row


def compatibility(args):
    """Small-image regression checks; deliberately not a full-image equivalence claim."""
    baseline_hash, current_hash = sha(args.baseline_exe), sha(args.exe)
    bridge = load(REPORT/'renderer_compatibility.json')
    assert bridge['passed'] and bridge['renderer_sha256'] == baseline_hash
    rows, source_records = [], []
    for group in args.groups:
        for mode in ('global_conditional', 'sparse_3d_renewal_plus', GLOBAL):
            source = (renewal_destination(group, 'renewal_3d') if mode == 'sparse_3d_renewal_plus'
                      else original(group, mode))
            archived = verified(source)
            old = archived['source']['result']
            if mode == 'sparse_3d_renewal_plus':
                assert old['provenance']['renderer_sha256'] == baseline_hash
            else:
                checks = [c for c in bridge['checks'] if c['group']==group and c['mode']==mode]
                assert len(checks)==3 and all(c['passed'] for c in checks)
                assert all(c['original_result_sha256']==sha(source/'result.json') for c in checks)
            images = {}
            for version, exe in (('baseline', args.baseline_exe), ('current', args.exe)):
                folder = OUT/'controls'/group/mode/version
                scene = copy.deepcopy(archived['scene'])
                scene['camera']['resolution'] = [32, 32]
                scene['renderer'].update(spp=8, spp_step=8, output_directory=str(folder))
                changes = {'resolution':[32,32], 'spp':8}
                if version == 'current' and mode == 'global_conditional':
                    scene['media'][0][KEY] = 'retain_endpoint'
                    changes[KEY] = 'retain_endpoint'
                row = render_one(exe, folder, scene, old, args.threads, changes)
                images[version] = pfm(folder/'render.pfm')
                source_records.append(dict(group=group, mode=mode, version=version,
                    folder=str(folder), result_sha256=sha(folder/'result.json'), result=row))
            error = float(np.max(np.abs(images['baseline']-images['current'])))
            assert error == 0, (group, mode, error)
            rows.append(dict(group=group, mode=mode, maximum_absolute_rgb_change=error,
                             exactly_equal=True))
            print('Compatibility', group, mode, 'exact match', flush=True)
        folder = OUT/'controls'/group/'global_conditional'/'restart_exterior'
        scene, old = policy_scene(group, 'restart_exterior', folder, 32, 8)
        row = render_one(args.exe, folder, scene, old, args.threads,
                         {KEY:'restart_exterior', 'resolution':[32,32], 'spp':8})
        a = pfm(folder/'render.pfm')
        b = pfm(OUT/'controls'/group/'global_conditional/current/render.pfm')
        changed = int(np.count_nonzero(np.any(a != b, axis=2)))
        assert changed > 0, 'Restart policy did not affect the control image'
        rows.append(dict(group=group, mode='global_conditional_restart_effect',
                         changed_pixels=changed, maximum_absolute_rgb_change=float(np.max(np.abs(a-b)))))
        source_records.append(dict(group=group, mode='global_conditional', version='restart_exterior',
            folder=str(folder), result_sha256=sha(folder/'result.json'), result=row))
    save(OUT/'compatibility.json', dict(passed=True, groups=args.groups,
        current_renderer_sha256=current_hash, baseline_renderer_sha256=baseline_hash,
        baseline_executable=str(args.baseline_exe), current_executable=str(args.exe),
        archived_compatibility=dict(path=str(REPORT/'renderer_compatibility.json'),
            sha256=sha(REPORT/'renderer_compatibility.json')),
        checks=rows, sources=source_records,
        scope='32x32, 8 spp, fixed seed. Original Renewal+ and Global plus explicit retain_endpoint must be bit-identical to baseline; restart must change the image. Historical three-tile checks link baseline to archived renders, not a full-image equivalence proof.'))


def render(args):
    controls = load(OUT/'compatibility.json')
    assert controls['passed'] and controls['current_renderer_sha256'] == sha(args.exe)
    assert set(args.groups) <= set(controls['groups'])
    rows = []
    for group in args.groups:
        for policy in args.policies:
            folder = destination(group, policy)
            scene, old = policy_scene(group, policy, folder)
            print('Render', group, policy, '256x256 128 spp', flush=True)
            row = render_one(args.exe, folder, scene, old, args.threads, {KEY:policy})
            rows.append(dict(group=group, policy=policy, folder=str(folder), result=row))
            print('Done', group, policy, round(row['seconds'],2), 'seconds', flush=True)
            save(OUT/'run.json', dict(complete=len(rows)==len(args.groups)*len(args.policies),
                groups=args.groups, policies=args.policies, executable=str(args.exe),
                renderer_sha256=sha(args.exe), runner_sha256=sha(Path(__file__)),
                compatibility_sha256=sha(OUT/'compatibility.json'), sources=rows,
                settings=dict(resolution=[256,256], spp=128, seed=SEED, threads=args.threads,
                    policy_key=KEY, all_other_scene_parameters='Copied unchanged from archived global_conditional scene.')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('all','compatibility','render','report'))
    parser.add_argument('--groups', nargs='+', choices=GROUPS, default=list(GROUPS))
    parser.add_argument('--policies', nargs='+', choices=POLICIES, default=list(POLICIES))
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--exe', type=Path, default=EXE)
    parser.add_argument('--baseline-exe', type=Path, default=BASELINE_EXE)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error('threads must be positive')
    if args.command in ('compatibility','all'):
        compatibility(args)
    if args.command in ('render','all'):
        render(args)
    if args.command in ('report','all'):
        from gpis_shaderball_conditional_surface_policies_figures import report
        report(args)


if __name__ == '__main__':
    main()
