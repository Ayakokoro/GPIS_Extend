"""Recover report publication while the original Renewal+ render jobs continue.

Windows readers can briefly prevent os.replace. Retry publication without
restarting or changing the running renderer, scene files, or sampler settings.
"""
import argparse
import os
from pathlib import Path
import time

import render_conditional_groove_renewal as r


def replace_retry(temporary,path):
    for attempt in range(20):
        try:
            os.replace(temporary,path)
            return
        except PermissionError:
            if attempt==19: raise
            time.sleep(.25)


def save_json(path,value):
    temporary=path.with_name(path.name+f'.reporter-{os.getpid()}.tmp')
    r.p.ORIGINAL_SAVE(temporary,value)
    replace_retry(temporary,path)


def save_text(path,value):
    temporary=path.with_name(path.name+f'.reporter-{os.getpid()}.tmp')
    temporary.write_text(value,encoding='utf-8')
    replace_retry(temporary,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    baseline=r.g.load(r.STAGE/'manifest.json')
    manifest=r.g.load(r.STAGE/'renewal_plus_manifest.json')
    if r.g.digest(Path(r.__file__))!=manifest['runner_sha256']:
        raise RuntimeError('Frozen render runner changed')
    r.g.save=r.p.atomic_save=save_json
    r.p.atomic_text=save_text
    save_json(r.STAGE/'renewal_plus_report_recovery.json',dict(
        reporter_sha256=r.g.digest(Path(__file__)),render_runner_unchanged=True,
        purpose='Retry transient Windows file replacement failures while original render workers continue.',
        scene_or_renderer_changes=False))
    while True:
        active=[r.g.job_key(j) for j in manifest['jobs'] if not (r.STAGE/r.g.job_key(j)/'result.json').exists()]
        for attempt in range(10):
            try:
                state=r.publish(baseline,manifest,active)
                break
            except PermissionError:
                if attempt==9: raise
                time.sleep(.5)
        print(f'REPORT {state["completed"]}/13 failed={state["failed"]}',flush=True)
        if state['completed']==13:
            r.validate(baseline,manifest)
            print('COMPLETE: all 13 native images validated',flush=True)
            return
        if state['failed'] and not active:
            raise RuntimeError('Failed render retained for diagnosis')
        if not args.watch: return
        time.sleep(30)


if __name__=='__main__': main()
