"""Run one batch with optional, recorded affinity to physical performance cores."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import struct
import subprocess
import time
import re

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument('spec', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--performance-cores', action='store_true')
p.add_argument('--exe', default='conditional_transition_benchmark')
args = p.parse_args()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
metadata = {'logical_cpus': os.cpu_count(), 'performance_core_affinity': args.performance_cores}
mask = None
if args.performance_cores:
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    size = ctypes.c_ulong()
    kernel.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), None, 0)
    buffer = ctypes.create_string_buffer(size.value)
    if not kernel.GetSystemCpuSetInformation(buffer, size, ctypes.byref(size), None, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    records, offset = [], 0
    while offset < size.value:
        length, kind = struct.unpack_from('<II', buffer.raw, offset)
        if kind == 0:
            ident, group, logical, core, cache, numa, efficiency, flags = struct.unpack_from('<IHBBBBBB', buffer.raw, offset+8)
            if group != 0:
                raise RuntimeError('This benchmark affinity helper requires one processor group')
            records.append(dict(logical=logical, core=core, efficiency=efficiency))
        offset += length
    best = max(r['efficiency'] for r in records)
    cores = {}
    for record in records:
        if record['efficiency'] == best:
            cores.setdefault(record['core'], record['logical'])
    mask = sum(1 << logical for logical in cores.values())
    metadata.update(cpu_sets=records, selected_logical_cpus=list(cores.values()), affinity_mask=mask)
with (output/'execution.log').open('wb') as log:
    command = [str(ROOT/'build/Release'/f'{args.exe}.exe'), str(args.spec.resolve()), str(output)]
    if args.exe == 'tungsten':
        spec = json.loads(args.spec.read_text(encoding='utf-8-sig'))
        command = [str(ROOT/'build/Release/tungsten.exe'), '-t', str(spec['threads']), '-s', str(spec['seed']),
                   '-d', str(output), spec['scene']]
    started = time.perf_counter()
    timing = {}
    child = subprocess.Popen(command,
                             cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if mask is not None:
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not kernel.SetProcessAffinityMask(int(child._handle), mask):
            raise ctypes.WinError(ctypes.get_last_error())
    (output/'host.json').write_text(json.dumps(metadata, indent=2))
    for line in child.stdout:
        log.write(line)
        log.flush()
        elapsed = time.perf_counter() - started
        if b'Starting render' in line:
            timing['startup_and_scene_loading_seconds'] = elapsed
        if b'Finished render. Render time ' in line:
            duration = line.split(b'Render time ', 1)[1]
            units = {b'd': 86400, b'h': 3600, b'm': 60, b's': 1, b'ms': .001}
            timing['render_seconds'] = sum(float(value) * units[unit]
                for value, unit in re.findall(rb'([0-9.eE+-]+)(ms|d|h|m|s)', duration))
    code = child.wait()
    timing.update(wall_seconds=time.perf_counter() - started, exit_code=code)
    (output/'timing.json').write_text(json.dumps(timing, indent=2))
print(f'{args.exe}: exit {code}, log {output / "execution.log"}', flush=True)
raise SystemExit(code)
