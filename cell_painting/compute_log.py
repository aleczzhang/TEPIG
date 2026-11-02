"""
compute_log.py
--------------
Track wall-clock time and peak memory of every pipeline step, on the Mac or on
Zaratan, so the multi-plate run can be sized to fit a ~12 h overnight job.
Every step appends one row to results/compute_log.csv:

  timestamp, host, script, label, wall_s, peak_rss_gb, extras(json)

compute_plan.py reads this file and extrapolates to P plates.

Usage:
    from compute_log import Timer, log_compute
    with Timer('read_sqlite', plate=24277, n_cells=226000):
        ...
"""

import os
import sys
import json
import time
import socket
import resource
import platform

_HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(_HERE, 'results', 'compute_log.csv')
_T0 = time.time()


def peak_rss_gb():
    """Peak resident set size of this process (GB). macOS reports bytes,
    Linux kilobytes."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if platform.system() == 'Darwin' else r / 1e6


def host():
    """'zaratan' under SLURM / on a zaratan node, 'mac' on macOS, else the short hostname."""
    h = socket.gethostname()
    if 'zaratan' in h or os.environ.get('SLURM_JOB_ID'):
        return 'zaratan'
    return 'mac' if platform.system() == 'Darwin' else h.split('.')[0]


def log_compute(label, wall_s, **extras):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    new = not os.path.exists(LOG)
    script = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else '?'
    with open(LOG, 'a') as f:
        if new:
            f.write('timestamp,host,script,label,wall_s,peak_rss_gb,extras\n')
        extras_csv = json.dumps(extras, default=str).replace('"', '""')
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')},{host()},{script},{label},"
                f"{wall_s:.2f},{peak_rss_gb():.3f},"
                f'"{extras_csv}"\n')
    el = int(time.time() - _T0)
    print(f"[{el // 60:02d}:{el % 60:02d}] {label}: {wall_s:.1f}s  "
          f"peak RSS {peak_rss_gb():.2f} GB  {extras if extras else ''}", flush=True)


class Timer:
    def __init__(self, label, **extras):
        self.label, self.extras = label, extras

    def __enter__(self):
        self.t = time.time()
        return self

    def add(self, **kw):
        self.extras.update(kw)

    def __exit__(self, *exc):
        log_compute(self.label, time.time() - self.t, **self.extras)
        return False


def read_log(path=LOG):
    """-> list of dict rows (extras parsed)."""
    import csv
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                r['extras'] = json.loads(r['extras'])
            except Exception:
                r['extras'] = {}
            r['wall_s'] = float(r['wall_s']); r['peak_rss_gb'] = float(r['peak_rss_gb'])
            rows.append(r)
    return rows
