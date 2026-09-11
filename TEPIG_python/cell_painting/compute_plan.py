"""
compute_plan.py
---------------
Size the multi-plate overnight run from MEASURED timings in results/compute_log.csv
(written by cdrp_extract.py, repeated_runs.py, run_panel.py via compute_log.py).

Cost model (P plates, ~320 compound wells each):
    T(P) = P * t_plate                      extraction: download+read+zscore+park
         + t_gmm(P)                         one global GMM (sample size fixed -> ~const)
         + P * t_assign                     assign cells + build tensor slice
         + n_genes * n_runs * sum_m t_m(n)  fits; t_m(n) = a_m + b_m * n per method,
                                            fitted by least squares to the logged runs
Peak memory = max over steps (extraction dominates: ~7 GB for one plate's raw cells).

Usage:
    python compute_plan.py                       # all hosts, defaults
    python compute_plan.py --host zaratan --hours 12 --genes 6 --runs 10
    python compute_plan.py --n-per-plate 320 --k 262
"""
import os
import sys
import argparse
from collections import defaultdict
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from compute_log import read_log, LOG                          # noqa: E402


def _mean(rows, label, key=None):
    v = [r['wall_s'] for r in rows if r['label'] == label
         and (key is None or key(r['extras']))]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def _min(t):
    """format seconds as minutes, or '   (none)' if not logged yet."""
    return f'{t / 60:6.1f} min' if t is not None else '   (none)'


def _fit_linear(rows, label, method):
    """t = a + b*n from rows of `label` whose extras carry n and method."""
    pts = [(r['extras'].get('n'), r['wall_s']) for r in rows
           if r['label'] == label and r['extras'].get('method') == method
           and r['extras'].get('n')]
    if not pts:
        return None
    n = np.array([p[0] for p in pts], float); t = np.array([p[1] for p in pts], float)
    if len(np.unique(n)) < 2:                       # one sample size: scale linearly
        return (0.0, float(t.mean() / n.mean()))
    b, a = np.polyfit(n, t, 1)
    return (max(0.0, float(a)), max(0.0, float(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default=None, help='use only rows from this host')
    ap.add_argument('--hours', type=float, default=12.0)
    ap.add_argument('--genes', type=int, default=6, help='outcome genes in the panel')
    ap.add_argument('--runs', type=int, default=10, help='repeated runs per gene')
    ap.add_argument('--n-per-plate', type=int, default=320)
    ap.add_argument('--k', type=int, default=None, help='restrict fit rows to this q')
    ap.add_argument('--max-plates', type=int, default=40)
    a = ap.parse_args()

    rows = read_log()
    if not rows:
        raise SystemExit(f'no timings yet in {LOG}')
    if a.host:
        rows = [r for r in rows if r['host'] == a.host]
    hosts = sorted({r['host'] for r in rows})
    print(f"timings from {LOG}: {len(rows)} rows, hosts={hosts}\n")

    t_plate, n1 = _mean(rows, 'plate_total')
    t_read, _ = _mean(rows, 'read_sqlite')
    t_dl, n_dl = _mean(rows, 'download')
    t_gmm, n2 = _mean(rows, 'gmm_fit')
    t_assign, n3 = _mean(rows, 'assign_and_tensor')
    peak = max(r['peak_rss_gb'] for r in rows)
    print('extraction (per plate):')
    print(f"  plate_total       {_min(t_plate):>12}  ({n1} obs"
          + (f"; includes download {t_dl/60:.1f} min" if t_dl else '; no download timed') + ')')
    print(f"  read_sqlite       {_min(t_read):>12}")
    print(f"  gmm_fit           {_min(t_gmm):>12}  ({n2} obs; fixed sample size)")
    print(f"  assign_and_tensor {_min(t_assign):>12}  ({n3} obs)")
    print(f"  peak RSS logged   {peak:6.2f} GB\n")

    fit_rows = [r for r in rows if r['label'] in ('fit_seed', 'fit_bench')
                and (a.k is None or r['extras'].get('q') == a.k)]
    methods = sorted({r['extras'].get('method') for r in fit_rows if r['extras'].get('method')})
    models = {m: _fit_linear(fit_rows, 'fit_seed', m) or _fit_linear(fit_rows, 'fit_bench', m)
              for m in methods}
    print('fit time per method, t(n) = a + b*n  (from logged fits'
          + (f', q={a.k}' if a.k else ', all q') + '):')
    for m, ab in models.items():
        if ab:
            print(f"  {m:<8} a={ab[0]:6.1f}s  b={ab[1]:.4f} s/well   "
                  f"-> n=320: {ab[0]+ab[1]*320:5.0f}s  n=2560: {ab[0]+ab[1]*2560:5.0f}s")
    if not any(models.values()):
        print('  (none logged yet: run repeated_runs.py / run_panel.py once)')

    def T(P):
        n = a.n_per_plate * P
        ext = P * (t_plate or 0) + (t_gmm or 0) + P * (t_assign or 0)
        fit = sum(ab[0] + ab[1] * n for ab in models.values() if ab)
        return ext, a.genes * a.runs * fit

    print(f"\nprojection: {a.genes} genes x {a.runs} runs, {a.n_per_plate} wells/plate, "
          f"budget {a.hours:.0f} h")
    print(f"{'plates':>7}{'n':>7}{'extract':>10}{'fits':>10}{'total':>10}")
    best = None
    for P in range(1, a.max_plates + 1):
        ext, fit = T(P)
        tot = ext + fit
        flag = ''
        if tot <= a.hours * 3600:
            best = P
        elif best is not None and P == best + 1:
            flag = '   <- exceeds budget'
        if P <= 12 or P == best or flag:
            print(f"{P:>7}{a.n_per_plate*P:>7}{ext/3600:>9.1f}h{fit/3600:>9.1f}h{tot/3600:>9.1f}h{flag}")
    print(f"\n=> largest plate count within {a.hours:.0f} h: {best}"
          + ('' if best else ' (even one plate exceeds the budget)'))
    print('   memory: extraction peak ~ one plate of raw cells (float32, ~3.6k cols) '
          '+ parked features; request >= 1.5 x logged peak.')
    print('   disk: 8 GB sqlite/plate (--delete-sqlite frees it) + ~1.8 GB parked .npy/plate.')


if __name__ == '__main__':
    main()
