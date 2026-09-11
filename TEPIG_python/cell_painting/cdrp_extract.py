"""
cdrp_extract.py
---------------
Build a TEPIG tensor cache from ONE OR MORE plates of single-cell CDRP data
(BBBC047 / cpg0012), reflecting the motivating tubule structure:

    subject  = well (a compound)
    objects  = single cells               (count varies -> the unbalanced mode)
    G = 2    = cell subpopulations, from ONE global GMM over all cells
    q        = morphology features (all raw ones; arms pre-screen later)
    S = 6    = imaging sites (fields of view) per well
    outcome  = the well's compound L1000 gene expression (continuous scalar)

Steps:
  1. platemap  -> well -> compound (broad_sample); DMSO wells are controls
  2. read single cells from the plate SQLite (Cells+Cytoplasm+Nuclei joined),
     tagged with Well + Site
  3. z-score each feature to the DMSO control cells (per-plate reference)
  4. hygiene only: drop nuisance families (coordinates, counts, IDs) and constant
     features. NO correlation prune: pre-screening is done per arm on the
     well-level matrix (preselect.py), so every arm starts from the same ~3,590
     raw morphology features. (--prune R restores the old cell-level prune.)
  5. ONE global GaussianMixture (G=2) over all cells -> cluster label per cell
  6. per well: x[g,:,s] = (cluster-g proportion at site s) * (mean of those cells)
     -> tensor X in R^(G, q, S, n)
  7. attach each well's compound L1000 expression -> outcome matrix
  8. save a cache compatible with repeated_runs.py / run_gene.py

Run schema check first (seconds, after downloading the sqlite):
    python cdrp_extract.py --sqlite 24277.sqlite --inspect

One plate (as before):
    python cdrp_extract.py --sqlite data/24277.sqlite --plate 24277

Several plates (the overnight run). Each plate's 8 GB sqlite is downloaded from the
Cell Painting Gallery if missing, read ONCE, z-scored to its own DMSO wells, reduced
to the feature set defined on the reference (first) plate, and parked as a float32
.npy in cache/ (~1.8 GB/plate); --delete-sqlite frees the 8 GB immediately after.
ONE global GMM is then fitted on a cell subsample drawn across all plates, every cell
is assigned, and the (G, q, S, n) tensor is assembled with n = wells over all plates.
    python cdrp_extract.py --plates 24277 24278 24279 --download --delete-sqlite \
                           --out cache/cdrp_multi.pkl
    python cdrp_extract.py --suggest 8      # picks 8 plates from 8 distinct platemaps

Every step's wall time and peak memory is appended to results/compute_log.csv
(see compute_log.py / compute_plan.py) so the plate count can be sized to ~12 h.
"""

import os
import re
import time
import argparse
import pickle
import sqlite3
import urllib.request

import numpy as np
import pandas as pd

_T0 = time.time()


def step(msg):
    """Timestamped progress line (mm:ss since start), flushed immediately."""
    el = int(time.time() - _T0)
    print(f"[{el // 60:02d}:{el % 60:02d}] {msg}", flush=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, 'cache')       # final tensor cache (.pkl) lives here
DATA_DIR = os.path.join(_HERE, 'data')         # raw downloads (sqlite, L1000, platemaps)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

GAL = "https://cellpainting-gallery.s3.amazonaws.com"
PLATEMAP_URL = (GAL + "/cpg0012-wawer-bioactivecompoundprofiling/broad/workspace/"
                "metadata/platemaps/CDRP/platemap/{pmap}.txt")
BARCODE_URL = (GAL + "/cpg0012-wawer-bioactivecompoundprofiling/broad/workspace/"
               "metadata/platemaps/CDRP/barcode_platemap.csv")
L1000_URL = (GAL + "/cpg0003-rosetta/broad/workspace/preprocessed_data/"
             "CDRP-BBBC047-Bray/L1000/replicate_level_l1k.csv.gz")
SQLITE_URL = (GAL + "/cpg0012-wawer-bioactivecompoundprofiling/broad/workspace/"
              "backend/CDRP/{plate}/{plate}.sqlite")

DROP_FAMILIES = ('Location', 'Number', 'Parent', 'Children', 'Neighbors',
                 'Metadata', 'Object')
DROP_PATTERNS = ('Center_X', 'Center_Y', 'Orientation', 'EulerNumber',
                 'ImageNumber', 'TableNumber', 'ObjectNumber')
N_CLUSTERS = 2
GMM_FIT_SAMPLE = 200_000     # cap cells used to FIT the GMM (assign uses all)


def _download(url, dest, chunk=1 << 24):
    """Streamed download with progress (the plate sqlites are ~8 GB each)."""
    if os.path.exists(dest):
        return dest
    print(f"  downloading {os.path.basename(dest)} ...", flush=True)
    tmp = dest + '.part'
    with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
        total = int(r.headers.get('Content-Length') or 0)
        done, last = 0, time.time()
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b); done += len(b)
            if total > 1e8 and time.time() - last > 30:
                step(f"    {done / 1e9:.1f} / {total / 1e9:.1f} GB"); last = time.time()
    os.replace(tmp, dest)
    return dest


# ── schema introspection ────────────────────────────────────────────────────
def introspect(conn):
    tabs = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    cols = {t: [r[1] for r in conn.execute(f'PRAGMA table_info("{t}")')]
            for t in tabs}
    return tabs, cols


def _find(cols, *needles):
    for c in cols:
        if all(n.lower() in c.lower() for n in needles):
            return c
    return None


def inspect(sqlite_path):
    conn = sqlite3.connect(sqlite_path)
    tabs, cols = introspect(conn)
    print(f"tables: {tabs}")
    for t in tabs:
        ff = [c for c in cols[t] if not c.startswith(('TableNumber', 'ImageNumber',
                                                      'ObjectNumber'))]
        print(f"\n  {t}: {len(cols[t])} cols")
        print(f"     keys present : "
              + ", ".join(k for k in ('TableNumber', 'ImageNumber', 'ObjectNumber')
                          if k in cols[t]))
        if t.lower() == 'image':
            meta = [c for c in cols[t] if 'Metadata' in c]
            print(f"     metadata cols: {meta[:12]}")
        else:
            par = [c for c in cols[t] if 'Parent' in c]
            print(f"     parent links : {par}")
            print(f"     sample feats : {ff[:4]}")
    # row counts
    for t in tabs:
        try:
            n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
            print(f"  rows[{t}] = {n:,}")
        except Exception:
            pass
    conn.close()


# ── load single cells ───────────────────────────────────────────────────────
def _feat_cols(cols):
    """Morphology feature columns for one object table (drop keys/nuisance)."""
    out = []
    for c in cols:
        if c in ('TableNumber', 'ImageNumber', 'ObjectNumber'):
            continue
        if any(p in c for p in DROP_PATTERNS):
            continue
        parts = c.split('_')
        fam = parts[1] if len(parts) > 1 else parts[0]     # e.g. Cells_AreaShape_.. -> AreaShape
        if fam in DROP_FAMILIES:
            continue
        out.append(c)
    return out


def load_cells(sqlite_path):
    conn = sqlite3.connect(sqlite_path)
    tabs, cols = introspect(conn)
    tab = {t.lower(): t for t in tabs}
    for need in ('image', 'cells', 'cytoplasm', 'nuclei'):
        if need not in tab:
            raise SystemExit(f"expected a '{need}' table; found {tabs}")
    Img, Cell, Cyto, Nuc = tab['image'], tab['cells'], tab['cytoplasm'], tab['nuclei']

    well = _find(cols[Img], 'Metadata', 'Well')
    site = _find(cols[Img], 'Metadata', 'Site') or _find(cols[Img], 'Metadata', 'Field')
    if well is None or site is None:
        raise SystemExit(f"could not find Well/Site metadata in Image: {cols[Img]}")
    keys = [k for k in ('TableNumber', 'ImageNumber') if k in cols[Cyto]]
    pcell = _find(cols[Cyto], 'Parent', 'Cells')
    pnuc = _find(cols[Cyto], 'Parent', 'Nuclei')

    # only pull morphology feature columns (not the ~750 nuisance/metadata cols)
    cyto_f, cell_f, nuc_f = _feat_cols(cols[Cyto]), _feat_cols(cols[Cell]), _feat_cols(cols[Nuc])
    feats = cyto_f + cell_f + nuc_f
    print(f"  join: Image[{well},{site}]  keys={keys}  "
          f"Cytoplasm.Parent_Cells={pcell}  Parent_Nuclei={pnuc}")
    print(f"  morphology feature columns: {len(cyto_f)}+{len(cell_f)}+{len(nuc_f)} = {len(feats)}")

    # SQLite caps a result set at 2000 columns, so we can't join all 3604 feature
    # columns in one SELECT. Read each table separately (each < 2000 cols). We
    # avoid BOTH slow paths that bit us: (a) pandas-merging three 226k x 1240
    # frames swap-thrashes on ~6.5 GB of copies, and (b) pandas read_sql on a
    # wide table materialises ~280M Python objects and is 30x+ slower than
    # linear. Instead read each table with a chunked cursor straight into a
    # preallocated float32 array (~38s/table), then resolve the parent-cell links
    # to row positions and assemble X by numpy fancy-indexing.
    def _read(table, id_cols, feat_cols, chunk=20000):
        # ID columns are read as exact int64 -- TableNumber reaches ~4.3e9, which
        # float32 (exact only to 16.7M) would corrupt, breaking the Image join.
        # Features go through the fast float32 path.
        step(f"  reading {table} ({len(feat_cols)} features)...")
        n = conn.execute('SELECT COUNT(*) FROM "%s"' % table).fetchone()[0]
        nid = len(id_cols)
        sel = ', '.join('"%s"' % c for c in id_cols + feat_cols)
        cur = conn.execute('SELECT %s FROM "%s"' % (sel, table))
        ids = np.empty((n, nid), dtype=np.int64)
        feat = np.empty((n, len(feat_cols)), dtype=np.float32)
        i = 0
        while True:
            rows = cur.fetchmany(chunk)
            if not rows:
                break
            k = len(rows)
            ids[i:i + k] = [r[:nid] for r in rows]          # exact ints
            fr = [r[nid:] for r in rows]
            try:                                            # fast path: all-numeric
                feat[i:i + k] = np.array(fr, dtype=np.float32)
            except (TypeError, ValueError):                 # a chunk with SQL NULLs
                b = np.array(fr, dtype=object)
                b[b == None] = np.nan                       # noqa: E711
                feat[i:i + k] = b.astype(np.float32)
            i += k
        return ids, feat

    # id-column layouts: cyto = [Table, Image, Object, Parent_Cells, Parent_Nuclei]
    #                    cell/nuc = [Table, Image, Object]
    k_cy, A_cy = _read(Cyto, keys + ['ObjectNumber', pcell, pnuc], cyto_f)
    k_ce, A_ce = _read(Cell, keys + ['ObjectNumber'], cell_f)
    k_nu, A_nu = _read(Nuc, keys + ['ObjectNumber'], nuc_f)
    img = pd.read_sql_query(
        'SELECT %s FROM "%s"' % (', '.join('"%s"' % c for c in keys + [well, site]), Img),
        conn)
    conn.close()

    step("  aligning cells <-> cytoplasm <-> nuclei by parent links...")

    def pos_index(kint, objcol):
        # (Table, Image, ObjectNumber) -> row position in that table's array
        mi = pd.MultiIndex.from_arrays([kint[:, 0], kint[:, 1], kint[:, objcol]])
        return pd.Series(np.arange(len(kint)), index=mi)

    cell_pos = pos_index(k_ce, 2)                   # ObjectNumber at column 2
    nuc_pos = pos_index(k_nu, 2)
    ce_key = pd.MultiIndex.from_arrays([k_cy[:, 0], k_cy[:, 1], k_cy[:, 3]])  # Parent_Cells
    nu_key = pd.MultiIndex.from_arrays([k_cy[:, 0], k_cy[:, 1], k_cy[:, 4]])  # Parent_Nuclei
    ci = cell_pos.reindex(ce_key).to_numpy()
    ni = nuc_pos.reindex(nu_key).to_numpy()
    good = ~(np.isnan(ci) | np.isnan(ni))
    ci, ni = ci[good].astype(np.int64), ni[good].astype(np.int64)

    X = np.hstack([A_cy[good], A_ce[ci], A_nu[ni]])
    del A_cy, A_ce, A_nu

    # Well / Site per surviving cytoplasm row, via the Image key
    imap = img.set_index(keys)
    ik = pd.MultiIndex.from_arrays([k_cy[good, 0], k_cy[good, 1]], names=keys)
    wells = imap[well].reindex(ik).to_numpy()
    sites = imap[site].reindex(ik).to_numpy()
    print(f"  loaded {X.shape[0]:,} cells x {X.shape[1]} features "
          f"({int((~good).sum()):,} dropped for missing parent link)")
    return X, feats, wells, sites

# ── per-plate processing ────────────────────────────────────────────────────
def resolve_platemap(plate):
    """-> (platemap name, well->broad_sample dict, set of DMSO control wells)."""
    bc = pd.read_csv(_download(BARCODE_URL, os.path.join(DATA_DIR, 'barcode.csv')))
    row = bc[bc['Assay_Plate_Barcode'].astype(str) == str(plate)]
    if row.empty:
        raise SystemExit(f"plate {plate} not in barcode_platemap")
    pmap = row['Plate_Map_Name'].iloc[0]
    pm = pd.read_csv(_download(PLATEMAP_URL.format(pmap=pmap),
                               os.path.join(DATA_DIR, f'{pmap}.txt')), sep='\t')
    pm['well'] = pm['well_position'].map(norm_well)
    well2cpd = dict(zip(pm['well'], pm['broad_sample']))
    control_wells = set(pm.loc[pm['broad_sample'].isna(), 'well'])
    return pmap, well2cpd, control_wells


def norm_well(w):
    return str(w).strip().upper()


def suggest_plates(n, min_cov=0.95):
    """n plates from n DISTINCT platemaps (= distinct compounds -> more subjects),
    restricted to platemaps whose compounds HAVE L1000 profiles. Coverage is
    all-or-nothing by platemap (e.g. H-BIOA-004-3 = 320/320, H-BIOA-007-3 = 0/320),
    so this matters: 24278 contributed ONE usable well. Uses
    data/platemap_l1000_coverage.csv if present (else computes it, ~1 min:
    downloads all 97 platemaps and matches compounds to the L1000 file)."""
    cov_path = os.path.join(DATA_DIR, 'platemap_l1000_coverage.csv')
    if not os.path.exists(cov_path):
        bc = pd.read_csv(_download(BARCODE_URL, os.path.join(DATA_DIR, 'barcode.csv')))
        l1k = pd.read_csv(_download(L1000_URL, os.path.join(DATA_DIR, 'cdrp_l1k.csv.gz')),
                          usecols=['pert_id'])
        def core(v):
            m = re.match(r'BRD-[A-Za-z](\d+)', str(v)); return m.group(1) if m else None
        have = set(l1k['pert_id'].map(core).dropna())
        rows = []
        for pmap, grp in bc.groupby('Plate_Map_Name'):
            pm = pd.read_csv(_download(PLATEMAP_URL.format(pmap=pmap),
                                       os.path.join(DATA_DIR, f'{pmap}.txt')), sep='\t')
            cp = pm['broad_sample'].dropna().map(core)
            rows.append((pmap, len(cp), int(cp.isin(have).sum()),
                         ' '.join(grp['Assay_Plate_Barcode'].astype(str))))
        pd.DataFrame(rows, columns=['platemap', 'compound_wells', 'with_L1000', 'plates']
                     ).to_csv(cov_path, index=False)
    cov = pd.read_csv(cov_path)
    cov = cov[cov['with_L1000'] >= min_cov * cov['compound_wells']]
    cov = cov.sort_values(['with_L1000', 'platemap'], ascending=[False, True])
    picks = ['24277'] if '24277' in ' '.join(cov['plates']) else []   # the plate we have
    for _, r in cov.iterrows():
        first = str(r['plates']).split()[0]
        if first not in picks and not str(r['plates']).startswith('24277'):
            picks.append(first)
        if len(picks) >= n:
            break
    return picks[:n]


def select_features_ref(X, feats, diagnostics=True, prune=None):
    """Feature set on the REFERENCE plate. By default this is HYGIENE ONLY --
    drop constant features -- so that every pre-screening arm (preselect.py)
    starts from the same raw morphology set and nothing is pre-pruned for it.
    prune=<r> optionally re-enables the old greedy |r|<=r cell-level prune (the
    pre-Sept-2026 caches were built with prune=0.95). Returns kept NAMES."""
    q0 = len(feats)
    keepvar = X.std(axis=0) > 1e-6
    X, feats = X[:, keepvar], [f for f, k in zip(feats, keepvar) if k]
    step(f"variance filter: {q0} -> {len(feats)} features "
         f"({q0 - len(feats)} dropped as constant)")
    if prune is None:
        step(f"no correlation prune at the cell level (arms preselect on the "
             f"well-level matrix); q={len(feats)}")
        return list(feats)
    from prune_sweep import prune_correlated
    step(f"computing {len(feats)}x{len(feats)} correlation matrix on {len(X):,} cells...")
    Rabs = np.abs(np.corrcoef(X.T))
    iu = np.triu_indices(len(feats), 1)
    n_hi = int((Rabs[iu] > prune).sum())
    step(f"pruning at |r|<={prune} ({n_hi:,} of {len(iu[0]):,} feature pairs exceed it)...")
    keep = prune_correlated(Rabs, prune)
    if diagnostics:
        Rk = Rabs[np.ix_(keep, keep)]
        mx = Rk[np.triu_indices(len(keep), 1)].max() if len(keep) > 1 else 0.0
        print("  kept features by family:")
        fam = pd.Series([f.split('_')[1] if '_' in f else f
                         for f in (feats[i] for i in keep)]).value_counts()
        for name, c in fam.items():
            print(f"    {name:<22}{c:>4}")
        print(f"  max |r| among kept features: {mx:.3f} (should be <= {prune})")
    return [feats[i] for i in keep]


def process_plate(plate, sqlite_path, feat_names=None, diagnostics=True, prune=None):
    """Read one plate, z-score to its DMSO cells, reduce to feat_names (or define
    them if None), park cells as cache/_plate_<id>_X.npy. Returns (feat_names,
    meta dict)."""
    from compute_log import Timer
    pmap, well2cpd, control_wells = resolve_platemap(plate)
    print(f"  plate {plate}: platemap {pmap}: "
          f"{sum(pd.notna(v) for v in well2cpd.values())} compound wells, "
          f"{len(control_wells)} controls")

    with Timer('read_sqlite', plate=plate) as t:
        X, feats, wells, sites = load_cells(sqlite_path)
        t.add(n_cells=int(X.shape[0]), q_raw=len(feats))
    wells = np.array([norm_well(w) for w in wells])
    sites = sites.astype(int)

    with Timer('zscore', plate=plate):
        ctrl = np.isin(wells, list(control_wells))
        print(f"  control cells: {ctrl.sum():,} / {len(wells):,}")
        mu = np.nanmean(X[ctrl], axis=0)
        sd = np.nanstd(X[ctrl], axis=0)
        sd = np.where((sd > 1e-9) & np.isfinite(sd), sd, 1.0)
        X -= np.where(np.isfinite(mu), mu, 0.0)
        X /= sd
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        np.clip(X, -20, 20, out=X)

    with Timer('select_features', plate=plate, reference=feat_names is None) as t:
        if feat_names is None:
            feat_names = select_features_ref(X, feats, diagnostics, prune)
        pos = {f: i for i, f in enumerate(feats)}
        missing = [f for f in feat_names if f not in pos]
        if missing:
            raise SystemExit(f"plate {plate} lacks {len(missing)} reference features, "
                             f"e.g. {missing[:3]}")
        X = np.ascontiguousarray(X[:, [pos[f] for f in feat_names]], dtype=np.float32)
        t.add(q=len(feat_names))

    npy = os.path.join(CACHE_DIR, f'_plate_{plate}_X.npy')
    with Timer('park_npy', plate=plate, gb=round(X.nbytes / 1e9, 2)):
        np.save(npy, X)
    meta = {'plate': plate, 'pmap': pmap, 'well2cpd': well2cpd,
            'control_wells': control_wells, 'wells': wells, 'sites': sites,
            'npy': npy, 'n_cells': int(X.shape[0]), 'feat_names': list(feat_names)}
    with open(_meta_path(plate), 'wb') as f:
        pickle.dump(meta, f)
    del X
    return feat_names, meta


def _meta_path(plate):
    return os.path.join(CACHE_DIR, f'_plate_{plate}_meta.pkl')


def load_parked(plate, feat_names):
    """Reuse a previously parked plate (cache/_plate_<id>_X.npy + _meta.pkl) if its
    feature list matches; else None."""
    mp, npy = _meta_path(plate), os.path.join(CACHE_DIR, f'_plate_{plate}_X.npy')
    if not (os.path.exists(mp) and os.path.exists(npy)):
        return None
    meta = pickle.load(open(mp, 'rb'))
    if feat_names is not None and list(meta['feat_names']) != list(feat_names):
        return None
    return meta


def build_tensor(meta, lab, q, S=6):
    """(G, q, S, n_plate) tensor for one plate from its parked cells + labels."""
    X = np.load(meta['npy'], mmap_mode='r')
    wells, sites, well2cpd = meta['wells'], meta['sites'], meta['well2cpd']
    subjects = sorted(w for w in set(wells) if w not in meta['control_wells']
                      and pd.notna(well2cpd.get(w)))
    n = len(subjects)
    Xten = np.zeros((N_CLUSTERS, q, S, n), dtype=np.float32)
    for i, w in enumerate(subjects):
        wm = np.where(wells == w)[0]
        Xw, lw, sw = np.asarray(X[wm]), lab[wm], sites[wm]
        for s in range(1, S + 1):
            sm = sw == s
            if not sm.any():
                continue
            tot = sm.sum()
            for g in range(N_CLUSTERS):
                gm_ = sm & (lw == g)
                if gm_.any():
                    Xten[g, :, s - 1, i] = (gm_.sum() / tot) * Xw[gm_].mean(axis=0)
    return Xten, subjects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sqlite', help='path to <plate>.sqlite (single-plate mode)')
    ap.add_argument('--plate', help='plate id (e.g. 24277); inferred from --sqlite if omitted')
    ap.add_argument('--plates', nargs='+', help='plate ids for a multi-plate cache; '
                    'sqlites are looked up as data/<plate>.sqlite')
    ap.add_argument('--download', action='store_true',
                    help='download missing sqlites from the Cell Painting Gallery')
    ap.add_argument('--delete-sqlite', action='store_true',
                    help='delete each 8 GB sqlite once its cells are parked (disk)')
    ap.add_argument('--keep-npy', action='store_true',
                    help='keep the parked per-plate .npy files after the cache is built')
    ap.add_argument('--reuse-parked', action='store_true',
                    help='skip the sqlite read for plates already parked in cache/ '
                         '(from a previous run with --keep-npy or one that died)')
    ap.add_argument('--suggest', type=int, metavar='N',
                    help='print N plate ids from N distinct platemaps and exit')
    ap.add_argument('--inspect', action='store_true', help='dump schema and exit')
    ap.add_argument('--out', default=None,
                    help='cache path (default cache/cdrp_singlecell.pkl for one plate, '
                         'cache/cdrp_<P>plates.pkl for P plates)')
    ap.add_argument('--l1000', default=os.path.join(DATA_DIR, 'cdrp_l1k.csv.gz'))
    ap.add_argument('--prune', type=float, default=None, metavar='R',
                    help='(legacy) greedy |r|<=R prune at the cell level before '
                         'clustering; OFF by default so each pre-screening arm '
                         'starts from all morphology features. Old caches used 0.95.')
    ap.add_argument('--gmm-sample', type=int, default=GMM_FIT_SAMPLE,
                    help='total cells (across plates) used to FIT the GMM')
    ap.add_argument('--diagnostics', action='store_true', default=True)
    ap.add_argument('--no-diagnostics', dest='diagnostics', action='store_false')
    ap.add_argument('--silhouette', action='store_true',
                    help='silhouette score on a cell subsample (slower)')
    args = ap.parse_args()

    if args.inspect:
        inspect(args.sqlite); return
    if args.suggest:
        print(' '.join(suggest_plates(args.suggest))); return

    import sys
    sys.path.insert(0, _HERE)
    from compute_log import Timer, log_compute, peak_rss_gb

    # ── which plates ─────────────────────────────────────────────────────────
    if args.plates:
        plates = [str(p) for p in args.plates]
        sqlites = {p: os.path.join(DATA_DIR, f'{p}.sqlite') for p in plates}
    elif args.sqlite:
        p = args.plate or re.sub(r'\D', '', os.path.basename(args.sqlite))[:5]
        plates, sqlites = [p], {p: args.sqlite}
    else:
        raise SystemExit('give --sqlite <file> (one plate) or --plates <ids...>')
    P = len(plates)
    out = args.out or os.path.join(
        CACHE_DIR, 'cdrp_singlecell.pkl' if P == 1 else f'cdrp_{P}plates.pkl')
    t_all = time.time()
    print(f"plates ({P}): {' '.join(plates)}\n")

    # ── pass 1: read, z-score, reduce, park each plate ───────────────────────
    feat_names, metas = None, []
    for p in plates:
        step(f"=== plate {p} ===")
        with Timer('plate_total', plate=p):
            if args.reuse_parked:
                meta = load_parked(p, feat_names)
                if meta is not None:
                    feat_names = meta['feat_names']
                    metas.append(meta)
                    step(f"  reusing parked cells for plate {p} ({meta['n_cells']:,} cells)")
                    continue
            if not os.path.exists(sqlites[p]):
                if not args.download:
                    raise SystemExit(f"{sqlites[p]} missing (add --download)")
                with Timer('download', plate=p):
                    _download(SQLITE_URL.format(plate=p), sqlites[p])
            feat_names, meta = process_plate(p, sqlites[p], feat_names, args.diagnostics,
                                             args.prune)
            metas.append(meta)
            if args.delete_sqlite and args.plates:
                os.remove(sqlites[p]); step(f"  deleted {sqlites[p]}")
    q = len(feat_names)
    n_cells = sum(m['n_cells'] for m in metas)
    step(f"all plates parked: {n_cells:,} cells x q={q} features")

    # ── pass 2: ONE global GMM on a subsample drawn across plates ────────────
    # Diagonal covariance (full is singular at this q). float32 on purpose: with
    # ~3,600 raw features the 200k-cell fit matrix is 2.9 GB in float32 and sklearn
    # allocates a same-size X**2 copy inside the diag E-step -- in float64 that pair
    # (11 GB) pushed a 16 GB Mac into swap (35 min at 17 % CPU, never finished).
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(0)
    with Timer('gmm_fit', n_plates=P, q=q) as t:
        parts = []
        for m in metas:
            Xp = np.load(m['npy'], mmap_mode='r')
            k = min(m['n_cells'], int(round(args.gmm_sample * m['n_cells'] / n_cells)))
            idx = np.sort(rng.choice(m['n_cells'], k, replace=False))
            parts.append(np.asarray(Xp[idx], dtype=np.float32))
        fit = np.vstack(parts); del parts
        step(f"fitting global GMM (G={N_CLUSTERS}, diag cov) on {len(fit):,} cells...")
        gm = GaussianMixture(n_components=N_CLUSTERS, covariance_type='diag',
                             random_state=0, n_init=3, reg_covar=1e-3).fit(fit)
        t.add(n_fit=int(len(fit)), converged=bool(gm.converged_), iters=int(gm.n_iter_))
    step(f"GMM done (converged={gm.converged_}, {gm.n_iter_} EM iters, "
         f"lower_bound={gm.lower_bound_:.2f})")

    if args.diagnostics:
        lab_fit = gm.predict(fit)
        cmean = np.stack([fit[lab_fit == g].mean(axis=0) for g in range(N_CLUSTERS)])
        diff = cmean[1] - cmean[0]
        order = np.argsort(-np.abs(diff))
        print("  top features separating cluster 1 from cluster 0 (mean gap, z-units):")
        for i in order[:12]:
            print(f"    {diff[i]:+6.2f}   {feat_names[i]}")
        post = gm.predict_proba(fit).max(axis=1)
        print(f"  assignment confidence: median max-posterior={np.median(post):.2f}, "
              f"{np.mean(post < 0.6):.1%} of cells ambiguous (<0.6)")
    if args.silhouette:
        from sklearn.metrics import silhouette_score
        si = rng.choice(len(fit), min(5000, len(fit)), replace=False)
        print(f"  silhouette score: {silhouette_score(fit[si], gm.predict(fit[si])):.3f} "
              f"(>0.5 strong, 0.25-0.5 moderate, <0.25 weak separation)")
    del fit

    # ── pass 3: assign every cell, build each plate's tensor slice ──────────
    tensors, subjects, obs_plate, obs_cpd, well2cpd_all = [], [], [], [], {}
    sizes = np.zeros(N_CLUSTERS, dtype=np.int64)
    for m in metas:
        with Timer('assign_and_tensor', plate=m['plate']) as t:
            Xp = np.load(m['npy'], mmap_mode='r')
            lab = np.concatenate([gm.predict(np.asarray(Xp[i:i + 50000], dtype=np.float32))
                                  for i in range(0, m['n_cells'], 50000)])
            sizes += np.bincount(lab, minlength=N_CLUSTERS)
            Xt, subs = build_tensor(m, lab, q)
            t.add(n_wells=len(subs))
        tensors.append(Xt)
        subjects += [f"{m['plate']}:{w}" for w in subs]
        obs_plate += [m['plate']] * len(subs)
        obs_cpd += [m['well2cpd'][w] for w in subs]
        if not args.keep_npy:
            os.remove(m['npy']); os.remove(_meta_path(m['plate']))
    for g in range(N_CLUSTERS):
        print(f"    cluster {g}: {sizes[g]:>9,} cells ({sizes[g] / sizes.sum():.1%})")
    Xten = np.concatenate(tensors, axis=3); del tensors
    n = Xten.shape[3]
    empty = np.mean((Xten == 0).all(axis=1))
    print(f"  tensor X: {Xten.shape}  (G, q, S, n={n} wells over {P} plates)")
    print(f"  empty cluster-x-site-x-well blocks: {empty:.1%}")

    # ── outcome: compound L1000 expression (affy probes; symbols via idmap) ─
    with Timer('outcomes', n=n):
        _download(L1000_URL, args.l1000)
        l1k = pd.read_csv(args.l1000)
        META = {'pert_id', 'pert_dose', 'det_plate', 'BROAD_CPD_ID', 'CPD_NAME',
                'CPD_TYPE', 'CPD_SMILES', 'pert_id_dose', 'pert_sample_dose',
                'pert_type', 'control_type'}
        genes = [c for c in l1k.columns if c not in META]
        idcol = 'pert_id' if 'pert_id' in l1k.columns else 'BROAD_CPD_ID'
        def core(v):
            mm = re.match(r'BRD-[A-Za-z](\d+)', str(v)); return mm.group(1) if mm else None
        l1k['core'] = l1k[idcol].map(core)
        gexpr = l1k.groupby('core')[genes].mean()
        expr = np.full((n, len(genes)), np.nan, dtype=np.float32)
        matched = 0
        for i, cpd in enumerate(obs_cpd):
            c = core(cpd)
            if c in gexpr.index:
                expr[i] = gexpr.loc[c].values; matched += 1
        print(f"  wells matched to an L1000 profile: {matched}/{n}")

    idmap_path = os.path.join(_HERE, '..', '..', 'LINCS-Pilot1', 'idmap.xlsx')
    sym2probe = {}
    if os.path.exists(idmap_path):
        pset = set(genes)
        for _, r in pd.read_excel(idmap_path).iterrows():
            if r['probe_id'] in pset and isinstance(r['symbol'], str):
                sym2probe.setdefault(r['symbol'], r['probe_id'])
    else:
        print("  (idmap.xlsx not found; pass a probe id as --gene instead of a symbol)")
    print(f"  gene symbols mapped to probes: {len(sym2probe)}")

    cache = {
        'X': Xten, 'features': list(feat_names),
        'compartments': [f'cluster{g}' for g in range(N_CLUSTERS)],
        'channels': [f'site{s}' for s in range(1, 6 + 1)],
        'obs_compound': np.array(obs_cpd, dtype=object),
        'obs_plate': np.array(obs_plate, dtype=object),
        'obs_well': np.array(subjects, dtype=object),
        'obs_dose_rank': np.ones(n, dtype=np.int64),
        'expr': expr, 'probes': list(genes),
        'sym2probe': sym2probe, 'probe2sym': {p: s for s, p in sym2probe.items()},
        'G': N_CLUSTERS, 'q': q, 'S': 6, 'n': n, 'plates': plates,
        'cell_prune': args.prune,
    }
    with open(out, 'wb') as f:
        pickle.dump(cache, f)
    log_compute('extract_total', time.time() - t_all, n_plates=P, n_cells=n_cells,
                q=q, n_wells=n, cache_mb=round(os.path.getsize(out) / 1e6))
    print(f"\nSaved -> {out}   (peak RSS {peak_rss_gb():.2f} GB)")
    print(f"  run it with:  python repeated_runs.py --gene <GENE> --dose 1 "
          f"--cache {os.path.basename(out)}")


if __name__ == '__main__':
    main()
