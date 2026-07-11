"""
build_data.py
-------------
Build the Cell Painting design tensor and matched L1000 outcomes for the
LINCS-Pilot1 dataset (cpg0003-rosetta), for use with TEPIG / CLUSSO / naive.

Tensor structure (channel as the third mode; mirrors TEPIG's (G, q, S, n)):
    G = 3 compartments : Cells, Cytoplasm, Nuclei          (the "cluster" analog)
    q = feature-cores  : single-channel measurements, channel stripped from name
    S = 5 channels     : DNA, RNA, ER, AGP, Mito           (the "slide" analog)
    n = observations   : (compound, dose) pairs            (the "subject" analog)

Observation unit = (compound, dose). We do NOT average over doses (each dose is a
different experiment, as in Haghighi et al.); we DO average the replicate wells
within each (compound, dose). Doses are matched between CP and L1000 by rank.

Only single-channel features are used (e.g. Intensity_MeanIntensity_DNA), because
they map cleanly onto the 5-channel axis. This drops shape (AreaShape, no channel)
and colocalization (Correlation, two channels) features by construction.

Steps:
  1. Load CP replicate-level augmented profiles (one row per well).
  2. Keep single-channel feature-cores present in all 3 compartments AND all 5
     channels (drops nuisance families + position/orientation sub-features).
  3. Normalize each feature per plate by z-scoring to that plate's negative
     control wells (cytominer "normalized").
  4. Average replicate wells -> one profile per (compound, dose).
  5. Keep compounds tested at exactly 6 doses; rank doses 1..6.
  6. Average L1000 replicate wells -> one expression profile per (compound, dose);
     rank doses 1..6.
  7. Match CP and L1000 observations by (compound, dose-rank).
  8. Assemble X (G, q, S, n) and the aligned expression matrix; cache everything.

Outputs (pickle): cell_painting/cache/cp_lincs_tensor.pkl
"""

import os
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.join(_HERE, '..', '..')
CP_PARQUET  = os.path.join(_ROOT, 'LINCS-Pilot1', 'CellPainting',
                           'replicate_level_cp_augmented.parquet')
L1K_PARQUET = os.path.join(_ROOT, 'LINCS-Pilot1', 'L1000',
                           'replicate_level_l1k.parquet')
IDMAP       = os.path.join(_ROOT, 'LINCS-Pilot1', 'idmap.xlsx')
CACHE_DIR   = os.path.join(_HERE, 'cache')
CACHE_PKL   = os.path.join(CACHE_DIR, 'cp_lincs_tensor.pkl')
os.makedirs(CACHE_DIR, exist_ok=True)

COMPARTMENTS  = ['Cells', 'Cytoplasm', 'Nuclei']
CHANNELS      = ['DNA', 'RNA', 'ER', 'AGP', 'Mito']
DROP_FAMILIES = {'Location', 'Number', 'Parent', 'Children', 'Neighbors'}
DROP_PATTERNS = ('Center_X', 'Center_Y', 'Orientation', 'EulerNumber')
N_DOSES = 6
# Robustize the z-scored tensor: a few features are near-constant in controls, so
# z-scoring explodes them. Drop feature-cores whose 99.5th-pct |z| exceeds CLIP,
# then clip survivors to +/- CLIP.
CLIP = 20.0


def _channel_cores(feature_cols):
    """Find single-channel feature-cores present in all 3 compartments AND all 5
    channels. Returns (cores, colmap) where colmap[(comp, core, channel)] is the
    original CellProfiler column name."""
    seen   = defaultdict(set)   # core -> set of (compartment, channel)
    colmap = {}                 # (compartment, core, channel) -> column name
    for c in feature_cols:
        parts = c.split('_')
        comp, fam = parts[0], parts[1]
        if comp not in COMPARTMENTS or fam in DROP_FAMILIES:
            continue
        if any(p in c for p in DROP_PATTERNS):
            continue
        rest = parts[1:]
        chans = [p for p in rest if p in CHANNELS]
        if len(chans) != 1:              # keep single-channel features only
            continue
        ch = chans[0]
        core = '_'.join(p for p in rest if p != ch)
        seen[core].add((comp, ch))
        colmap[(comp, core, ch)] = c
    full = {(cc, ch) for cc in COMPARTMENTS for ch in CHANNELS}   # 15 combos
    cores = sorted(core for core, s in seen.items() if full <= s)
    return cores, colmap


def _normalize_to_controls(df, morph_cols):
    """Per-plate z-score of each morphology column to the plate's control wells."""
    plate = df['Metadata_Plate'].values
    ctrl  = df[df['Metadata_pert_type'] == 'control']
    cmean = ctrl.groupby('Metadata_Plate')[morph_cols].mean()
    cstd  = ctrl.groupby('Metadata_Plate')[morph_cols].std()
    M  = cmean.reindex(plate).values
    Sd = cstd.reindex(plate).values
    Sd = np.where((Sd > 1e-9) & np.isfinite(Sd), Sd, 1.0)
    M  = np.where(np.isfinite(M), M, 0.0)
    Z  = (df[morph_cols].values - M) / Sd
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def build():
    print("Loading CP augmented profiles...")
    cp = pd.read_parquet(CP_PARQUET)
    feature_cols = [c for c in cp.columns if not c.startswith('Metadata')]
    print(f"  wells={len(cp)}  raw feature cols={len(feature_cols)}")

    cores, colmap = _channel_cores(feature_cols)
    G, S, q = len(COMPARTMENTS), len(CHANNELS), len(cores)
    # column order: compartment outer, core middle, channel inner
    #   -> a row's values reshape directly to (G, q, S)
    morph_cols = [colmap[(comp, core, ch)]
                  for comp in COMPARTMENTS for core in cores for ch in CHANNELS]
    print(f"  single-channel feature-cores (q)={q}  "
          f"-> {len(morph_cols)} columns  (G={G} compartments x S={S} channels)")

    # ── Normalize per plate to negative controls ─────────────────────────────
    print("Normalizing per plate to negative controls...")
    Z = _normalize_to_controls(cp, morph_cols)
    norm = pd.DataFrame(Z, columns=morph_cols)
    norm['pert_id']   = cp['Metadata_pert_id'].values
    norm['dose']      = cp['Metadata_pert_dose_micromolar'].values
    norm['pert_type'] = cp['Metadata_pert_type'].values

    # ── Treatments only; compounds with exactly N_DOSES doses ────────────────
    trt = norm[norm['pert_type'] == 'trt'].copy()
    dpc = trt.groupby('pert_id')['dose'].nunique()
    trt = trt[trt['pert_id'].isin(dpc[dpc == N_DOSES].index)].copy()
    print(f"  compounds with exactly {N_DOSES} doses: {trt['pert_id'].nunique()}")

    # ── Average replicate wells -> (compound, dose); rank doses ──────────────
    cp_agg = trt.groupby(['pert_id', 'dose'])[morph_cols].mean().reset_index()
    cp_agg['dose_rank'] = (cp_agg.groupby('pert_id')['dose']
                           .rank(method='dense').astype(int))

    # ── L1000: probes, symbol map, average replicates -> (compound, dose) ────
    print("Loading L1000 profiles + idmap...")
    l1k   = pd.read_parquet(L1K_PARQUET)
    idmap = pd.read_excel(IDMAP)
    probes = [c for c in l1k.columns if not c.startswith('Metadata')]
    probeset = set(probes)
    sym2probe = {}
    for _, r in idmap.iterrows():
        if r['probe_id'] in probeset and isinstance(r['symbol'], str):
            sym2probe.setdefault(r['symbol'], r['probe_id'])
    probe2sym = {p: s for s, p in sym2probe.items()}
    print(f"  probes={len(probes)}  mapped symbols={len(sym2probe)}")

    lk = l1k[l1k['Metadata_pert_type'] == 'trt'].copy()
    l1k_agg = (lk.groupby(['Metadata_pert_id', 'Metadata_pert_dose_micromolar'])[probes]
               .mean().reset_index()
               .rename(columns={'Metadata_pert_id': 'pert_id',
                                'Metadata_pert_dose_micromolar': 'dose'}))
    l1k_agg['dose_rank'] = (l1k_agg.groupby('pert_id')['dose']
                            .rank(method='dense').astype(int))

    # ── Match CP and L1000 observations by (compound, dose-rank) ─────────────
    merged = cp_agg.merge(l1k_agg, on=['pert_id', 'dose_rank'],
                          suffixes=('_cp', '_l1k'))
    n = len(merged)
    print(f"  matched (compound, dose) observations: {n}  "
          f"({merged['pert_id'].nunique()} compounds)")

    # ── Assemble tensor X[g, j, s, obs] ──────────────────────────────────────
    print(f"Assembling tensor ({G}, {q}, {S}, {n})...")
    X = np.zeros((G, q, S, n), dtype=np.float32)
    vals = merged[morph_cols].values.astype(np.float32)
    for obs in range(n):
        X[:, :, :, obs] = vals[obs].reshape(G, q, S)

    # ── Robustize: drop pathological cores, clip survivors ───────────────────
    p995 = np.percentile(np.abs(X), 99.5, axis=(0, 2, 3))
    keep = p995 <= CLIP
    X = np.clip(X[:, keep], -CLIP, CLIP)
    cores = [c for c, k in zip(cores, keep) if k]
    q = len(cores)
    print(f"  robustize: dropped {int((~keep).sum())} unstable cores, "
          f"kept q={q}; clipped to +/-{CLIP:g}")

    # Coerce all pandas/pyarrow-backed values to plain Python/numpy types so the
    # pickle carries no pandas extension dtypes (keeps it portable across pandas
    # versions on the cluster).
    cache = {
        'X': np.asarray(X, dtype=np.float32),     # (G, q, S, n)
        'features': [str(c) for c in cores],      # feature-cores (q,)
        'compartments': list(COMPARTMENTS),
        'channels': list(CHANNELS),
        'obs_compound': np.array([str(x) for x in merged['pert_id']], dtype=object),
        'obs_dose_rank': np.asarray(merged['dose_rank'], dtype=np.int64),
        'expr': np.asarray(merged[probes].values, dtype=np.float32),  # (n, 978)
        'probes': [str(p) for p in probes],
        'sym2probe': {str(k): str(v) for k, v in sym2probe.items()},
        'probe2sym': {str(k): str(v) for k, v in probe2sym.items()},
        'G': int(G), 'q': int(q), 'S': int(S), 'n': int(n),
    }
    with open(CACHE_PKL, 'wb') as f:
        pickle.dump(cache, f)

    print("\nSanity check:")
    print(f"  X shape        : {X.shape}  (G=compartment, q=core, S=channel, n=obs)")
    print(f"  X finite       : {np.isfinite(X).all()}")
    print(f"  X mean/std     : {X.mean():.3f} / {X.std():.3f}")
    print(f"  observations   : {n}  ({merged['pert_id'].nunique()} compounds x ~6 doses)")
    print(f"  GLRX probe     : {sym2probe.get('GLRX')}")
    print(f"Cached -> {CACHE_PKL}")


if __name__ == '__main__':
    build()
