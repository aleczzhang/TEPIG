"""Compare preselection arms using the panel pickles written by run_panel.py.

Why this comparison, not just "highest mean R^2":
  * Every arm runs the same seeds (42+100i in run_panel.py), so all arms see the
    IDENTICAL train/test well splits.  Per-run R^2 values are therefore paired,
    and paired tests cancel split-to-split luck -- by far the largest noise
    source at 5 runs.
  * A screening method is only defensible if no other method beats it on BOTH
    prediction and selection stability (Nogueira et al. 2018, JMLR) -- the
    stability/accuracy Pareto plot makes dominance visible.
  * Independent screens agreeing on the same biology is evidence the signal is
    real; we measure cross-arm agreement of the stable sets (pi >= 0.6) mapped
    back to raw CellProfiler feature names, both exact-name and feature-family
    level (correlated twins within a family should not count as disagreement).

Outputs
  results/arm_comparison_<tag>.csv      per-gene x arm x method summary table
  results/figures/pareto_<tag>.png      stability vs R^2, one panel per gene
  stdout                                tables, paired Wilcoxon tests, agreement

Usage
  python compare_arms.py [--cache cdrp_4plates.pkl] [--method TEPIG]
Picks up every results/panel_*_<tag>.pkl automatically, so re-running after the
pycytominer arm finishes folds it in with no changes.
"""
import argparse, csv, glob, itertools, os, pickle, re

import numpy as np
from scipy import stats

CHANNELS = {"DNA", "ER", "RNA", "AGP", "Mito", "Hoechst"}
ARM_ORDER = ["varclust", "pycytominer", "sis", "dcsis", "prune"]
ARM_COLOR = {  # fixed categorical slots; an arm keeps its hue everywhere
    "varclust": "#2a78d6", "pycytominer": "#eb6834",
    "sis": "#1baf7a", "dcsis": "#eda100", "prune": "#e87ba4",
}
METHOD_MARKER = {"TEPIG": "o", "clusso": "^", "naive": "s"}


def arm_name(path, tag):
    """panel_varclust262_cdrp_4plates.pkl -> varclust"""
    base = os.path.basename(path)[len("panel_"):-len(f"_{tag}.pkl")]
    return re.sub(r"\d+$", "", base)


def family(feat):
    """Cells_Intensity_MeanIntensity_ER -> Cells_Intensity_ER (compartment,
    measurement group, channel); shape features have no channel."""
    t = feat.split("_")
    chan = next((x for x in t[2:] if x in CHANNELS), "shape")
    return f"{t[0]}_{t[1]}_{chan}"


def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else float("nan")


def load_arms(tag):
    arms = {}
    for p in sorted(glob.glob(f"results/panel_*_{tag}.pkl")):
        with open(p, "rb") as f:
            arms[arm_name(p, tag)] = pickle.load(f)
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cdrp_4plates.pkl")
    ap.add_argument("--method", default="TEPIG",
                    help="method whose selections drive the agreement analysis")
    a = ap.parse_args()
    tag = a.cache.replace(".pkl", "")

    arms = load_arms(tag)
    if not arms:
        raise SystemExit(f"no results/panel_*_{tag}.pkl found")
    print(f"arms found: {', '.join(arms)}   (cache {a.cache})\n")
    genes = sorted(set.intersection(*({g for g in d if g != "_meta"} for d in arms.values())))
    methods = list(next(iter(arms.values()))[genes[0]]["r2"])
    order = [x for x in ARM_ORDER if x in arms] + [x for x in arms if x not in ARM_ORDER]

    # ---- 1. summary table -------------------------------------------------
    rows = []
    for g in genes:
        for arm in order:
            d = arms[arm][g]
            for m in methods:
                r2 = np.array(d["r2"][m], float)
                st = d["stability"][m]
                rows.append(dict(gene=g, arm=arm, method=m,
                                 r2_mean=r2.mean(), r2_sd=r2.std(ddof=1),
                                 phi=st["nogueira"], size=st["size_mean"],
                                 q=arms[arm]["_meta"]["q"]))
    print(f"{'gene':>9} {'arm':>12} {'method':>7} {'q':>5} "
          f"{'R2 mean+-sd':>16} {'phi':>6} {'|sel|':>6}")
    for r in rows:
        print(f"{r['gene']:>9} {r['arm']:>12} {r['method']:>7} {r['q']:>5} "
              f"{r['r2_mean']:>+8.3f} +-{r['r2_sd']:<5.3f} "
              f"{r['phi']:>+6.2f} {r['size']:>6.1f}")
    os.makedirs("results", exist_ok=True)
    out_csv = f"results/arm_comparison_{tag}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    print(f"\nmean over genes ({a.method}):")
    for arm in order:
        sel = [r for r in rows if r["arm"] == arm and r["method"] == a.method]
        print(f"  {arm:>12}: R2 {np.mean([r['r2_mean'] for r in sel]):+.3f}"
              f"   phi {np.mean([r['phi'] for r in sel]):+.3f}"
              f"   |sel| {np.mean([r['size'] for r in sel]):.1f}")

    # ---- 2. paired tests: arm vs arm, within a method ---------------------
    # same seed index = same split, so difference is purely the screen's doing
    print(f"\npaired arm comparisons, {a.method}, per-split differences pooled "
          f"over {len(genes)} genes (Wilcoxon signed-rank):")
    for a1, a2 in itertools.combinations(order, 2):
        diffs = np.concatenate([
            np.array(arms[a1][g]["r2"][a.method]) - np.array(arms[a2][g]["r2"][a.method])
            for g in genes])
        wins = int((diffs > 0).sum())
        try:
            pval = stats.wilcoxon(diffs).pvalue
        except ValueError:
            pval = float("nan")
        print(f"  {a1:>12} vs {a2:<12} median dR2 {np.median(diffs):+.3f}  "
              f"wins {wins}/{len(diffs)}  p={pval:.3f}")

    # ---- 3. paired tests: TEPIG vs naive, within each arm -----------------
    if "naive" in methods and a.method != "naive":
        print(f"\n{a.method} vs naive within each arm (same splits):")
        for arm in order:
            diffs = np.concatenate([
                np.array(arms[arm][g]["r2"][a.method]) - np.array(arms[arm][g]["r2"]["naive"])
                for g in genes])
            try:
                pval = stats.wilcoxon(diffs).pvalue
            except ValueError:
                pval = float("nan")
            print(f"  {arm:>12}: median dR2 {np.median(diffs):+.3f}  "
                  f"wins {int((diffs > 0).sum())}/{len(diffs)}  p={pval:.3f}")

    # ---- 4. cross-arm agreement of the stable sets ------------------------
    print(f"\ncross-arm agreement of {a.method} stable sets (pi>=0.6), mapped to "
          f"raw feature names; family = compartment_group_channel:")
    stable = {arm: {g: [arms[arm][g]["features"][i]
                        for i in arms[arm][g]["stability"][a.method]["stable_0.6"]]
                    for g in genes} for arm in order}
    for a1, a2 in itertools.combinations(order, 2):
        nm = np.nanmean([jacc(stable[a1][g], stable[a2][g]) for g in genes])
        fm = np.nanmean([jacc({family(x) for x in stable[a1][g]},
                              {family(x) for x in stable[a2][g]}) for g in genes])
        print(f"  {a1:>12} vs {a2:<12} name-Jaccard {nm:.2f}   family-Jaccard {fm:.2f}")
    print("\nconsensus families (selected stably by >=2 independent arms):")
    for g in genes:
        fam_votes = {}
        for arm in order:
            for fam in {family(x) for x in stable[arm][g]}:
                fam_votes.setdefault(fam, set()).add(arm)
        cons = sorted((f for f, v in fam_votes.items() if len(v) >= 2),
                      key=lambda f: -len(fam_votes[f]))
        print(f"  {g:>9}: " + (", ".join(f"{f}({len(fam_votes[f])})" for f in cons) or "-"))

    # ---- 5. Pareto figure -------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("results/figures", exist_ok=True)
    fig, axes = plt.subplots(1, len(genes), figsize=(3.1 * len(genes), 3.4),
                             sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, g in zip(np.atleast_1d(axes), genes):
        ax.set_facecolor("white")
        ax.axhline(0, color="#d8d8d3", lw=1, zorder=0)
        ax.grid(color="#ecece7", lw=0.6, zorder=0)
        for r in rows:
            if r["gene"] != g:
                continue
            emph = r["method"] == a.method
            ax.errorbar(r["phi"], r["r2_mean"], yerr=r["r2_sd"] if emph else None,
                        fmt=METHOD_MARKER[r["method"]], ms=8 if emph else 5,
                        color=ARM_COLOR.get(r["arm"], "#666"),
                        alpha=1.0 if emph else 0.35, mec="white", mew=1,
                        elinewidth=1, ecolor="#bbbbb4", capsize=2, zorder=3 if emph else 2)
        ax.set_title(g, fontsize=11, color="#333")
        ax.set_xlabel("stability $\\Phi$ (Nogueira)", fontsize=9, color="#555")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("test $R^2$ (mean $\\pm$ sd)", fontsize=9, color="#555")
    handles = ([plt.Line2D([], [], marker="o", ls="", color=ARM_COLOR[arm], mec="white",
                           label=arm) for arm in order] +
               [plt.Line2D([], [], marker=METHOD_MARKER[m], ls="", color="#666",
                           label=m) for m in methods])
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Stability vs accuracy by preselection arm  ({tag}, "
                 f"{arms[order[0]]['_meta']['runs']} splits; big markers = {a.method})",
                 fontsize=11, color="#333")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out_png = f"results/figures/pareto_{tag}.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out_csv} and {out_png}")


if __name__ == "__main__":
    main()
