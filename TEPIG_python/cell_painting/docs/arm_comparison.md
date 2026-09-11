# Comparing preselection arms: methodology, results, and recommendation

*(4-plate CDRP cache: plates 24277 26724 26071 26058; n = 1,280 wells, all
L1000-matched; raw q = 3,579. Five genes × 5 train/test splits per arm.
Produced by `compare_arms.py`; figure at `results/figures/pareto_cdrp_4plates.png`,
table at `results/arm_comparison_cdrp_4plates.csv`. 5 Sep 2026.)*

## Why not just rank arms by mean R²?

At 5 splits the R² standard deviations (±0.05–0.15) are as large as the
between-arm differences, so a mean-R² ranking mostly ranks noise. We use three
complementary comparisons instead, each answering a different question:

**1. Paired per-split tests (is the difference real?).** Every arm runs the
same seeds (42, 142, 242, …), so all arms see *identical* train/test well
splits. The R² difference between two arms on the same split is therefore
purely the screen's doing — split-to-split luck, the dominant noise source,
cancels. We pool the per-split differences over genes (25 pairs) and apply the
Wilcoxon signed-rank test. This detects effects the marginal means hide.

**2. The stability–accuracy Pareto plot (is the arm defensible?).** With no
true support on real data, TPR/FPR are unavailable; the accepted replacement
(Nogueira, Sechidis & Brown 2018, JMLR) is selection *stability* Φ — the
chance-corrected agreement of the selected sets across resamples — read jointly
with accuracy. An arm is defensible iff no other arm beats it on **both** axes.
One point per arm×method per gene: Φ on x, test R² on y.

**3. Cross-arm agreement of stable sets (is the biology real?).** The arms
were deliberately made fully independent (no shared prune), so when two
different algorithms stably select the same feature, that is corroborating
evidence rather than pipeline artifact. We take each arm's stable set
(selection probability π ≥ 0.6 across splits), map indices back to raw
CellProfiler feature names, and compare at the *feature-family* level
(compartment × measurement group × channel) so that |r| > 0.9 twins within a
family don't count as disagreement.

## Results

**Paired tests.**

| comparison (TEPIG R², 25 paired splits) | median ΔR² | wins | pooled Wilcoxon p | gene-level p |
|---|---|---|---|---|
| pycytominer vs varclust | +0.028 | 17/25 | 0.016 | **0.035** (better in 5/5 genes) |
| pycytominer vs sis | +0.009 | 16/25 | 0.13 | 0.45 (better in 4/5 genes) |
| pycytominer vs dcsis | +0.016 | 19/25 | 0.003 | — |
| sis vs dcsis | +0.017 | 21/25 | < 0.001 | 0.052 (≥0 in 5/5 genes) |
| varclust vs sis | −0.004 | 12/25 | 0.62 | — |
| varclust vs dcsis | +0.011 | 13/25 | 0.47 | — |

The clearest result: **SIS beats DC-SIS on 21 of 25 splits** — invisible in
the means (0.163 vs 0.142). The 25 pairs are not independent (5 genes × 5
splits; splits within a gene share training wells), so the pooled p-value
overstates certainty. Respecting the grouping: per-gene mean ΔR² is
CCNA2 +0.037, CELLCYCLE −0.000, CDKN1A +0.029, DDIT4 +0.032, GLRX +0.005
(positive or zero in all five genes; t-test on the 5 gene-level means
p = 0.052). Honest summary: *consistent in direction across every gene,
borderline significant at gene level* — with DC-SIS also costing ~20× the
screening compute, that is still enough to retire it, but "p < 0.001" is not
the defensible claim. varclust vs sis shows no consistent direction at either
level (gene means +0.02, −0.01, −0.07, +0.04, −0.05). TEPIG vs naive is a tie
inside every arm (12/25, 12/25, 9/25 wins; all pooled p > 0.15) — resolving
that needs more plates/splits, not a different screen.

**Pareto.** Mean over genes, TEPIG: **pycytominer R² 0.181 / Φ 0.32** / 10
features; varclust R² 0.149 / Φ 0.33 / 13; sis R² 0.163 / Φ 0.31 / 6; dcsis
R² 0.142 / Φ 0.26 / 3. pycytominer sits on or above the Pareto front in 4 of 5
genes (CDKN1A is its one weak cell: Φ = 0.08) and owns the study's best single
result — DDIT4 R² 0.290 ± 0.098, Φ 0.42. dcsis is weakly dominated everywhere.
Naive lasso remains the most *stable* selector (Φ 0.4–0.6) at similar R².

**Agreement.** The supervised screens and pycytominer cluster together
(family-Jaccard: sis–dcsis 0.57, pycytominer–sis 0.43, pycytominer–dcsis 0.30)
while varclust agrees with everyone only at 0.07–0.12 — its cluster
representatives are a genuinely different (but similarly predictive)
parameterization of the same signal. Consensus families (stably selected by
≥ 2 of the 4 independent arms) are biologically coherent:

- **CDKN1A (p21, DNA-damage response): `Cells_Correlation_DNA` picked by all
  FOUR arms** — the strongest cross-validation of signal in the study.
- DDIT4: `Cytoplasm_RadialDistribution_RNA` (3 arms); `Cells/Nuclei_AreaShape`,
  `Cells_Correlation_ER`, `Cytoplasm_Correlation_ER` (2 arms).
- GLRX: `Cytoplasm_Intensity_RNA`, `Cells_Intensity_AGP` (3 arms);
  `Nuclei_Granularity_ER` (2 arms).
- CCNA2: `Cytoplasm_RadialDistribution_AGP` (3 arms); `Nuclei_Correlation_RNA`,
  `Nuclei_Intensity_AGP`, `Nuclei_Granularity_AGP` (2 arms).
- CELLCYCLE: `Cytoplasm_RadialDistribution_AGP`, `Cytoplasm_Granularity_ER`
  (2 arms).

## Recommendation (updated 6 Sep, all four arms in): pycytominer

The community-standard cut wins on the evidence, not just on convention:

1. **Accuracy.** Best mean R² (0.181) and the only arm with a grouping-robust
   paired win: better than varclust in **5/5 genes** (gene-level p = 0.035,
   pooled p = 0.016). Also ahead of sis in 4/5 genes (n.s.). Owns the study's
   best cell: DDIT4 R² 0.290 ± 0.098 at Φ = 0.42.
2. **Stability.** Mean Φ 0.32 — statistically indistinguishable from
   varclust's 0.33, despite keeping 2.4× the features (624 vs 262). The worry
   that r ≤ 0.9 twins would destabilize lasso selection did not materialize.
3. **External justification.** It is the field's default (Caicedo 2017 best
   practices; the pycytominer *Nat Methods* paper), and the exact recipe used
   by Haghighi 2022 — the source of our L1000 outcomes — making results
   directly comparable to the literature. Unsupervised/outcome-blind, so all
   outcome-driven selection is attributable to TEPIG (transductive caveat in
   the Q&A below applies to varclust equally).
4. **Costs.** ~5× fit compute (q = 624: TEPIG ≈ 4.4 min, CLUSSO ≈ 8.8 min per
   fit) — a Zaratan-sizing concern, not a validity one. Its one weak cell is
   CDKN1A stability (Φ = 0.08).

Supporting roles: keep **varclust** as the interpretability lens (named
correlated-feature clusters; agreement with pycytominer is only ~0.1, so it
offers a genuinely different view of the same signal) and **sis** as the
supervised benchmark. **Retire dcsis** (dominated; ~20× screen cost).
TEPIG vs naive remains a tie in every arm — the question for the bigger run.

## Exact definitions and provenance (reviewer Q&A, 5 Sep 2026)

1. **SIS cutoff d.** `d = ⌊n_train / ln(n_train)⌋`, computed from the
   *training-fold* size inside each split (`preselect.make_supervised_screen`
   calls `sis_d(len(y_tr))`). Splits are 80/20 (`TEST_FRAC = 0.2` in
   run_gene.py), so n_train = 1,024 and d = ⌊1024/6.93⌋ = 147 — not 179 from
   the full n = 1,280.
2. **Is q < n required?** Not mathematically. Plain lasso, and TEPIG/CLUSSO's
   proximal-gradient solvers, run fine at q > n (the sis/dcsis arms literally
   start from q = 3,579). The motivations are practical: (a) at q > n_train
   test-R² estimates were too noisy to tune anything (the one-plate varclust
   cutoff sweep failed for exactly this reason), (b) fit time grows with q
   (q = 624 → ~7 min/TEPIG fit vs ~40 s at q = 262), (c) selection stability
   degrades in the ultra-redundant full space. Earlier phrasing "lasso-type
   models want q < n" was loose; q < n_train is a soft target, not a
   constraint of the estimators.
3. **Unsupervised screens: global, i.e. transductive.** varclust and
   pycytominer are fitted once on the full 1,280-well matrix
   (`preselect.unsupervised_select`), before splitting. They never see y
   (outcome-blind) but do use test-fold covariates when learning clusters /
   correlation filters. Defensible as a fixed preprocessing step, and standard
   practice in this field, but the strictest inductive protocol would re-learn
   the screen on training wells per split and apply the frozen feature list to
   the test fold. Worth doing once on Zaratan to confirm it doesn't move the
   numbers.
4. **Paired p-values.** `scipy.stats.wilcoxon` on the 25 pooled per-split
   differences, which ignores gene grouping and shared training wells across
   splits — see the corrected sis-vs-dcsis paragraph above for the
   grouping-aware version (gene-level t-test p = 0.052). At 5 splits, per-gene
   tests are underpowered; the Zaratan run (≥ 10 splits) enables a proper
   per-gene Wilcoxon and a mixed-effects check.
5. **Stability Φ.** Nogueira et al. (2018) estimator, `selection_metrics.nogueira`:
   with M = 5 runs and q features, p̂_f = selection frequency of feature f,
   s²_f = M/(M−1)·p̂_f(1−p̂_f), k̄ = mean selected-set size, then
   Φ = 1 − mean_f(s²_f) / [(k̄/q)(1 − k̄/q)]. Computed per gene, per method,
   across that gene's 5 splits (within one arm, so q is the arm's feature
   space). Reported "mean Φ" is the unweighted mean over the 5 genes.
6. **varclust k = 262: fixed a priori, not tuned.** Carried over from the
   one-plate phase (262 = the feature count the old |r| ≤ 0.95 prune produced,
   adopted as the default k in run_panel.py). The ClustOfVar bootstrap-stability
   curve on one plate was smooth with no elbow — it only bounds a defensible
   range (~250–400). k was NOT chosen using validation performance of these
   runs (no leakage), and not matched to the other arms' counts (they differ:
   147/624). Fairness caveat stands: a k-sweep on the multi-plate data is the
   planned resolution.
7. **"Overlap ~0.1" = Jaccard similarity** |A∩B| / |A∪B| between the two arms'
   stable sets (features with selection probability π ≥ 0.6 across the 5
   splits), computed per gene and averaged unweighted over genes; reported at
   exact-name and feature-family level (family = compartment × measurement
   group × channel).
8. **pycytominer recipe, exactly.** pycytominer **1.7.1**,
   `feature_select(df, features='infer', operation=['variance_threshold',
   'correlation_threshold', 'drop_na_columns', 'blocklist'], corr_threshold=0.9)`.
   Package defaults therefore apply: variance_threshold freq_cut = 0.05,
   unique_cut = 0.01; correlation pearson, threshold 0.9, tie-break = of each
   over-threshold pair, drop the member with the larger summed |r| to all
   features; blocklist = the package's default
   `pycytominer/data/blocklist_features.txt`. Applied once to the combined
   4-plate well-level matrix *after* per-plate DMSO z-scoring (not plate-wise).
   Result on this cache: 3,579 → 624 features.

## Caveats

- 5 splits ⇒ Φ itself is noisy (bootstrap CIs are wide); the Zaratan run should
  use ≥ 10 splits.
- Wells from the same plate appear in both train and test; leave-one-plate-out
  (cache has `obs_plate`) is the honest batch-transfer test — planned for the
  8-plate Zaratan run.
