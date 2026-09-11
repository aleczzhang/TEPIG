# Feature pre-screening for the CDRP single-cell TEPIG run — survey of existing methods

Context: after clustering we have ~1,950 morphology features (after a constant-feature
filter and a |r| <= 0.95 greedy prune of the ~3,600 raw CellProfiler columns) for 320
wells/plate. The professor asked for *existing* pre-selection / screening methods
(unsupervised and supervised, some with resampling inside), the cutoffs practitioners
actually use, and alternatives to a plain correlation threshold. Nothing below is our own
method; every item names the paper and the off-the-shelf implementation.

All package checks were done in `.venv_inspect` (Python 3.14): **pycytominer 1.7.1,
skfeature-chappers 1.2.1 (scikit-feature), dcor 0.7** install and run.

---

## 1. What the Cell Painting community itself does (unsupervised)

The standard image-based profiling pipeline (Caicedo et al. 2017, *Nat Methods*
[link](https://www.nature.com/articles/nmeth.4397)) has an explicit "feature selection"
level (4b) after normalization. It is implemented in **pycytominer** (Serrano et al.,
*Nat Methods* 2025 / arXiv:2311.13417) as `pycytominer.feature_select`, and it is what
the Broad `profiling-recipe` runs by default
([recipe](https://github.com/cytomining/profiling-recipe)). Operations and defaults
([source](https://github.com/cytomining/pycytominer/blob/main/pycytominer/feature_select.py),
[docs](https://pycytominer.readthedocs.io/en/stable/pycytominer.html)):

| operation | what it removes | default cutoff |
|---|---|---|
| `variance_threshold` | near-constant features: ratio of 2nd-most-common / most-common value < `freq_cut`, or #unique/#samples < `unique_cut` | `freq_cut=0.05`, `unique_cut=0.01` |
| `correlation_threshold` | one of every pair with \|r\| above the cutoff (drops the member with the higher mean \|r\| to everything else — same greedy rule as caret::findCorrelation and our `prune_correlated`) | `corr_threshold=0.9`, Pearson |
| `drop_na_columns` | features with > `na_cutoff` missing | `na_cutoff=0.05` |
| `blocklist` | a curated list of known-unreliable features (e.g. `Correlation_Manders_*`, `Correlation_RWC_*`, `Granularity_14–16`, Costes) | Broad standard blocklist file |
| `drop_outliers` | features with any \|value\| > `outlier_cutoff` after normalization | `outlier_cutoff=500` |
| `noise_removal` | features whose within-replicate-group sd exceeds `noise_removal_stdev_cutoff` (needs perturbation-group labels; uses replicates, not the outcome) | no default; user-set |

The recipe order is `variance_threshold → correlation_threshold → drop_na_columns →
blocklist`. In the rosetta resource we take the L1000 outcomes from (Haghighi et al. 2022,
*Nat Methods*, [repo](https://github.com/carpenter-singh-lab/2022_Haghighi_NatureMethods)),
this exact pipeline took CDRP-BBBC047 from **1,565 normalized well-level features to 727**
"variable-selected" features (LINCS-Pilot1: 1,670 → 119). Way et al. 2022 (*Cell Systems*,
[link](https://www.cell.com/cell-systems/fulltext/S2405-4712(22)00402-1)) use the same
recipe for the Cell Painting–L1000 comparison.

**Takeaway.** The field's standard answer to "which features" is unsupervised: variance
filter + \|r\| ≤ 0.9 + blocklist, giving ~700 features on this very dataset. Our current
0.95 prune is *looser* than the community default; running `pycytominer.feature_select`
on our well-level (or cell-level) matrix with defaults is the most defensible "existing
method" and costs nothing. `noise_removal` is the one community operation that uses
resampling-like information (replicate wells) without touching y.

## 2. General unsupervised feature-screening methods (filter methods)

Taxonomy and comparison: Solorio-Fernández, Carrasco-Ochoa & Martínez-Trinidad 2020,
*Artif Intell Rev* 53:907–948 ([link](https://link.springer.com/article/10.1007/s10462-019-09682-y)).
Implementations: **scikit-feature** (Li et al. 2018, *ACM Comput Surv*; pip
`skfeature-chappers`; [algorithm list](https://jundongl.github.io/scikit-feature/algorithms.html)).

| method | idea | package call | cutoff in practice |
|---|---|---|---|
| Max-variance / `low_variance` | keep high-variance features (after control z-scoring: "responds to compounds") | `sklearn.feature_selection.VarianceThreshold`, `skfeature...low_variance` | top-k, or variance > threshold |
| **Laplacian score** (He, Cai & Niyogi 2005) | keep features that vary smoothly over the kNN graph of samples (locality preserving) | `skfeature.function.similarity_based.lap_score` | rank; keep top-k (no theory for k) |
| **SPEC** (Zhao & Liu 2007) | spectral generalization of Laplacian score | `skfeature...similarity_based.SPEC` | rank; top-k |
| **MCFS** (Cai, Zhang & He 2010) | spectral embedding + L1 regression, selects jointly (handles redundancy) | `skfeature...sparse_learning_based.MCFS` (`n_selected_features`, `n_clusters`) | k given |
| **UDFS** (Yang et al. 2011) / **NDFS** (Li et al. 2012) | L2,1-regularized discriminative / nonnegative spectral selection | `skfeature...sparse_learning_based.UDFS`, `NDFS` | k given; slower, parameter-sensitive |
| **ClustOfVar** (Chavent et al. 2012, *JSS* 50(13)) | hierarchical clustering of variables, one synthetic (or real) representative per cluster; **bootstrap stability of the partition (ARI) to choose the number of clusters** | R `ClustOfVar` (`hclustvar`, `stability`); our `varclust.py` is a port of the real-representative variant | k by the bootstrap-ARI curve |
| Principal Feature Analysis (Lu et al. 2007) | k-means on PCA loading rows, keep nearest real feature per cluster | no maintained package; ~20 lines on top of sklearn | PCs explaining 90–95 % variance |

A recent evaluation (arXiv:2601.08257, "On Evaluation of Unsupervised Feature Selection
for Pattern Classification") notes the practical split: similarity-score filters
(variance, Laplacian, SPEC) are fast but ignore redundancy; the sparse-regression ones
(MCFS/UDFS/NDFS) handle redundancy jointly but are costlier and parameter-sensitive.
Because Cell Painting's problem is *redundancy* far more than noise (Way 2022: CP
features are much more inter-correlated than L1000 genes), a redundancy-aware method
(correlation cut, ClustOfVar, MCFS) is the sensible default, possibly *after* a
variance/Laplacian filter.

## 3. Supervised screening (marginal utility → keep top d)

These use y, so in our pipeline they must run **inside each train/test split on the
training wells only** (otherwise the test R² is leaked). Reference review: Fan & Lv,
"Sure Independence Screening" (2018 review, [pdf](http://faculty.marshall.usc.edu/jinchi-lv/publications/SISReview-FL18.pdf)).

| method | statistic | reference | implementation |
|---|---|---|---|
| **SIS** | \|Pearson corr(x_j, y)\| | Fan & Lv 2008, *JRSS-B* 70:849 ([link](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2008.00674.x)) | R `SIS`, R `MFSIS::SIS`; in Python it is one `np.corrcoef` + `argsort` |
| **DC-SIS** | distance correlation (model-free, catches non-linear dependence) | Li, Zhong & Zhu 2012, *JASA* ([arXiv](https://arxiv.org/abs/1205.4701)) | R `MFSIS::DCSIS`; Python **`dcor`** (Ramos-Carreño 2023; `dcor.rowwise(dcor.distance_correlation, ...)`) |
| SIRS, MV-SIS, Kolmogorov filter, projection-correlation SIS, Ball-correlation SIS, MDC-SIS… | other model-free utilities | catalogued in R **MFSIS** ([CRAN](https://cran.r-project.org/web/packages/MFSIS/MFSIS.pdf), [index](https://rdrr.io/cran/MFSIS/)) | R only |
| ISIS | iterate SIS with residuals to recover jointly-important features | Fan & Lv 2008 | R `SIS` |

**Cutoff in practice:** Fan & Lv's rule is d = ⌊n / log n⌋ (or the more conservative
n − 1). For us: n_train = 256 → d = 46; with 8 plates (n_train ≈ 2,050) → d ≈ 268. This
is the number every SIS paper and package uses as default.

## 4. Screening *with resampling inside*

| method | idea | cutoff | implementation |
|---|---|---|---|
| **Stability selection** (Meinshausen & Bühlmann 2010, *JRSS-B* [link](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/j.1467-9868.2010.00740.x)) | run any selector (lasso, SIS…) on many random half-samples; keep features whose selection frequency ≥ π_thr; comes with a bound on the expected number of false selections | π_thr ∈ [0.6, 0.9] (their recommendation; results insensitive within the range) | R **`stabs`** (Hofner; [GitHub](https://github.com/hofnerb/stabs)), Python **`stability-selection`** (scikit-learn-contrib, [GitHub](https://github.com/scikit-learn-contrib/stability-selection); not on PyPI, install from GitHub) |
| Complementary-pairs stability selection (Shah & Samworth 2013) | same, with paired complementary halves and a tighter error bound | same π range | `stabs(sampling.type="SS")` |
| ClustOfVar bootstrap stability | resample samples, recluster the variables, ARI vs full-data partition to pick the number of clusters | choose k on the ARI plateau | R `ClustOfVar::stability`; `varclust.bootstrap_stability` |
| Bootstrap-ranked filters | average a filter's rank over bootstrap resamples (ensemble feature selection) | top-k by mean rank | trivial wrapper around any filter |

Stability selection is the natural "resampling inside screening" the professor described,
and it also gives a **per-feature selection probability**, which is the quantity our
repeated-runs frequency table approximates (see §5).

## 5. Metric for selection *consistency* on real data (replacing TPR/FPR)

TPR/FPR need the true support, which real data doesn't have. The feature-selection
literature measures **stability** of the selected set across resamples/runs instead:

| measure | property | reference / code |
|---|---|---|
| mean pairwise Jaccard (what `stability_check.py` reports) | intuitive; **not chance-corrected** (selecting most features every time scores high) | — |
| Kuncheva consistency index (2007) | chance-corrected; assumes equal set sizes (generalized versions exist) | R `stabm::stabilityKuncheva` |
| **Nogueira–Sechidis–Brown Φ** (2018, *JMLR* 18(174):1–54, [paper](https://jmlr.org/papers/v18/17-514.html)) | chance-corrected, handles unequal set sizes, has variance, **confidence intervals and hypothesis tests** (e.g. "is TEPIG's stability > CLUSSO's?"); now the default in the stability literature | authors' Python package (`stability/` in [nogueirs/JMLR2018](https://github.com/nogueirs/JMLR2018): `getStability(Z)`, `confidenceIntervals`, `hypothesisTestT`); R `stabm` (Bommert 2021, *JOSS*) |
| per-feature selection probability π_f | interpretable on its own; the stability-selection quantity; "stable set" = {f : π_f ≥ 0.6–0.9} | Meinshausen & Bühlmann 2010 |

`selection_metrics.py` in this folder just codes these formulas (Jaccard, generalized
Kuncheva, Nogueira Φ with a bootstrap CI, π_f) so `repeated_runs.py` can print them; if
we prefer to call the authors' code, their `getStability` takes exactly the same 0/1
(runs × features) matrix.

## 6. Recommendation for our pipeline

Design decision (1 Sep): the arms are run **independently** on the same cache — the
extractor now does hygiene only (drop coordinates/counts/IDs and constant features; no
cell-level correlation prune), so every arm starts from the same ~3,590 raw morphology
features and nothing is chained. The GMM cell clustering (which defines the tensor's
G=2 mode) necessarily runs before any arm and is shared; it clusters on all features.

1. **Unsupervised, community standard, first:** `pycytominer.feature_select` with defaults
   (variance_threshold, correlation_threshold 0.9, blocklist) on the well-level matrix.
   Expect ~700 features (Haghighi's number for CDRP). This replaces "some correlation
   threshold" with the field's published cutoff and is defensible to a reviewer.
2. **Unsupervised, redundancy-aware, to reach q < n:** ClustOfVar/varclust (already in place)
   with k from the bootstrap-ARI curve — the multi-plate run should make that curve
   informative — or MCFS from scikit-feature as an independent check.
3. **Supervised screening as a separate arm:** SIS or DC-SIS with d = n/log n on the training
   wells inside each split (via `dcor` for DC-SIS), and stability selection (π ≥ 0.6) as the
   resampled version. Report these as a second setting so the comparison "unsupervised vs
   supervised preselection" the professor asked about is a table, not an assumption.
4. **Consistency metric:** Nogueira Φ with CI (plus π_f) instead of TPR/FPR on real data;
   keep TPR/FPR for the simulations where the support is known.

Open questions for the professor: (a) is q < n a hard requirement for the preselected set
(then k = n/log n or the ARI-chosen k), or is q ≈ n acceptable for TEPIG's group lasso?
(b) should the blocklist be applied at the single-cell level before clustering, as the
community does, or only before regression?
