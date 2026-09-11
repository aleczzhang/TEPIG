# TEPIG_python — layout

```
core/            shared estimator machinery, imported by everything else
  Mainfunction_albet.py   CLUSSO alternating structured lasso (+ _glmnet_lasso shim)
  SLasso_MSE.py           5-fold CV loss for the structured lasso
  K_prdu.py               Kronecker product of the two rank-one factors
  mat_vec_prd.py          tensor/vector contractions
  utils.py                tubule loading, naive averaging, correlation pruning

simulation/      simulation studies (the papers' Section 4)
  simulation_synthetic.py   fully synthetic, CLUSSO 4.1 protocol, rank-2 true B
  simulation_rankone.py     semi-synthetic: real Coimbra X, synthetic rank-1 B
  simulation_bootstrap.py   bootstrap variant
  summarize_synthetic.py    combines the per-setting .pkl files into a summary table

coimbra/         real kidney transplant data (Centro Hospitalar e Universitario de Coimbra)
  gmm_clustering.py       pooled G=2 GMM over all tubules -> subject tensors
  real_data_analysis.py   TEPIG / CLUSSO / naive on the Coimbra cohort
  explore_features.py     feature correlation screening and heatmaps

figures/         plot generation (reads outputs/, writes outputs/figures/)
  plot_synthetic_poster.py       poster subset (4 panels)
  plot_synthetic_all_figures.py  full grid, all metrics
  plot_real_data.py              Coimbra result figures

tests/           local smoke tests -- single-rep drivers, not pytest
  test_synthetic.py, test_bootstrap.py, test_proxgrad.py

slurm/           HPC job scripts (run with cwd = TEPIG_python/)

cell_painting/   Cell Painting extension (LINCS-Pilot1 well-level + CDRP single-cell).
                 Self-contained; has its own slurm scripts, cache/, results/, figures/.
  cdrp_extract.py         plate sqlite(s) -> (G,q,S,n) tensor cache; --plates for multi-plate
  varclust.py             ClustOfVar-style feature clustering (one real rep per cluster)
  stability_check.py      one_seed(): fits TEPIG/CLUSSO/naive for one split (times each)
  repeated_runs.py        mean+/-sd R^2 over runs, selection frequencies, stability metrics
  run_panel.py            repeated_runs over several genes in one process
  preselect.py            pre-screening arms via existing packages: prune, varclust,
                          pycytominer (community cut), sis / dcsis (supervised, in-split)
  selection_metrics.py    Jaccard / Kuncheva / Nogueira stability, selection probabilities
  compute_log.py          wall-time + peak-RSS logging -> results/compute_log.csv
  compute_plan.py         sizes the overnight multi-plate job from the logged timings
  run_cdrp_multi.slurm    12 h Zaratan job: extract P plates + gene panel
  run_local_overnight.sh  same on the Mac (nohup caffeinate -i ./run_local_overnight.sh)
  compare_arms.py         compares arms: Pareto (phi vs R^2), paired Wilcoxon on
                          shared splits, cross-arm stable-set agreement
  docs/feature_screening_survey.md  existing pre-screening methods, packages, cutoffs
  docs/arm_comparison.md  methodology + results + recommendation (varclust) for the arms
```

## Conventions

Scripts are run from any directory; all data and output paths are resolved relative to
`__file__`, so `python simulation/simulation_synthetic.py` and
`cd simulation && python simulation_synthetic.py` both work.

Anything that needs the shared estimators puts `core/` on the path first:

```python
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))
from Mainfunction_albet import Mainfunction_albet
```

Results are written outside this tree, to the repo-level `outputs/` directory
(`data/`, `results/`, `summaries/`, `figures/`, `reference/`). `cell_painting/` is the
exception: it keeps its own `cache/`, `results/`, and `figures/` locally.

## Known issue

`slurm/run_simulation.slurm` invokes `python simulation.py`, which does not exist in this
tree. It was already stale before the reorganization and was moved as-is rather than
guessed at.
