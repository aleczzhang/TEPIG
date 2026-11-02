"""
genes.py
--------
Outcome gene sets for the LINCS Cell Painting analysis.

TOP10  : the 10 most linearly-predictable LINCS genes (Lasso R^2 above the 99th
         percentile of all 978 genes, per Haghighi et al. Sup_dataset_4) — the
         individual-gene outcomes for the heatmap.
CELL_CYCLE : canonical cell-cycle / proliferation genes present in the L1000
         panel; combined into one continuous score (mean of z-scored expression)
         as the gene-set outcome. Genes absent from the panel are skipped at
         run time.
"""

# Top 10 by LINCS-Lasso R^2 (99th-percentile threshold; 4 are cell-cycle genes)
TOP10 = ['GLRX', 'RAP1GAP', 'IGFBP3', 'ATF6', 'CDKN1A',
         'DDIT4', 'CCNA2', 'SMC4', 'TSC22D3', 'MELK']

# Canonical cell-cycle / proliferation markers (intersected with the panel at run time)
CELL_CYCLE = ['CCNA2', 'CCNB1', 'CCNB2', 'BIRC5', 'TOP2A', 'UBE2C', 'MELK',
              'CDKN1A', 'AURKA', 'AURKB', 'PLK1', 'POLE2', 'POLD1', 'SMC4',
              'CDC20', 'KIF14', 'PCNA']

# All outcomes to run (individual genes + the cell-cycle set score)
ALL_OUTCOMES = TOP10 + ['CELLCYCLE']
