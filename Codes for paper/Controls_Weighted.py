"""Controls study on NODE-WEIGHTED Erdos-Renyi graphs (maximum-weight clique problem).

Same study as Controls_Graphs.py (N_GRAPHS different graphs, one run per graph, all the samplers of
Controls_Comparison.py, paired differences to D-GBS), but every node gets a weight drawn uniformly in [1, WEIGHT_MAX]
(graph_setup.py). The weights enter the D-GBS state exactly as in the library (omega = c (1 + alpha diag(w)), displacement
profile from the weights), and the target is the maximum-WEIGHT clique, which need not be the planted clique.
Note that the displacement profile (hence Coherent only, Thermal, Phase-random., Independent nodes) is also weight-aware;
only 'Uniform' and plain 'GBS' ignore the weights.

Usage: WEIGHT_MAX=3 python Controls_Weighted.py      (all Controls_Graphs.py settings apply; defaults below)
Outputs: same as Controls_Graphs.py, with '_w<WEIGHT_MAX>' in the file names.
"""
import os
import runpy

for k, v in dict(WEIGHT_MAX="3", EDGE_PROB="0.1", LOSS="0.2", NCOH="3", N_ITER="8", NSAMPLES="500").items():
    os.environ.setdefault(k, v)
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Controls_Graphs.py"), run_name="__main__")
