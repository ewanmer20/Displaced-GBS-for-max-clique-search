"""Graph choice shared by Controls_Comparison.py and Correlation_Diagnostic.py (selected with environment variables).

GRAPH=taceas (default)  the 24-mode TaceAs molecular-docking graph used in the paper
GRAPH=er                an Erdos-Renyi graph with an embedded clique, sampled with DGBS_Sampler(arb_graph=True)
    N_MODES (26)  number of nodes / modes
    CLIQUE  (8)   size of the embedded clique (nodes 0..CLIQUE-1; the true max clique is found by the pipeline)
    EDGE_PROB (0.3), GRAPH_SEED (0)
    WEIGHT_MAX (unset)  if set (> 1), the node weights are drawn uniformly in [1, WEIGHT_MAX] (seeded by GRAPH_SEED, the
                        edges are unchanged); the target becomes the maximum-WEIGHT clique, so it need not be the planted one.
Without WEIGHT_MAX the graph is unweighted (all node weights = 1), so the displacement profile comes only from the graph
structure.
"""
import os
import sys
import numpy as np

sys.path.append(r'C:\Users\em1120\DisplacedMolecularDocking')
from Scripts_DGBS.probability_max_clique import generate_adjacency_matrix_with_clique


def graph_from_env():
    """Return dict(arb_graph, n_modes, Adj, weights, tag)."""
    if os.environ.get("GRAPH", "taceas").lower() != "er":
        return dict(arb_graph=False, n_modes=24, Adj=None, weights=None, tag="")
    n = int(os.environ.get("N_MODES", 26))
    k = int(os.environ.get("CLIQUE", 8))
    p = float(os.environ.get("EDGE_PROB", 0.3))
    seed = int(os.environ.get("GRAPH_SEED", 0))
    np.random.seed(seed)
    Adj, _ = generate_adjacency_matrix_with_clique(n, k, p)
    weights, wtag = np.ones(n), ""
    if os.environ.get("WEIGHT_MAX"):
        w_max = float(os.environ["WEIGHT_MAX"])
        weights = np.random.default_rng(seed + 10_000).uniform(1.0, w_max, size=n)  # own generator: edges unchanged
        wtag = f"_w{w_max:g}"
    return dict(arb_graph=True, n_modes=n, Adj=np.asarray(Adj, dtype=float), weights=weights,
                tag=f"er{n}_k{k}_p{p:g}_s{seed}{wtag}")
