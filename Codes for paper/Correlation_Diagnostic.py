"""Exact click-pattern probabilities (no sampling, no local search) to test whether the squeezed-light correlations
help D-GBS find the max clique, compared with controls that have the same photon numbers but no correlations.

For each arm we compute (thewalrus threshold_detection_prob, exact):
  P_clique    probability of the max-clique click pattern (clicks on the clique nodes, none elsewhere)
  enhancement P_clique / mean P(random k-subset pattern of the same size)   (how much the clique is favoured)
  correlation P_clique / prod_i p_i^c_i (1-p_i)^(1-c_i), with p_i the arm's own single-node click probabilities
              (>1: positive correlations between the clique nodes; =1: independent clicks)

Usage:  NSQZ=2.34 NCOH=5 LOSS=0.5 python Correlation_Diagnostic.py
"""
import sys
import os
import numpy as np
import thewalrus as thw
from numpy.linalg import inv
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import imperial_style  # Imperial College fonts (keeps the same environment as the other scripts)
import networkx as nx
from strawberryfields.decompositions import takagi

sys.path.append(r'C:\Users\em1120\DisplacedMolecularDocking')
import Scripts_DGBS.DGBS_ArbitraryGraph_class as dgbs_module
from Scripts_DGBS.DGBS_ArbitraryGraph_class import DGBS_Sampler, PostProcessingSamples
from Scripts_DGBS.utils import make_omega, mean_nsqz
from graph_setup import graph_from_env
import thewalrus.quantum as qt

NSQZ = float(os.environ.get("NSQZ", 2.34))
NCOH = float(os.environ.get("NCOH", 5))
LOSS = float(os.environ.get("LOSS", 0.0))
N_RANDOM = 300
GRAPH = graph_from_env()
hbar, m = 2, GRAPH["n_modes"]
T = 1 - LOSS
rng = np.random.default_rng(1)


def tune_c_consistent(alpha, target_nsqz, Adjtot, weights, nsubpsace):
    Adj = Adjtot[:nsubpsace, :nsubpsace]
    om = make_omega(1.0, alpha, weights)[:nsubpsace, :nsubpsace]
    c_max = 1 / np.sqrt(np.max(takagi(om @ Adj @ om)[0]))

    def gap(c):
        omega = make_omega(c, alpha, weights)[:nsubpsace, :nsubpsace]
        return mean_nsqz(omega @ Adj @ omega) - target_nsqz
    return brentq(gap, 1e-6, 0.9999 * c_max)


dgbs_module.tune_c = tune_c_consistent  # squeezing really tuned to NSQZ
sampler = DGBS_Sampler(tau=1.1, alpha=2.1, target_nsqz=NSQZ, target_ncoh=NCOH, loss_mode=0.0, hbar=2,
                       n_subspace=m, conv="real", save=False, arb_graph=GRAPH["arb_graph"])
sampler.run_sampler(nsamples=1, Adj=GRAPH["Adj"], weights=GRAPH["weights"])
Adj, weights, BIG = sampler.Adj, sampler.weights, sampler.BIG
cov0, mu0 = np.real(sampler.covariance_matrix), np.real(sampler.mean_displacement_vector)
mu0 = mu0 * np.sqrt(NCOH / np.sum((mu0[:m] ** 2 + mu0[m:] ** 2) / (2 * hbar)))   # displacement rescaled to NCOH photons
n_i = (mu0[:m] ** 2 + mu0[m:] ** 2) / (2 * hbar)
N_sqz = sampler.mean_photon_sqz

pp = PostProcessingSamples(np.ones((2, m)), Adj)
pp.weights = weights
clique_nodes = sorted(pp.find_max_clique(networkx_conv=True))
k = len(clique_nodes)
print(f"squeezing {N_sqz:.2f}, displacement {n_i.sum():.2f}, loss {LOSS:g}; max clique {clique_nodes} (size {k})")


def lossy(cov, mu):
    return T * cov + (1 - T) * (hbar / 2) * np.eye(len(cov)), np.sqrt(T) * mu


def reduce_state(cov, mu, modes):
    idx = list(modes) + [i + m for i in modes]
    return cov[np.ix_(idx, idx)], mu[idx]


def prob_pattern(cov, mu, clicked):
    pattern = [1 if i in clicked else 0 for i in range(m)]
    return float(np.real(thw.threshold_detection_prob(mu, cov, pattern, hbar=hbar)))


def single_click_probs(cov, mu):
    p = np.empty(m)
    for i in range(m):
        c1, m1 = reduce_state(cov, mu, [i])
        p[i] = 1 - float(np.real(thw.threshold_detection_prob(m1, c1, [0], hbar=hbar)))
    return p


# --- the arms (same definitions as Controls_Comparison.py)
BIGs = BIG


def cov_from_scale(s):
    Id = np.eye(m)
    return np.real(qt.Covmat(inv(np.block([[Id, -s * BIGs], [-s * BIGs, Id]])), hbar=hbar))


def scale_for_nsqz(target):
    lam_max = np.max(takagi(BIGs)[0])
    return brentq(lambda s: mean_nsqz(s * BIGs) - target, 1e-6, 0.99999 / lam_max)


# coherent light with the same total photon number spread evenly over the modes (graph-blind)
mu_uniform = np.concatenate([np.sqrt(2 * hbar * n_i.sum() / m) * np.ones(m), np.zeros(m)])
arms = {
    "D-GBS": lossy(cov0, mu0),
    "GBS (same squeezing)": lossy(cov0, np.zeros(2 * m)),
    "GBS (matched photon number)": lossy(cov_from_scale(scale_for_nsqz(N_sqz + n_i.sum())), np.zeros(2 * m)),
    "Thermal": lossy(cov0 + hbar * np.diag(np.concatenate([n_i, n_i])), np.zeros(2 * m)),
    "Coherent only (D-GBS profile)": lossy(np.eye(2 * m), mu0),
    "Coherent only (uniform amplitude)": lossy(np.eye(2 * m), mu_uniform),
}

subsets = [set(rng.choice(m, k, replace=False).tolist()) for _ in range(N_RANDOM)]
print(f"{'arm':36s} {'P(clique)':>11s} {'enhancement':>12s} {'correlation':>12s}   mean photons")
for name, (cov, mu) in arms.items():
    p_c = prob_pattern(cov, mu, set(clique_nodes))
    p_rand = np.mean([prob_pattern(cov, mu, s) for s in subsets])
    p_i = single_click_probs(cov, mu)
    ind = np.prod([p_i[i] if i in clique_nodes else 1 - p_i[i] for i in range(m)])
    n_tot = np.sum(((np.diag(cov)[:m] + np.diag(cov)[m:]) / hbar - 1) / 2) + np.sum((mu[:m] ** 2 + mu[m:] ** 2) / (2 * hbar))
    print(f"{name:36s} {p_c:11.3e} {p_c / p_rand:12.2f} {p_c / ind:12.2f}   {n_tot:.2f}")
