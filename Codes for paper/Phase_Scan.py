"""Exact probability of the max-clique click pattern as a function of the displacement phase (D-GBS state).
theta rotates all displacements by the same phase (alpha -> alpha e^{i theta}) relative to the squeezing.
Also: the theta-average (= global phase randomisation) and a Monte-Carlo average over independent per-mode phases.
Usage: GRAPH=er LOSS=0.1 NSQZ=2.34 NCOH=5 python Phase_Scan.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import thewalrus as thw
from scipy.optimize import brentq
from strawberryfields.decompositions import takagi
import imperial_style  # Imperial College fonts
sys.path.append(r'C:\Users\em1120\DisplacedMolecularDocking')
import Scripts_DGBS.DGBS_ArbitraryGraph_class as dgbs_module
from Scripts_DGBS.DGBS_ArbitraryGraph_class import DGBS_Sampler, PostProcessingSamples
from Scripts_DGBS.utils import make_omega, mean_nsqz
from graph_setup import graph_from_env

NSQZ, NCOH, LOSS = float(os.environ.get("NSQZ", 2.34)), float(os.environ.get("NCOH", 5)), float(os.environ.get("LOSS", 0.0))
G = graph_from_env(); m, hbar, T = G["n_modes"], 2, 1 - LOSS
rng = np.random.default_rng(0)


def tune_c_consistent(alpha, target_nsqz, Adjtot, weights, n):
    Adj = Adjtot[:n, :n]
    om = make_omega(1.0, alpha, weights)[:n, :n]
    cmax = 1 / np.sqrt(np.max(takagi(om @ Adj @ om)[0]))
    return brentq(lambda c: mean_nsqz(make_omega(c, alpha, weights)[:n, :n] @ Adj @ make_omega(c, alpha, weights)[:n, :n]) - target_nsqz, 1e-6, 0.9999 * cmax)


dgbs_module.tune_c = tune_c_consistent
s = DGBS_Sampler(tau=1.1, alpha=2.1, target_nsqz=NSQZ, target_ncoh=NCOH, loss_mode=0.0, hbar=2, n_subspace=m, conv="real",
                 save=False, arb_graph=G["arb_graph"])
s.run_sampler(nsamples=1, Adj=G["Adj"], weights=G["weights"])
cov0, mu0 = np.real(s.covariance_matrix), np.real(s.mean_displacement_vector)
mu0 = mu0 * np.sqrt(NCOH / np.sum((mu0[:m] ** 2 + mu0[m:] ** 2) / (2 * hbar)))
pp = PostProcessingSamples(np.ones((2, m)), s.Adj); pp.weights = s.weights
clique = set(pp.find_max_clique(networkx_conv=True))
pattern = [1 if i in clique else 0 for i in range(m)]
cov = T * cov0 + (1 - T) * (hbar / 2) * np.eye(2 * m)


def P(alpha_complex):
    mu = np.sqrt(T) * np.concatenate([alpha_complex.real, alpha_complex.imag])
    return float(np.real(thw.threshold_detection_prob(mu, cov, pattern, hbar=hbar)))


alpha = mu0[:m] + 1j * mu0[m:]
print("signs of the displacement amplitudes (real axis):", "".join("+" if a.real > 0 else "-" for a in alpha))
thetas = np.linspace(0, 2 * np.pi, 73)
Pt = np.array([P(alpha * np.exp(1j * th)) for th in thetas])
P_default = Pt[0]
P_global_avg = Pt[:-1].mean()
P_permode_avg = np.mean([P(alpha * np.exp(1j * rng.uniform(0, 2 * np.pi, m))) for _ in range(300)])
P_no_squeeze = float(np.real(thw.threshold_detection_prob(np.sqrt(T) * mu0, T * np.eye(2 * m) + (1 - T) * np.eye(2 * m), pattern, hbar=hbar)))
print(f"loss {LOSS:g}, clique {sorted(clique)}")
print(f"P(clique) at the default phase (theta=0)      : {P_default:.3e}")
print(f"  best phase  theta={np.degrees(thetas[Pt.argmax()]):5.1f} deg   : {Pt.max():.3e}   (x{Pt.max() / P_default:.2f} vs default)")
print(f"  worst phase theta={np.degrees(thetas[Pt.argmin()]):5.1f} deg   : {Pt.min():.3e}   (x{Pt.min() / P_default:.2f} vs default)")
print(f"  average over global phase                    : {P_global_avg:.3e}   (x{P_global_avg / P_default:.2f})")
print(f"  average over independent per-mode phases     : {P_permode_avg:.3e}   (x{P_permode_avg / P_default:.2f})")
print(f"  coherent only (same |alpha_i|, no squeezing) : {P_no_squeeze:.3e}   (x{P_no_squeeze / P_default:.3f})")

plt.rcParams.update({'font.size': 6, 'axes.labelsize': 6.5, 'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5, 'legend.fontsize': 5,
                     'axes.linewidth': 0.5, 'pdf.fonttype': 42, 'svg.fonttype': 'path'})
fig, ax = plt.subplots(figsize=(3.4, 2.3), constrained_layout=True)
ax.plot(np.degrees(thetas), Pt, color='#0072B2', lw=0.9, label='D-GBS, global phase $\theta$')
for val, c, lab in [(P_global_avg, '#009E73', 'average over global phase'), (P_permode_avg, '#D55E00', 'average over per-mode phases'),
                    (P_no_squeeze, '#CC79A7', 'coherent only')]:
    ax.axhline(val, color=c, lw=0.7, ls='--', label=lab)
ax.axvline(0, color='k', lw=0.4, ls=':')
ax.set_xlabel('Displacement phase $\theta$ (degrees)'); ax.set_ylabel('P(max-clique pattern)')
ax.set_xlim(0, 360); ax.set_xticks([0, 90, 180, 270, 360]); ax.set_yscale('log'); ax.legend(frameon=False, loc='lower left')
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Plots', f'Phase_Scan_loss{LOSS:g}')
for ext in ('pdf', 'svg', 'png'):
    for suf in [''] + [f'_new{i}' for i in range(2, 20)]:
        try:
            fig.savefig(out + suf + '.' + ext, dpi=600); break
        except PermissionError:
            continue
print("saved", out)
