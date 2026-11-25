"""Controls requested by the reviewer: is the D-GBS advantage due to coherent displacement, or just to
the extra (classical) photons?

All arms are post-processed with the same pipeline as SuccessRate.py (clean -> shrink -> local search)
and compared through the success rate of finding the max-clique vs the local-search iteration step.

Quantum arms (Gaussian threshold-detection sampling with thewalrus, same graph / squeezing as D-GBS):
  D-GBS                         squeezing + coherent displacement (reference)
  GBS                           same squeezing, no displacement
  GBS (matched photon number)   no displacement, more squeezing so that <n_total> equals D-GBS
  Phase-randomised (per mode)   D-GBS amplitudes |alpha_i|, independent random phase per mode and per shot
  Phase-randomised (global)     D-GBS displacement with one random common phase per shot
  Thermal                       same squeezing + zero-mean thermal noise with <n_i> = |alpha_i|^2
  Coherent only                 no squeezing; independent clicks with p_i = 1 - exp(-|alpha_i|^2)
Classical arms (matched to the click-number distribution of the D-GBS samples; Oh: even click numbers only):
  Uniform, Oh sampler, independent nodes (D-GBS single-node click marginals)

Run:  python Controls_Comparison.py            (settings below)
"""
import sys
import os
import copy
from time import time
import numpy as np
import matplotlib.pyplot as plt
import imperial_style  # Imperial College fonts
import networkx as nx
import thewalrus.samples as sp
from scipy.linalg import inv
from scipy.optimize import brentq
from strawberryfields.apps import sample, clique
import thewalrus.quantum as qt

sys.path.append(r'C:\Users\em1120\DisplacedMolecularDocking')
import Scripts_DGBS.DGBS_ArbitraryGraph_class as dgbs_module
from Scripts_DGBS.DGBS_ArbitraryGraph_class import DGBS_Sampler, PostProcessingSamples
from Scripts_DGBS.OhSampler import get_G_sample
from Scripts_DGBS.utils import mean_nsqz, make_omega, count_clique_occurence_networkx, is_clique_networkx
from graph_setup import graph_from_env

# ---------------------------------------------------------------- settings
SIM_PARAMS = {"tau": 1.1, "alpha": 2.1, "target_nsqz": 2.34, "target_ncoh": 10, "loss_mode": 0.0,
              "hbar": 2, "n_subspace": 24, "conv": "real", "save": False}
GRAPH = graph_from_env()  # TaceAs (default) or an Erdos-Renyi graph with arb_graph=True (see graph_setup.py)
if GRAPH["arb_graph"]:
    SIM_PARAMS.update(arb_graph=True, n_subspace=GRAPH["n_modes"])
NSAMPLES = int(os.environ.get("NSAMPLES", 2000))
# "intended": the squeezing is really tuned to target_nsqz (as stated in the paper).
# "as_coded": the sampler's own tune_c, which tunes with the Laplacian of Adj but then builds the state
#             from Adj, so the squeezing actually obtained is far below target_nsqz (~0.11 instead of 2.34).
# "direct":   squeezing tuned to target_nsqz as in "intended", then the sampler's displacement vector is rescaled
#             (profile kept) so that sum_i |alpha_i|^2 = target_ncoh exactly. This reaches combinations the
#             sampler's own parametrisation cannot, e.g. 2.34 squeezing photons with only 3 displacement photons.
SQUEEZING_MODE = os.environ.get("SQUEEZING_MODE", "intended")
if os.environ.get("NCOH"):
    SIM_PARAMS["target_ncoh"] = float(os.environ["NCOH"])
if os.environ.get("NSQZ"):
    SIM_PARAMS["target_nsqz"] = float(os.environ["NSQZ"])
TAG = SQUEEZING_MODE if SQUEEZING_MODE != "direct" else f"direct_ncoh{SIM_PARAMS['target_ncoh']:g}"
if SIM_PARAMS["target_nsqz"] != 2.34:
    TAG += f"_nsqz{SIM_PARAMS['target_nsqz']:g}"
# Uniform loss applied to every arm after the source (0 = lossless, 0.5 = half of the photons lost per mode)
LOSS = float(os.environ.get("LOSS", 0.0))
T_TRANS = 1.0 - LOSS
if LOSS > 0:
    TAG += f"_loss{LOSS:g}"
if GRAPH["tag"]:
    TAG = GRAPH["tag"] + "_" + TAG
if os.environ.get("SAMPLE_SEED"):
    TAG += f"_ss{os.environ['SAMPLE_SEED']}"
TAG_SAMPLES = TAG  # cache name of the raw samples: does not depend on the number of local-search iterations
NITERATIONS = int(os.environ.get("N_ITER", 7))  # points per success-rate curve (steps 0 .. N_ITER-1)
if NITERATIONS != 7:
    TAG += f"_it{NITERATIONS}"
SEED = int(os.environ.get("SAMPLE_SEED", 1234))  # seed of the sampling (not of the graph)
FIG_W = 7.0  # inches (two-column width); the figure has two panels side by side

current_dir = os.path.dirname(os.path.abspath(__file__))
plot_dir = os.path.join(current_dir, 'Plots')
data_dir = os.path.join(current_dir, 'Data')
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
rng = np.random.default_rng(SEED)
np.random.seed(SEED)  # used by the Oh sampler / strawberryfields helpers
hbar = SIM_PARAMS["hbar"]


def tune_c_consistent(alpha, target_nsqz, Adjtot, weights, nsubpsace):
    """Like utils.tune_c but with the same BIG = omega @ Adj @ omega that run_sampler uses (BIG ~ c^2)."""
    from strawberryfields.decompositions import takagi
    Adj = Adjtot[:nsubpsace, :nsubpsace]
    om = make_omega(1.0, alpha, weights)[:nsubpsace, :nsubpsace]
    c_max = 1 / np.sqrt(np.max(takagi(om @ Adj @ om)[0]))

    def gap(c):
        omega = make_omega(c, alpha, weights)[:nsubpsace, :nsubpsace]
        return mean_nsqz(omega @ Adj @ omega) - target_nsqz
    return brentq(gap, 1e-6, 0.9999 * c_max)


if SQUEEZING_MODE in ("intended", "direct"):
    dgbs_module.tune_c = tune_c_consistent


# ---------------------------------------------------------------- helpers
def lossy(cov, mu):
    """Uniform loss channel (transmission 1 - LOSS) on every mode: cov -> T cov + (1-T) hbar/2 I, mu -> sqrt(T) mu."""
    return T_TRANS * cov + (1 - T_TRANS) * (hbar / 2) * np.eye(len(cov)), np.sqrt(T_TRANS) * mu


def displacement_photons(mu, m):
    """Mean photon number per mode of the displacement vector mu (xxpp ordering)."""
    return (mu[:m] ** 2 + mu[m:] ** 2) / (2 * hbar)


def torontonian_samples(cov, mu, nsamples):
    return np.array(sp.torontonian_sample_state(cov=cov, mu=mu, samples=nsamples, hbar=hbar))


def torontonian_samples_random_phase(cov, mu, nsamples, mode):
    """One shot per call with a fresh random displacement phase ('mode' = 'per_mode' or 'global')."""
    m = len(mu) // 2
    alpha = (mu[:m] + 1j * mu[m:])
    out = []
    for _ in range(nsamples):
        theta = rng.uniform(0, 2 * np.pi, size=m) if mode == 'per_mode' else rng.uniform(0, 2 * np.pi)
        a = alpha * np.exp(1j * theta)
        out.append(sp.torontonian_sample_state(cov=cov, mu=np.concatenate([a.real, a.imag]), samples=1, hbar=hbar)[0])
    return np.array(out)


def cov_from_scale(BIG, s):
    m = BIG.shape[0]
    Id = np.eye(m)
    return np.real(qt.Covmat(inv(np.block([[Id, -s * BIG], [-s * BIG, Id]])), hbar=hbar))


def scale_for_nsqz(BIG, target):
    """Rescale factor s of BIG such that the mean squeezing photon number equals `target`."""
    from strawberryfields.decompositions import takagi
    lam_max = np.max(takagi(BIG)[0])
    return brentq(lambda s: mean_nsqz(s * BIG) - target, 1e-6, 0.99999 / lam_max)


def to_subgraphs(raw_samples, graph, weights, adj):
    """Raw click samples -> cleaned -> networkx node lists (same cleaning as PostProcessingSamples)."""
    pp = PostProcessingSamples(raw_samples, adj)
    pp.weights = weights
    return sample.to_subgraphs(copy.deepcopy(pp.cleaned_samples), graph), pp


def success_curve(subgraphs, graph, max_clique_nx, weights, niterations):
    """Success rate (%) of reaching the max-clique after shrink + local search (as in plot_success_rate_vs_niter)."""
    shrunk = [clique.shrink(s, graph) for s in subgraphs]
    n = len(shrunk)
    rates = [count_clique_occurence_networkx(shrunk, max_clique_nx) / n * 100]
    searched = [clique.search(clique=s, graph=graph, iterations=1) for s in shrunk]
    searched = [s for s in searched if not is_clique_networkx(s, max_clique_nx)]
    rates.append((n - len(searched)) / n * 100)
    for _ in range(1, niterations - 1):
        searched = [clique.search(clique=s, graph=graph, iterations=1, node_select=weights) for s in searched]
        searched = [s for s in searched if not is_clique_networkx(s, max_clique_nx)]
        rates.append((n - len(searched)) / n * 100)
    return np.array(rates)


def cached(name, fn):
    """Run fn() once and cache the raw samples on disk (so an interrupted run can resume); print timing."""
    path = os.path.join(data_dir, f"controls_raw_{TAG_SAMPLES}_{NSAMPLES}_{name.replace(' ', '_').replace('(', '').replace(')', '')}.npy")
    if os.path.exists(path):
        print(f"[cache] {name}", flush=True)
        return np.load(path)
    t_start = time()
    print(f"[start] {name}", flush=True)
    out = np.asarray(fn())
    np.save(path, out)
    print(f"[done ] {name}: {time() - t_start:.0f}s, mean clicks {(out >= 1).sum(axis=1).mean():.2f}", flush=True)
    return out


def click_numbers(raw):
    return (np.asarray(raw) >= 1).sum(axis=1)


# ---------------------------------------------------------------- reference D-GBS state
t0 = time()
sampler = DGBS_Sampler(**SIM_PARAMS)
sampler.run_sampler(nsamples=1, foldername="controls_comparison", Adj=GRAPH["Adj"], weights=GRAPH["weights"])  # builds the covariance matrix / mean vector
Adj, weights, m = sampler.Adj, sampler.weights, sampler.n_subspace
cov, mu = np.real(sampler.covariance_matrix), np.real(sampler.mean_displacement_vector)
if SQUEEZING_MODE == "direct":
    mu = mu * np.sqrt(SIM_PARAMS["target_ncoh"] / displacement_photons(mu, m).sum())
n_i = displacement_photons(mu, m)                      # per-mode photons from the displacement
N_disp, N_sqz = n_i.sum(), sampler.mean_photon_sqz
print(f"[{TAG}] D-GBS: <n_sqz> = {N_sqz:.2f}, <n_disp> = {N_disp:.2f} (sum |alpha_i|^2), total = {N_sqz + N_disp:.2f}")

dgbs_raw = cached("D-GBS", lambda: torontonian_samples(*lossy(cov, mu), NSAMPLES))
graph = nx.Graph(Adj)
pp_ref = PostProcessingSamples(dgbs_raw, Adj)
pp_ref.weights = weights
max_clique_nx = pp_ref.find_max_clique(networkx_conv=True)
print("max clique:", sorted(max_clique_nx), "| samples:", NSAMPLES, f"| setup {time() - t0:.0f}s")

BIG = sampler.BIG
raw = {"D-GBS": dgbs_raw}

# GBS: same squeezing, no displacement
raw["GBS"] = cached("GBS", lambda: torontonian_samples(*lossy(cov, np.zeros(2 * m)), NSAMPLES))

# GBS with more squeezing so that <n_total> matches D-GBS (no displacement)
try:
    s_match = scale_for_nsqz(BIG, N_sqz + N_disp)
    raw["GBS (matched photon number)"] = cached("GBS (matched photon number)", lambda: torontonian_samples(*lossy(cov_from_scale(BIG, s_match), np.zeros(2 * m)), NSAMPLES))
except ValueError as err:
    print("matched-photon GBS not reachable:", err)

# thermal noise added after the interferometer: <n_i> = |alpha_i|^2  ->  cov += hbar * n_i on the x and p blocks
cov_th = cov + hbar * np.diag(np.concatenate([n_i, n_i]))
raw["Thermal"] = cached("Thermal", lambda: torontonian_samples(*lossy(cov_th, np.zeros(2 * m)), NSAMPLES))

# phase-randomised displacement (same squeezing, same |alpha_i|)
raw["Phase-random. (per mode)"] = cached("Phase-random. (per mode)", lambda: torontonian_samples_random_phase(*lossy(cov, mu), NSAMPLES, 'per_mode'))
raw["Phase-random. (global)"] = cached("Phase-random. (global)", lambda: torontonian_samples_random_phase(*lossy(cov, mu), NSAMPLES, 'global'))

# coherent states only: independent clicks with p_i = 1 - exp(-|alpha_i|^2)
p_click = 1 - np.exp(-T_TRANS * n_i)
raw["Coherent only"] = (rng.random((NSAMPLES, m)) < p_click).astype(int)

# ---------------------------------------------------------------- classical samplers matched to D-GBS click numbers
k_dgbs = click_numbers(dgbs_raw)
k_dgbs = k_dgbs[k_dgbs > 0]
marginals = (dgbs_raw >= 1).mean(axis=0)
classical = {}
classical["Uniform"] = [list(rng.choice(m, k, replace=False)) for k in k_dgbs]
classical["Independent nodes"] = [list(rng.choice(m, k, replace=False, p=marginals / marginals.sum())) for k in k_dgbs]
# Oh's sampler is built from a zero-mean pure Gaussian state, which only has EVEN photon numbers; asking it for an
# odd number would make its rejection sampling run forever. Odd click numbers are therefore mapped at random to
# k-1 or k+1 (k=1 -> 2), which preserves the mean click number.
k_oh = np.array([k if k % 2 == 0 else (2 if k == 1 else k + rng.choice([-1, 1])) for k in k_dgbs])
hist = np.bincount(k_oh)
oh = []
if os.environ.get("SKIP_OH") != "1":  # the Oh sampler is exponentially slow for large click numbers
    for k in range(1, len(hist)):
        if hist[k] >= 1:
            oh += get_G_sample(A_in=Adj, N=k, n_samples=int(hist[k]), fix_photon=True).tolist()
    classical["Oh sampler"] = [np.where(np.array(s) >= 1)[0].tolist() for s in oh]

# ---------------------------------------------------------------- success rates
curves, clicks = {}, {}
for name, r in raw.items():
    t1 = time()
    subgraphs, _ = to_subgraphs(r, graph, weights, Adj)
    curves[name] = success_curve(subgraphs, graph, max_clique_nx, weights, NITERATIONS)
    clicks[name] = click_numbers(r)
    print(f"{name:32s} mean clicks {clicks[name].mean():5.2f} | final success {curves[name][-1]:5.1f}% | {time() - t1:.0f}s")
for name, sg in classical.items():
    curves[name] = success_curve(sg, graph, max_clique_nx, weights, NITERATIONS)
    print(f"{name:32s} final success {curves[name][-1]:5.1f}%")

np.savez(os.path.join(data_dir, f'controls_comparison_{TAG}_{NSAMPLES}.npz'), n_i=n_i, N_sqz=N_sqz, N_disp=N_disp,
         **{f"curve_{k}": v for k, v in curves.items()}, **{f"clicks_{k}": v for k, v in clicks.items()})

# the raw-sample caches (one file per sampler) only serve to resume an interrupted run: remove them once the curves are saved
# (set KEEP_RAW=1 to keep them, e.g. to re-run the local search with another N_ITER without resampling)
if os.environ.get("KEEP_RAW") != "1":
    import glob
    for f in glob.glob(os.path.join(glob.escape(data_dir), f"controls_raw_{glob.escape(TAG_SAMPLES)}_{NSAMPLES}_*.npy")):
        os.remove(f)

if os.environ.get("NO_FIGURE") == "1":  # used by Controls_Repeats.py (many runs, one summary figure)
    sys.exit(0)

# ---------------------------------------------------------------- figure (Imperial font, two-column width)
OI = dict(black='#000000', orange='#E69F00', sky='#56B4E9', green='#009E73', yellow='#F0E442',
          blue='#0072B2', verm='#D55E00', pink='#CC79A7', grey='#7F7F7F')
style = {
    "D-GBS": (OI['green'], '-', 1.4), "GBS": (OI['black'], '--', 0.9),
    "GBS (matched photon number)": (OI['grey'], '--', 0.9), "Thermal": (OI['verm'], '-', 0.9),
    "Phase-random. (per mode)": (OI['blue'], '-', 0.9), "Phase-random. (global)": (OI['sky'], '-', 0.9),
    "Coherent only": (OI['pink'], '-', 0.9), "Uniform": (OI['verm'], '-', 0.9),
    "Oh sampler": (OI['blue'], '-', 0.9), "Independent nodes": (OI['orange'], '-', 0.9),
}
plt.rcParams.update({'font.size': 6, 'axes.labelsize': 6.5, 'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5,
                     'legend.fontsize': 5, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
                     'xtick.major.size': 2, 'ytick.major.size': 2, 'pdf.fonttype': 42, 'svg.fonttype': 'path'})
fig, axs = plt.subplots(1, 2, figsize=(FIG_W, 2.4), constrained_layout=True, sharey=True)
panels = [("a)", ["D-GBS", "GBS", "GBS (matched photon number)", "Thermal", "Phase-random. (per mode)",
                  "Phase-random. (global)", "Coherent only"]),
          ("b)", ["D-GBS", "Uniform", "Oh sampler", "Independent nodes"])]
for ax, (lab, names) in zip(axs, panels):
    for name in names:
        if name in curves:
            c, ls, lw = style[name]
            ax.plot(curves[name], color=c, linestyle=ls, lw=lw, marker='o', ms=1.5, label=name)
    ax.set_xlabel('Iteration step of local search algorithm')
    ax.grid(True, linewidth=0.3)
    ax.legend(loc='upper left', frameon=False, handlelength=1.6, labelspacing=0.3)
    ax.text(-0.14, 1.03, lab, transform=ax.transAxes, fontsize=9, fontname='Imperial Sans Display',
            fontweight='semibold', va='bottom')
axs[0].set_ylabel('Success rate (%)')

out = os.path.join(plot_dir, f'Controls_Comparison_{TAG}_{NSAMPLES}')
for ext in ('pdf', 'svg', 'png'):
    for suffix in [''] + [f'_new{i}' for i in range(2, 20)]:  # skip names locked by another program
        try:
            fig.savefig(out + suffix + '.' + ext, dpi=600)
            break
        except PermissionError:
            continue
print("saved", out)
