"""Repeat Controls_Comparison.py many times on the SAME graph (different sampling seeds) and plot the mean
success rate vs local-search iteration with a shaded band (+-1 standard deviation across runs).

Settings (environment variables, defaults in brackets):
  N_RUNS (50)  LOSS (0.1)  NSAMPLES (500)  NCOH (5)  WORKERS (10)  SEED0 (1000)  GRAPH/N_MODES/CLIQUE/EDGE_PROB/GRAPH_SEED
Each run is Controls_Comparison.py with SKIP_OH=1 and NO_FIGURE=1; runs whose result file already exists are skipped,
so an interrupted study can be resumed. Result: Plots/Controls_Repeats_*.{pdf,svg,png} and Data/controls_repeats_*.npz
"""
import os
import sys
import subprocess
import textwrap
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imperial_style  # Imperial College fonts
from graph_setup import graph_from_env

N_RUNS = int(os.environ.get("N_RUNS", 50))
LOSS = float(os.environ.get("LOSS", 0.1))
NSAMPLES = int(os.environ.get("NSAMPLES", 500))
NCOH = float(os.environ.get("NCOH", 5))
WORKERS = int(os.environ.get("WORKERS", 10))
SEED0 = int(os.environ.get("SEED0", 1000))
FIG_W = 7.0

here = os.path.dirname(os.path.abspath(__file__))
data_dir, plot_dir = os.path.join(here, "Data"), os.path.join(here, "Plots")
graph_tag = graph_from_env()["tag"]
base = f"direct_ncoh{NCOH:g}" + (f"_loss{LOSS:g}" if LOSS > 0 else "")
if graph_tag:
    base = graph_tag + "_" + base
result_file = lambda seed: os.path.join(data_dir, f"controls_comparison_{base}_ss{seed}_{NSAMPLES}.npz")


def run(seed):
    if os.path.exists(result_file(seed)):
        return seed, 0
    env = dict(os.environ, SAMPLE_SEED=str(seed), SQUEEZING_MODE="direct", NCOH=str(NCOH), LOSS=str(LOSS),
               NSAMPLES=str(NSAMPLES), SKIP_OH="1", NO_FIGURE="1", MPLBACKEND="Agg",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    with open(os.path.join(here, "Data", f"controls_repeats_run{seed}.log"), "w") as log:
        rc = subprocess.call([sys.executable, "-u", os.path.join(here, "Controls_Comparison.py")], env=env,
                             stdout=log, stderr=subprocess.STDOUT, cwd=here)
    return seed, rc


seeds = list(range(SEED0, SEED0 + N_RUNS))
if os.environ.get("ONLY_PLOT") != "1":
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for seed, rc in pool.map(run, seeds):
            print(f"seed {seed}: {'ok' if os.path.exists(result_file(seed)) else f'FAILED (rc={rc})'}", flush=True)

# ---------------------------------------------------------------- aggregate
runs = [np.load(result_file(s)) for s in seeds if os.path.exists(result_file(s))]
print(f"{len(runs)} / {N_RUNS} runs available")
names = [k[len("curve_"):] for k in runs[0].files if k.startswith("curve_")]
curves = {n: np.array([r["curve_" + n] for r in runs]) for n in names}   # (runs, iterations)
mean = {n: c.mean(0) for n, c in curves.items()}
std = {n: c.std(0, ddof=1) for n, c in curves.items()}
np.savez(os.path.join(data_dir, f"controls_repeats_{base}_{NSAMPLES}x{len(runs)}.npz"),
         **{f"mean_{n}": mean[n] for n in names}, **{f"std_{n}": std[n] for n in names})
print(f"\n{'sampler':32s} final success rate: mean +- std (run-to-run) | s.e.m.")
for n in sorted(names, key=lambda n: -mean[n][-1]):
    print(f"{n:32s} {mean[n][-1]:5.1f} +- {std[n][-1]:3.1f} | {std[n][-1] / np.sqrt(len(runs)):.2f}")

# ---------------------------------------------------------------- figure
OI = dict(black='#000000', orange='#E69F00', sky='#56B4E9', green='#009E73', blue='#0072B2',
          verm='#D55E00', pink='#CC79A7', grey='#7F7F7F')
style = {"D-GBS": (OI['green'], '-', 1.4), "GBS": (OI['black'], '--', 0.9),
         "GBS (matched photon number)": (OI['grey'], '--', 0.9), "Thermal": (OI['verm'], '-', 0.9),
         "Phase-random. (per mode)": (OI['blue'], '-', 0.9), "Phase-random. (global)": (OI['sky'], '-', 0.9),
         "Coherent only": (OI['pink'], '-', 0.9), "Uniform": (OI['verm'], '-', 0.9),
         "Independent nodes": (OI['orange'], '-', 0.9)}
plt.rcParams.update({'font.size': 6, 'axes.labelsize': 6.5, 'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5,
                     'legend.fontsize': 5, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
                     'xtick.major.size': 2, 'ytick.major.size': 2, 'pdf.fonttype': 42, 'svg.fonttype': 'path'})
fig, axs = plt.subplots(1, 2, figsize=(FIG_W, 3.1), constrained_layout=True, sharey=True)
panels = [("a)", ["D-GBS", "GBS", "GBS (matched photon number)", "Thermal", "Phase-random. (per mode)",
                  "Phase-random. (global)", "Coherent only"]),
          ("b)", ["D-GBS", "Uniform", "Independent nodes"])]
for ax, (lab, group) in zip(axs, panels):
    for n in group:
        if n in mean:
            c, ls, lw = style[n]
            x = np.arange(len(mean[n]))
            ax.plot(x, mean[n], color=c, linestyle=ls, lw=lw, marker='o', ms=1.5, label=n)
            ax.fill_between(x, mean[n] - std[n], mean[n] + std[n], color=c, alpha=0.18, lw=0)
    ax.set_xlabel('Iteration step of local search algorithm')
    ax.grid(True, linewidth=0.3)
    ax.legend(loc='lower right', frameon=False, handlelength=1.6, labelspacing=0.3)
    ax.text(-0.14, 1.03, lab, transform=ax.transAxes, fontsize=9, fontname='Imperial Sans Display',
            fontweight='semibold', va='bottom')
axs[0].set_ylabel('Success rate (%)')

# ---- caption with the test parameters (taken from the saved runs / environment, so it matches what was run)
G = graph_from_env()
n_nodes, n_edges = G["n_modes"], int(G["Adj"].sum() // 2)
density = n_edges / (n_nodes * (n_nodes - 1) / 2)
clique_size, edge_prob = int(os.environ.get("CLIQUE", 8)), float(os.environ.get("EDGE_PROB", 0.3))
N_sqz, N_disp = float(runs[0]["N_sqz"]), float(runs[0]["N_disp"])
caption = (
    f"Success rate of finding the planted maximum clique versus local-search iteration step: mean over {len(runs)} independent runs "
    f"(shaded: ±1 standard deviation across runs) on the same graph. Graph: Erdős–Rényi with {n_nodes} nodes (modes), "
    f"embedded {clique_size}-node clique, edge probability p = {edge_prob:g} before planting (realised edge density {density:.2f}). "
    f"Each run: {NSAMPLES} samples, uniform loss {LOSS * 100:g}% per mode, mean squeezing photon number {N_sqz:.2f}, "
    rf"mean displacement photon number {N_disp:g} ($\Sigma_i|\alpha_i|^2$). "
    "a) D-GBS (squeezing + displacement); GBS without displacement at the same squeezing, and at the same total photon number; "
    "thermal noise with the displacement photon number; D-GBS with random displacement phases (one global phase or independent per mode); "
    "coherent light only (D-GBS photon profile, no squeezing). "
    "b) Classical samplers matched to the click statistics of D-GBS: uniform and independent nodes (D-GBS single-node click rates).")
fig.get_layout_engine().set(rect=(0, 0.2, 1, 0.8))
fig.text(0.005, 0.005, textwrap.fill(caption, 175), ha="left", va="bottom", fontsize=5.2, linespacing=1.25)
out = os.path.join(plot_dir, f"Controls_Repeats_{base}_{NSAMPLES}x{len(runs)}")
for ext in ('pdf', 'svg', 'png'):
    for suffix in [''] + [f'_new{i}' for i in range(2, 20)]:  # skip names locked by another program
        try:
            fig.savefig(out + suffix + '.' + ext, dpi=600)
            break
        except PermissionError:
            continue
print("saved", out)
