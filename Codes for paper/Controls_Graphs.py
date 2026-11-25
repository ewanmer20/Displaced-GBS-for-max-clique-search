"""Controls study over many different graphs: N_GRAPHS random Erdos-Renyi graphs (embedded clique), ONE run per graph
(Controls_Comparison.py, NSAMPLES samples per sampler), then average over graphs.

Outputs
  Plots/Controls_Graphs_<tag>.*          mean success rate vs local-search iteration (shaded: +-1 s.d. across graphs)
  Plots/Controls_Graphs_<tag>_paired.*   paired difference to D-GBS in the final success rate (mean +- 95% CI over graphs)
  Data/controls_graphs_<tag>.npz         all per-graph curves

Settings (environment variables, defaults in brackets):
  N_GRAPHS (100)  GRAPH_SEED0 (0)  N_MODES (32)  CLIQUE (6)  EDGE_PROB (0.3)  LOSS (0.1)  NSAMPLES (500)  NCOH (5)
  WORKERS (10)  ONLY_PLOT (0)
Graph g uses graph seed GRAPH_SEED0 + g (so every graph is different) and its own sampling seed. Runs whose result file
already exists are skipped, so an interrupted study can be resumed. The Oh sampler is not included (too slow).
"""
import os
import re
import sys
import subprocess
import textwrap
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imperial_style  # Imperial College fonts

os.environ.setdefault("GRAPH", "er")
from graph_setup import graph_from_env

N_GRAPHS = int(os.environ.get("N_GRAPHS", 100))
GRAPH_SEED0 = int(os.environ.get("GRAPH_SEED0", 0))
N_MODES = int(os.environ.get("N_MODES", 32))
CLIQUE = int(os.environ.get("CLIQUE", 6))
EDGE_PROB = float(os.environ.get("EDGE_PROB", 0.3))
LOSS = float(os.environ.get("LOSS", 0.1))
NSAMPLES = int(os.environ.get("NSAMPLES", 500))
NCOH = float(os.environ.get("NCOH", 5))
NSQZ = float(os.environ.get("NSQZ", 2.34))  # mean squeezing photon number
WEIGHT_MAX = float(os.environ["WEIGHT_MAX"]) if os.environ.get("WEIGHT_MAX") else None  # node weights U[1, WEIGHT_MAX]
W_TAG = f"_w{WEIGHT_MAX:g}" if WEIGHT_MAX else ""
NSQZ_TAG = f"_nsqz{NSQZ:g}" if NSQZ != 2.34 else ""
N_ITER = int(os.environ.get("N_ITER", 7))  # local-search curve points (steps 0 .. N_ITER-1)
IT = f"_it{N_ITER}" if N_ITER != 7 else ""
WORKERS = int(os.environ.get("WORKERS", 10))
# the graph settings must be in the environment for graph_from_env() (also used here to compute the graph densities)
os.environ.update(N_MODES=str(N_MODES), CLIQUE=str(CLIQUE), EDGE_PROB=str(EDGE_PROB))
SAMPLE_SEED0 = int(os.environ.get("SAMPLE_SEED0", 7000))  # sampling seed of graph g is SAMPLE_SEED0 + g (change it to replicate with new samples)
SEED_TAG = f"_rep{SAMPLE_SEED0}" if SAMPLE_SEED0 != 7000 else ""

here = os.path.dirname(os.path.abspath(__file__))
data_dir, plot_dir = os.path.join(here, "Data"), os.path.join(here, "Plots")
tag = f"er{N_MODES}_k{CLIQUE}_p{EDGE_PROB:g}{W_TAG}_direct_ncoh{NCOH:g}{NSQZ_TAG}" + (f"_loss{LOSS:g}" if LOSS > 0 else "") + IT + SEED_TAG
graph_seeds = list(range(GRAPH_SEED0, GRAPH_SEED0 + N_GRAPHS))


def result_file(gs):
    t = f"er{N_MODES}_k{CLIQUE}_p{EDGE_PROB:g}_s{gs}{W_TAG}_direct_ncoh{NCOH:g}{NSQZ_TAG}" + (f"_loss{LOSS:g}" if LOSS > 0 else "")
    return os.path.join(data_dir, f"controls_comparison_{t}_ss{SAMPLE_SEED0 + gs - GRAPH_SEED0}{IT}_{NSAMPLES}.npz")


def log_file(gs):
    return os.path.join(data_dir, f"controls_graphs_run_{tag}_s{gs}.log")


def run(gs):
    if os.path.exists(result_file(gs)):
        return gs, 0
    env = dict(os.environ, GRAPH="er", N_MODES=str(N_MODES), CLIQUE=str(CLIQUE), EDGE_PROB=str(EDGE_PROB),
               GRAPH_SEED=str(gs), SAMPLE_SEED=str(SAMPLE_SEED0 + gs - GRAPH_SEED0), SQUEEZING_MODE="direct",
               NCOH=str(NCOH), NSQZ=str(NSQZ), LOSS=str(LOSS), NSAMPLES=str(NSAMPLES), N_ITER=str(N_ITER), SKIP_OH="1", NO_FIGURE="1", MPLBACKEND="Agg",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    with open(log_file(gs), "w") as log:
        rc = subprocess.call([sys.executable, "-u", os.path.join(here, "Controls_Comparison.py")], env=env,
                             stdout=log, stderr=subprocess.STDOUT, cwd=here)
    return gs, rc


if os.environ.get("ONLY_PLOT") != "1":
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for gs, rc in pool.map(run, graph_seeds):
            print(f"graph {gs}: {'ok' if os.path.exists(result_file(gs)) else f'FAILED (rc={rc})'}", flush=True)

# ---------------------------------------------------------------- aggregate
seeds = [g for g in graph_seeds if os.path.exists(result_file(g))]
runs = [np.load(result_file(g)) for g in seeds]
n = len(runs)
print(f"\n{n} / {N_GRAPHS} graphs available")
names = [k[len("curve_"):] for k in runs[0].files if k.startswith("curve_")]
curves = {k: np.array([r["curve_" + k] for r in runs]) for k in names}       # (graphs, iterations)
final = {k: c[:, -1] for k, c in curves.items()}
mean = {k: c.mean(0) for k, c in curves.items()}
std = {k: c.std(0, ddof=1) for k, c in curves.items()}

# graph properties and the max clique the pipeline used (parsed from each run log)
density, planted_is_max, maxclique_size = [], 0, []
for g in seeds:
    os.environ["GRAPH_SEED"] = str(g)
    A = graph_from_env()["Adj"]
    density.append(A.sum() / (N_MODES * (N_MODES - 1)))
    m = re.search(r"max clique: \[([0-9, ]+)\]", open(log_file(g)).read()) if os.path.exists(log_file(g)) else None
    if m:
        nodes = [int(x) for x in m.group(1).split(",")]
        maxclique_size.append(len(nodes))
        planted_is_max += int(sorted(nodes) == list(range(CLIQUE)))
density = np.array(density)
print(f"edge density of the graphs: {density.mean():.3f} +- {density.std(ddof=1):.3f} (min {density.min():.3f}, max {density.max():.3f})")
if maxclique_size:
    sizes, counts = np.unique(maxclique_size, return_counts=True)
    print(f"maximum clique found: planted clique in {planted_is_max}/{len(maxclique_size)} graphs; sizes {dict(zip(sizes.tolist(), counts.tolist()))}")

ref = "D-GBS"
print(f"\n{'sampler':32s} final success (mean +- s.d. over graphs) | paired difference to {ref}: mean +- 95% CI | graphs better")
paired = {}
for k in sorted(names, key=lambda k: -final[k].mean()):
    d = final[k] - final[ref]
    ci = 1.96 * d.std(ddof=1) / np.sqrt(n)
    paired[k] = (d.mean(), ci, (d > 0).mean())
    print(f"{k:32s} {final[k].mean():5.1f} +- {final[k].std(ddof=1):4.1f} | {d.mean():+5.2f} +- {ci:4.2f} | {100 * (d > 0).mean():3.0f}%")

np.savez(os.path.join(data_dir, f"controls_graphs_{tag}_{n}graphs.npz"), graph_seeds=np.array(seeds), density=density,
         **{f"curve_{k}": v for k, v in curves.items()})

if os.environ.get("SKIP_FIGURES") == "1":  # used by Controls_Ratio.py (it makes one summary figure)
    sys.exit(0)

# ---------------------------------------------------------------- figures
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
N_sqz, N_disp = float(runs[0]["N_sqz"]), float(runs[0]["N_disp"])
setup = (f"Graphs: {n} independent Erdős–Rényi graphs with {N_MODES} nodes (modes), embedded {CLIQUE}-node clique, "
         f"edge probability p = {EDGE_PROB:g} before planting (mean realised edge density {density.mean():.2f})"
         + (f", node weights uniform in [1, {WEIGHT_MAX:g}] (target: maximum-weight clique)" if WEIGHT_MAX else "") + "; one run per graph. "
         f"Each run: {NSAMPLES} samples, uniform loss {LOSS * 100:g}% per mode, mean squeezing photon number {N_sqz:.2f}, "
         f"mean displacement photon number {N_disp:g} (" + r"$\Sigma_i|\alpha_i|^2$" + "). ")


def save(fig, stem):
    out = os.path.join(plot_dir, stem)
    for ext in ('pdf', 'svg', 'png'):
        for suffix in [''] + [f'_new{i}' for i in range(2, 20)]:  # skip names locked by another program
            try:
                fig.savefig(out + suffix + '.' + ext, dpi=600)
                break
            except PermissionError:
                continue
    print("saved", out)


# figure 1: mean curves over graphs
FIG_W = 7.0
fig, axs = plt.subplots(1, 2, figsize=(FIG_W, 3.1), constrained_layout=True, sharey=True)
panels = [("a)", ["D-GBS", "GBS", "GBS (matched photon number)", "Thermal", "Phase-random. (per mode)",
                  "Phase-random. (global)", "Coherent only"]),
          ("b)", ["D-GBS", "Uniform", "Independent nodes"])]
for ax, (lab, group) in zip(axs, panels):
    for k in group:
        if k in mean:
            c, ls, lw = style[k]
            x = np.arange(len(mean[k]))
            ax.plot(x, mean[k], color=c, linestyle=ls, lw=lw, marker='o', ms=1.5, label=k)
            ax.fill_between(x, mean[k] - std[k], mean[k] + std[k], color=c, alpha=0.18, lw=0)
    ax.set_xlabel('Iteration step of local search algorithm')
    ax.grid(True, linewidth=0.3)
    ax.legend(loc='lower right', frameon=False, handlelength=1.6, labelspacing=0.3)
    ax.text(-0.14, 1.03, lab, transform=ax.transAxes, fontsize=9, fontname='Imperial Sans Display',
            fontweight='semibold', va='bottom')
axs[0].set_ylabel('Success rate (%)')
caption = ("Success rate of finding the planted maximum clique versus local-search iteration step: mean over graphs "
           "(shaded: ±1 standard deviation across graphs). " + setup +
           "a) D-GBS (squeezing + displacement); GBS without displacement at the same squeezing, and at the same total photon number; "
           "thermal noise with the displacement photon number; D-GBS with random displacement phases (one global phase or independent per mode); "
           "coherent light only (D-GBS photon profile, no squeezing). "
           "b) Classical samplers matched to the click statistics of D-GBS: uniform and independent nodes (D-GBS single-node click rates).")
fig.get_layout_engine().set(rect=(0, 0.2, 1, 0.8))
fig.text(0.005, 0.005, textwrap.fill(caption, 175), ha="left", va="bottom", fontsize=5.2, linespacing=1.25)
save(fig, f"Controls_Graphs_{tag}_{n}graphs")

# figure 2: paired difference to D-GBS in the final success rate
order = sorted([k for k in names if k != ref], key=lambda k: paired[k][0])
fig, ax = plt.subplots(figsize=(4.6, 2.55), constrained_layout=True)
ypos = np.arange(len(order))
ax.axvline(0, color='k', lw=0.5)
for y, k in zip(ypos, order):
    d, ci, _ = paired[k]
    ax.errorbar(d, y, xerr=ci, fmt='o', ms=3, color=style[k][0], ecolor=style[k][0], elinewidth=0.9, capsize=1.5)
ax.set_yticks(ypos)
ax.set_yticklabels(order)
ax.set_xlabel(f'Final success rate minus {ref} (percentage points)')
ax.grid(True, axis='x', linewidth=0.3)
cap2 = ("Paired difference between each sampler and D-GBS in the final success rate (after the last local-search step), "
        "computed graph by graph and averaged over graphs; error bars: 95% confidence interval of the mean difference "
        "(positive: better than D-GBS). " + setup)
fig.get_layout_engine().set(rect=(0, 0.27, 1, 0.73))
fig.text(0.005, 0.005, textwrap.fill(cap2, 105), ha="left", va="bottom", fontsize=5.2, linespacing=1.25)
save(fig, f"Controls_Graphs_{tag}_{n}graphs_paired")
