"""Displacement-vs-squeezing ratio scan at fixed total source photon number.

For each point the mean squeezing photon number is NSQZ = TOTAL - NCOH and the mean displacement photon number is NCOH.
Every point is a full Controls_Graphs.py study (N_GRAPHS Erdos-Renyi graphs, one run per graph), so all the other
parameters are those of Controls_Graphs.py (N_MODES 32, CLIQUE 6, EDGE_PROB 0.1, LOSS 0.2, N_ITER 8, NSAMPLES 500 by default here).
Finished runs are cached, so an interrupted scan can be resumed.

Outputs
  Plots/Controls_Ratio_<tag>.*   a) final success rate  b) paired difference to D-GBS (mean +- 95% CI over graphs)
                                 versus the displacement fraction NCOH / TOTAL
  Data/controls_ratio_<tag>.npz

Settings (environment variables, defaults in brackets): TOTAL (6)  NCOHS ("0.5,1,2,3,4,5")  N_GRAPHS (100)  WORKERS (10)
EDGE_PROB (0.1)  LOSS (0.2)  N_ITER (8)  NSAMPLES (500)  N_MODES (32)  CLIQUE (6)  ONLY_PLOT (0)
"""
import os
import sys
import glob
import subprocess
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imperial_style  # Imperial College fonts

TOTAL = float(os.environ.get("TOTAL", 6))
NCOHS = [float(x) for x in os.environ.get("NCOHS", "0.5,1,2,3,4,5").split(",")]
N_GRAPHS = int(os.environ.get("N_GRAPHS", 100))
defaults = dict(EDGE_PROB="0.1", LOSS="0.2", N_ITER="8", NSAMPLES="500", N_MODES="32", CLIQUE="6", WORKERS="10")
for k, v in defaults.items():
    os.environ.setdefault(k, v)
os.environ["N_GRAPHS"] = str(N_GRAPHS)
here = os.path.dirname(os.path.abspath(__file__))
data_dir, plot_dir = os.path.join(here, "Data"), os.path.join(here, "Plots")
P = {k: os.environ[k] for k in defaults}
W_TAG = f"_w{float(os.environ['WEIGHT_MAX']):g}" if os.environ.get("WEIGHT_MAX") else ""  # node-weighted graphs (graph_setup.py)
tag = (f"er{P['N_MODES']}_k{P['CLIQUE']}_p{float(P['EDGE_PROB']):g}{W_TAG}_loss{float(P['LOSS']):g}_it{P['N_ITER']}"
       f"_total{TOTAL:g}")

ARMS = ["D-GBS", "Coherent only", "Phase-random. (per mode)", "Phase-random. (global)", "GBS", "GBS (matched photon number)"]
rows = {}
for ncoh in NCOHS:
    nsqz = TOTAL - ncoh
    env = dict(os.environ, NCOH=f"{ncoh:g}", NSQZ=f"{nsqz:g}", SKIP_FIGURES="1")
    if os.environ.get("ONLY_PLOT") != "1":
        print(f"=== displacement {ncoh:g} + squeezing {nsqz:g} photons", flush=True)
        subprocess.call([sys.executable, "-u", os.path.join(here, "Controls_Graphs.py")], env=env, cwd=here)
    nsqz_tag = f"_nsqz{nsqz:g}" if nsqz != 2.34 else ""
    pat = (f"controls_graphs_er{P['N_MODES']}_k{P['CLIQUE']}_p{float(P['EDGE_PROB']):g}{W_TAG}_direct_ncoh{ncoh:g}{nsqz_tag}"
           f"_loss{float(P['LOSS']):g}_it{P['N_ITER']}_*graphs.npz")
    files = glob.glob(os.path.join(data_dir, pat))
    if not files:
        print("missing", pat)
        continue
    r = np.load(files[0])
    rows[ncoh] = {k[len("curve_"):]: r[k][:, -1] for k in r.files if k.startswith("curve_")}

x = np.array(sorted(rows)) / TOTAL
ncs = sorted(rows)
n = len(next(iter(rows[ncs[0]].values())))
mean = {a: np.array([rows[c][a].mean() for c in ncs]) for a in ARMS}
sem = {a: np.array([rows[c][a].std(ddof=1) / np.sqrt(n) for c in ncs]) for a in ARMS}
diff = {a: np.array([(rows[c][a] - rows[c]["D-GBS"]).mean() for c in ncs]) for a in ARMS}
ci = {a: np.array([1.96 * (rows[c][a] - rows[c]["D-GBS"]).std(ddof=1) / np.sqrt(n) for c in ncs]) for a in ARMS}
print(f"\n{'displacement fraction':>22s} " + " ".join(f"{a[:14]:>14s}" for a in ARMS))
for i, c in enumerate(ncs):
    print(f"{x[i]:22.2f} " + " ".join(f"{mean[a][i]:14.1f}" for a in ARMS), "  diff to D-GBS:",
          " ".join(f"{diff[a][i]:+.2f}" for a in ARMS[1:]))
np.savez(os.path.join(data_dir, f"controls_ratio_{tag}.npz"), ncoh=np.array(ncs), total=TOTAL,
         **{f"final_{a}": np.array([rows[c][a] for c in ncs]) for a in ARMS})

OI = dict(black='#000000', orange='#E69F00', sky='#56B4E9', green='#009E73', blue='#0072B2',
          verm='#D55E00', pink='#CC79A7', grey='#7F7F7F')
style = {"D-GBS": (OI['green'], '-', 'o'), "Coherent only": (OI['pink'], '-', 's'),
         "Phase-random. (per mode)": (OI['blue'], '-', '^'), "Phase-random. (global)": (OI['sky'], '-', 'v'),
         "GBS": (OI['black'], '--', 'x'), "GBS (matched photon number)": (OI['grey'], '--', 'd')}
plt.rcParams.update({'font.size': 6, 'axes.labelsize': 6.5, 'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5,
                     'legend.fontsize': 5, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
                     'xtick.major.size': 2, 'ytick.major.size': 2, 'pdf.fonttype': 42, 'svg.fonttype': 'path'})
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
for a in ARMS:
    c, ls, mk = style[a]
    axs[0].errorbar(x, mean[a], yerr=1.96 * sem[a], color=c, ls=ls, marker=mk, ms=2.5, lw=0.9, capsize=1.2,
                    elinewidth=0.6, label=a)
    if a != "D-GBS":
        axs[1].errorbar(x, diff[a], yerr=ci[a], color=c, ls=ls, marker=mk, ms=2.5, lw=0.9, capsize=1.2,
                        elinewidth=0.6, label=a)
axs[1].axhline(0, color='k', lw=0.5)
axs[0].set_ylabel('Final success rate (%)')
axs[1].set_ylabel('Final success rate minus D-GBS (percentage points)')
for ax, lab in zip(axs, ['a)', 'b)']):
    ax.set_xlabel(r'Displacement fraction  $\Sigma_i|\alpha_i|^2$ / total photon number')
    ax.grid(True, linewidth=0.3)
    ax.legend(loc='best', frameon=False, handlelength=1.6, labelspacing=0.3)
    ax.text(-0.14, 1.03, lab, transform=ax.transAxes, fontsize=9, fontname='Imperial Sans Display',
            fontweight='semibold', va='bottom')
caption = (f"Success rate of finding the planted clique versus the split of a fixed total source photon number of {TOTAL:g} "
           f"between displacement (fraction on the x axis) and squeezing (remainder). {n} independent Erdős–Rényi graphs per point "
           f"with {P['N_MODES']} nodes (modes), embedded {P['CLIQUE']}-node clique, edge probability p = {float(P['EDGE_PROB']):g}"
           + (f", node weights uniform in [1, {float(os.environ['WEIGHT_MAX']):g}]" if W_TAG else "") + f"; one run per graph, "
           f"{P['NSAMPLES']} samples, uniform loss {float(P['LOSS']) * 100:g}% per mode, {P['N_ITER']} local-search steps (final step shown). "
           "Error bars: 95% confidence interval of the mean (panel b: of the paired difference to D-GBS). "
           "Coherent only: D-GBS displacement profile, no squeezing; phase-random.: D-GBS with random displacement phases "
           "(per mode or one global phase); GBS: same squeezing without displacement; matched photon number: GBS with the same total photon number as D-GBS.")
fig.get_layout_engine().set(rect=(0, 0.2, 1, 0.8))
fig.text(0.005, 0.005, textwrap.fill(caption, 175), ha="left", va="bottom", fontsize=5.2, linespacing=1.25)
out = os.path.join(plot_dir, f"Controls_Ratio_{tag}")
for ext in ('pdf', 'svg', 'png'):
    for suffix in [''] + [f'_new{i}' for i in range(2, 20)]:
        try:
            fig.savefig(out + suffix + '.' + ext, dpi=600)
            break
        except PermissionError:
            continue
print("saved", out)
