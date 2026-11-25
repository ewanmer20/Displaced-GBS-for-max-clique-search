"""Figure 3 composite (a-f) in Imperial Sans. Run fig3_compute.py first for each (clique, method).
Left column: LEFT; right column: RIGHT, each = (n_nodes, clique_size, edge_prob, method)."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import imperial_style  # Imperial College fonts

# (n_nodes, clique_size, edge_prob, method)
LEFT = (24, 4, 0.4, 'hafnian')
RIGHT = (24, 12, 0.4, 'hafnian')

# Physical size for the paper: one journal column (~3.4 in) so it can be included at width=\columnwidth
# without rescaling; font sizes below are the real printed sizes in points.
FIG_W = 3.4  # height follows from the layout below

d = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'svg.fonttype': 'path',  # outlines: Imperial Sans has no lambda glyph
                     'font.size': 5.5, 'axes.labelsize': 6, 'axes.titlesize': 6,
                     'xtick.labelsize': 5, 'ytick.labelsize': 5, 'axes.linewidth': 0.5,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
                     'xtick.major.size': 2, 'ytick.major.size': 2, 'xtick.minor.size': 1, 'ytick.minor.size': 1})

# ---- manual layout in inches (one journal column) ----
L, R, T, B = 0.50, 0.52, 0.14, 0.34      # margins
W = (FIG_W - L - R - 0.74) / 2           # panel width; 0.74 in gap = colorbar + next ylabel/ticks
H1 = W                                   # rows 1-2 square
H3 = 0.78 * W                            # row 3
GAP = 0.34                               # vertical gap (xlabel + ticks)
FIG_H = T + H1 + GAP + H1 + GAP + H3 + B
fig = plt.figure(figsize=(FIG_W, FIG_H))
lefts = [L, L + W + 0.74]
bottoms = [FIG_H - T - H1, FIG_H - T - 2 * H1 - GAP, B]
heights = [H1, H1, H3]
axs = np.array([[fig.add_axes([lefts[c] / FIG_W, bottoms[r] / FIG_H, W / FIG_W, heights[r] / FIG_H])
                 for c in range(2)] for r in range(3)])


def add_colorbar(im, ax, **kw):
    cax = ax.inset_axes([1.05, 0, 0.06, 1])  # does not take space from the panel
    cb = fig.colorbar(im, cax=cax, **kw)
    cb.outline.set_linewidth(0.5); cb.ax.tick_params(width=0.5, length=2)
    return cb


for col, (N, k, p, method) in enumerate([LEFT, RIGHT]):
    Adj = np.load(os.path.join(d, 'Data', f'fixed_adj_{N}_{k}_{p}.npy'))
    r = np.load(os.path.join(d, 'Data', f'fig3_{N}_{k}_{p}_{method}.npz'))
    P, tanh_max, gamma = r['P'], r['tanh_max'], r['gamma']
    improvement = np.max(P / P[0][None, :], axis=0)

    ax = axs[0, col]
    im = ax.imshow(Adj, cmap='Greys', interpolation='none')
    ax.set_xlabel('Node Index', labelpad=1); ax.set_ylabel('Node Index', labelpad=1)
    ax.set_xticks([0, 6, 12, 18]); ax.set_yticks([0, 6, 12, 18])
    cb0 = add_colorbar(im, ax, ticks=[0, 0.5, 1])
    cb0.set_label('Edge Presence', labelpad=2)

    ax = axs[1, col]
    im = ax.imshow(P, origin='lower')
    cs = ax.contour(P, levels=8, colors='white', origin='lower', linewidths=0.8)
    cb = add_colorbar(im, ax)
    sf = ticker.ScalarFormatter(); sf.set_powerlimits((0, 0))  # scientific notation on the colorbar
    cb.formatter = sf; cb.update_ticks()
    cb.ax.yaxis.get_offset_text().set_fontsize(5)
    # contour lines are left unlabelled for readability at column width (values are given by the colorbar)
    ax.set_xticks(np.linspace(0, len(tanh_max) - 1, 3))
    ax.set_xticklabels(np.round(np.linspace(tanh_max[0], tanh_max[-1], 3), 2))
    ax.set_yticks(np.linspace(0, len(gamma) - 1, 3))
    ax.set_yticklabels(np.round(np.linspace(gamma[0], gamma[-1], 3), 2))
    ax.set_xlabel(r'$\lambda_{max}$', labelpad=1); ax.set_ylabel(r'$\gamma$', labelpad=1)

    ax = axs[2, col]
    ax.plot(tanh_max[1:], improvement[1:], marker='o', ms=1.0, lw=0.7, color='#3953a4',
            label='Max Improvement (DGBS/GBS)')
    ax.set_yscale('log')
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.tick_params(axis='both', which='major', labelsize=4, length=1.5, width=0.4, pad=1.5)  # smaller ticks
    ax.tick_params(axis='both', which='minor', length=0.8, width=0.3)
    ax.set_xlabel(r'Squeezing Parameter ($\lambda_{max}$)', labelpad=1); ax.set_ylabel('Max Improvement', labelpad=1)
    ax.legend(fontsize=3.4, borderpad=0.3, handlelength=1.3, loc='upper right')

for ax, lab in zip(axs.ravel(), 'abcdef'):
    ax.text(-(L - 0.03) / W, 1.04, f'{lab})', transform=ax.transAxes, fontsize=8,
            fontname='Imperial Sans Display', fontweight='semibold', va='bottom')

out = os.path.join(d, 'Plots', 'Figure_3_ImperialFont')
for ext, kw in [('.svg', {}), ('.pdf', {}), ('.png', {'dpi': 600})]:
    for suffix in ['', '_new'] + [f'_new{i}' for i in range(2, 20)]:  # skip names locked by another program
        try:
            fig.savefig(out + suffix + ext, **kw)  # no tight bbox: keeps the exact column width
            print('saved', out + suffix + ext)
            break
        except PermissionError:
            continue
