"""Compute DGBS max-clique probability grids for Figure 3 (cached to Data/fig3_*.npz).
usage: python fig3_compute.py <n_nodes> <clique_size> <edge_prob> <hafnian|torontonian>"""
import sys, os
sys.path.append(r'C:\Users\em1120\DisplacedMolecularDocking')
import numpy as np
from strawberryfields.decompositions import takagi
from Scripts_DGBS.probability_max_clique import probability_DGBS_subgraph
N, k, p, method = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
d = os.path.dirname(os.path.abspath(__file__))
Adj = np.load(os.path.join(d, 'Data', f'fixed_adj_{N}_{k}_{p}.npy'))
sub = np.zeros(N, dtype=int); sub[:k] = 1
c_array = np.linspace(0.01, 1/np.max(takagi(Adj)[0])*0.95, 200)
gamma_array = np.linspace(0, 1, 200)
tanh_max = np.array([np.max(takagi(c*Adj)[0]) for c in c_array])
P = np.array([[probability_DGBS_subgraph(c, g, Adj, sub, method=method) for c in c_array] for g in gamma_array])
np.savez(os.path.join(d, 'Data', f'fig3_{N}_{k}_{p}_{method}.npz'), P=P, tanh_max=tanh_max, gamma=gamma_array)
print('done', k, method, P.max())
