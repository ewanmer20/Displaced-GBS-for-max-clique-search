import sys
import time
# Add the Prakash folder to sys.path
# Add the Script_DGBS directory to sys.path
sys.path.append(r'C:\Users\em1120\DisplacedMolecularDocking')
from datetime import datetime
# Import calc_unitary from scripts
import os
import numpy as np
import matplotlib.pyplot as plt
import imperial_style  # Imperial College fonts
import pandas as pd
from Scripts_DGBS.probability_max_clique import *
current_dir = os.path.dirname(__file__)
plot_dir = os.path.join(current_dir, 'Plots')
os.makedirs(plot_dir, exist_ok=True)
data_dir = os.path.join(current_dir, 'Data')
os.makedirs(data_dir, exist_ok=True)
os.chdir(plot_dir)

Adj=np.array([
  [0, 1, 1, 1, 0, 0],
  [1, 0, 1, 1, 0, 1],
  [1, 1, 0, 1, 0, 0],
  [1, 1, 1, 0, 1, 0],
  [0, 0, 0, 1, 0, 1],
  [0, 1, 0, 0, 1, 0]
])
subgraph_1=np.array([1,1,1,1,0,0])
METHOD = "torontonian"  # "torontonian" (threshold detectors, click pattern) or "hafnian" (photon-number-resolving)
loss=np.linspace(0,1,50)
gamma_val=[0,0.2,1,2]
c=0.08
cutoff=3
results_df = pd.DataFrame(columns=['Loss','c', 'Gamma_value','Mean_photon_sqz','Mean_photon_disp','Probability','Adjacency_Matrix'])
loss_result=[]
start=time.time()
c_max=find_max_c(Adj)
c_trunc=0.5
for j in range(len(gamma_val)):
    for i in range(len(loss)):
        start_i=time.time()
        if METHOD == "torontonian":
            lossy_prob=probability_DGBS_subgraph(c_max*c_trunc, gamma_val[j], Adj, subgraph_1, method="torontonian", loss=loss[i])
        else:
            lossy_prob=probability_array_DGBS(c_max*c_trunc, gamma_val[j], Adj,loss=loss[i], cutoff=cutoff)[tuple(subgraph_1)]
        loss_result.append(lossy_prob)
        serie_df = pd.DataFrame([{
            'Loss': loss[i],
            'c': c_max*c_trunc,
            'Gamma_value': gamma_val[j],
            'Mean_photon_sqz':sqz_photon_number(c_max*c_trunc,Adj),
            'Mean_photon_disp':disp_photon_number(c_max*c_trunc,gamma_val[j],Adj),
            'Probability': lossy_prob,
            'Adjacency_Matrix': np.ravel(Adj)
        }])
        print(f"Loss: {loss[i]:.2f}, Probability: {lossy_prob:.2e}")
        results_df = pd.concat([results_df, serie_df], ignore_index=True)
        end_i=time.time()
        print("Time taken for compute lossy_prob:", end_i-start_i)
    end=time.time()
print("Time taken for compute lossy_prob:", end-start)

now = datetime.now()
formatted_time = now.strftime("%Y%m%d_%H%M%S")
# Save the DataFrame to a CSV file in the Data subdirectory
filename = f"results_{METHOD}_{formatted_time}.csv"

# Plot the probability loss as a function of loss grouped by different values of gamma_value
# Sized for one column of the two-column paper (3.4 in wide); font sizes are the printed sizes in points.
plt.rcParams.update({'font.size': 6, 'axes.labelsize': 6.5, 'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5,
                     'legend.fontsize': 5.5, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
                     'xtick.major.size': 2, 'ytick.major.size': 2, 'pdf.fonttype': 42, 'svg.fonttype': 'path'})
fig, ax = plt.subplots(figsize=(3.4, 2.3), constrained_layout=True)
for gamma in results_df['Gamma_value'].unique():
    subset = results_df[results_df['Gamma_value'] == gamma]
    ax.plot(subset['Loss'], subset['Probability'], marker='o', ms=1.5, lw=0.9, linestyle='-', label=fr'$\gamma = {gamma:.2f}$')

ax.set_xlabel('Loss')
ax.set_ylabel('Probability')
ax.legend(loc='upper left', frameon=False, handlelength=1.5, labelspacing=0.3)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax.yaxis.get_offset_text().set_fontsize(5.5)
for ext in ('pdf', 'svg', 'png'):
    for suffix in [''] + [f'_new{i}' for i in range(2, 20)]:  # skip names locked by another program
        try:
            fig.savefig(os.path.join(plot_dir, f"lossy_GBS_{METHOD}_{formatted_time}{suffix}.{ext}"), dpi=600)
            break
        except PermissionError:
            continue
# plt.show()


    
