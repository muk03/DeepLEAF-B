# %%

import eos
import numpy as np
import pandas as pd
from pathlib import Path
import json
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as ticker

# %%

# Set random seed and RNG
np.random.seed(672)
rng = np.random.mtrand.RandomState(672)

# Define save directory
# save_dir = '/home/hep/md1021/msci_project/dstore_tau_csr_cvr'
# os.makedirs(save_dir, exist_ok=True)

# output_path = Path(save_dir)

# Define kinematics
dstarlnu_kinematics = eos.Kinematics(**{
    'q2':            2.0,  'q2_min':            0.02,     'q2_max':           10.5,
    'cos(theta_l)':  0.0,  'cos(theta_l)_min': -1.0,      'cos(theta_l)_max': +1.0,
    'cos(theta_d)':  0.0,  'cos(theta_d)_min': -1.0,      'cos(theta_d)_max': +1.0,
    'phi':           0.3,  'phi_min':           0.0,      'phi_max':           2.0 * np.pi
})

# Define options
options = eos.Options(**{
    'form-factors': 'BSZ2015',  
    'model': 'WET',              
    'l': 'tau'
})

# Initialize parameters for the first set of WCs
params_1 = eos.Parameters()
param_wc_1_1 = params_1['cbtaunutau::Re{cVR}']
param_wc_1_1.set(5)
param_wc_1_2 = params_1['cbtaunutau::Re{cSR}']
param_wc_1_2.set(2.5)

# Initialize the decay PDF object for the first set
dstarlnu_pdf_1 = eos.SignalPDF.make('B->D^*lnu::d^4Gamma', params_1, dstarlnu_kinematics, options)

# Generate the first set of MC samples
dstarlnu_samples_1, _ = dstarlnu_pdf_1.sample_mcmc(N=300000, stride=5, pre_N=10000, preruns=2, rng=rng)

# Create a DataFrame for the first set of samples
col_names = ["q2", "cos_theta_l", "cos_theta_d", "phi"]
df_events_1 = pd.DataFrame(dstarlnu_samples_1, columns=col_names)

# Initialize parameters for the second set of WCs
params_2 = eos.Parameters()
param_wc_2_1 = params_2['cbtaunutau::Re{cVR}']
param_wc_2_1.set(0)
param_wc_2_2 = params_2['cbtaunutau::Re{cSR}']
param_wc_2_2.set(2.5)

# Initialize the decay PDF object for the second set
dstarlnu_pdf_2 = eos.SignalPDF.make('B->D^*lnu::d^4Gamma', params_2, dstarlnu_kinematics, options)

# Generate the second set of MC samples
dstarlnu_samples_2, _ = dstarlnu_pdf_2.sample_mcmc(N=300000, stride=5, pre_N=10000, preruns=2, rng=rng)

# Create a DataFrame for the second set of samples
df_events_2 = pd.DataFrame(dstarlnu_samples_2, columns=col_names)

params_3 = eos.Parameters()
param_wc_3_1 = params_3['cbtaunutau::Re{cVR}']
param_wc_3_1.set(-5.0)  # Set a new value for cSR
param_wc_3_2 = params_3['cbtaunutau::Re{cSR}']
param_wc_3_2.set(2.5)  # Set a new value for cVR

# Initialize the decay PDF object for the third set
dstarlnu_pdf_3 = eos.SignalPDF.make('B->D^*lnu::d^4Gamma', params_3, dstarlnu_kinematics, options)

# Generate the third set of MC samples
dstarlnu_samples_3, _ = dstarlnu_pdf_3.sample_mcmc(N=300000, stride=5, pre_N=10000, preruns=2, rng=rng)

# Create a DataFrame for the third set of samples
df_events_3 = pd.DataFrame(dstarlnu_samples_3, columns=col_names)

# %%

nbins = 10
hist_1, edges_1 = np.histogramdd(df_events_1.values, bins=nbins)

bin_centers_1 = [0.5 * (edges_1[i][1:] + edges_1[i][:-1]) for i in range(4)]
xpos_1, ypos_1, zpos_1, wpos_1 = np.meshgrid(*bin_centers_1, indexing="ij")

df_eos_1 = pd.DataFrame({
    'q2': xpos_1.ravel(),
    'cos_theta_l': ypos_1.ravel(),
    'cos_theta_d': zpos_1.ravel(),
    'phi': wpos_1.ravel(),
    'bin_height': hist_1.ravel()
})

hist_2, edges_2 = np.histogramdd(df_events_2.values, bins=nbins)

bin_centers_2 = [0.5 * (edges_2[i][1:] + edges_2[i][:-1]) for i in range(4)]
xpos_2, ypos_2, zpos_2, wpos_2 = np.meshgrid(*bin_centers_2, indexing="ij")

df_eos_2 = pd.DataFrame({
    'q2': xpos_2.ravel(),
    'cos_theta_l': ypos_2.ravel(),
    'cos_theta_d': zpos_2.ravel(),
    'phi': wpos_2.ravel(),
    'bin_height': hist_2.ravel()
})

hist_3, edges_3 = np.histogramdd(df_events_3.values, bins=nbins)

# Compute the bin centers for the third dataset
bin_centers_3 = [0.5 * (edges_3[i][1:] + edges_3[i][:-1]) for i in range(4)]
xpos_3, ypos_3, zpos_3, wpos_3 = np.meshgrid(*bin_centers_3, indexing="ij")

# Create the DataFrame for the third dataset
df_eos_3 = pd.DataFrame({
    'q2': xpos_3.ravel(),
    'cos_theta_l': ypos_3.ravel(),
    'cos_theta_d': zpos_3.ravel(),
    'phi': wpos_3.ravel(),
    'bin_height': hist_3.ravel()
})

# %%

observables = [col for col in df_eos_1.columns if col != "bin_height"]
handles, labels = [], []

csr_1 = params_1['cbtaunutau::Re{cSR}'].evaluate()
cvr_1 = params_1['cbtaunutau::Re{cVR}'].evaluate()

csr_2 = params_2['cbtaunutau::Re{cSR}'].evaluate()
cvr_2 = params_2['cbtaunutau::Re{cVR}'].evaluate()

csr_3 = params_3['cbtaunutau::Re{cSR}'].evaluate()
cvr_3 = params_3['cbtaunutau::Re{cVR}'].evaluate()

plt.rcParams["font.family"] = "Serif"
plt.rcParams['font.size'] = 30

# Set up subplots
fig, axes = plt.subplots(2, 2, figsize=(20, 20), dpi=300)
axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

lbl = ['$q^2$', '$\cos(\\theta_l)$', '$\cos(\\theta_d)$', '$\phi$']

# Loop through each observable and plot it
# Loop through each observable and plot it
for i, obs in enumerate(observables):
    ax = axes[i]

    # Integrate over the remaining columns for the first dataset
    df_eos_1_integrated = df_eos_1.groupby([obs])["bin_height"].sum().reset_index()
    x_unq_1 = np.unique(df_eos_1_integrated[obs].values)
    height_eos_1 = df_eos_1_integrated["bin_height"].values

    w = np.diff(x_unq_1)[-1]

    # Integrate over the remaining columns for the second dataset
    df_eos_2_integrated = df_eos_2.groupby([obs])["bin_height"].sum().reset_index()
    x_unq_2 = np.unique(df_eos_2_integrated[obs].values)
    height_eos_2 = df_eos_2_integrated["bin_height"].values

    # Integrate over the remaining columns for the third dataset
    df_eos_3_integrated = df_eos_3.groupby([obs])["bin_height"].sum().reset_index()
    x_unq_3 = np.unique(df_eos_3_integrated[obs].values)
    height_eos_3 = df_eos_3_integrated["bin_height"].values

    # Correct the bin edges for the first dataset
    bin_edges_1 = np.append(x_unq_1 - np.diff(x_unq_1)[0] / 2, x_unq_1[-1] + np.diff(x_unq_1)[0] / 2)
    ax.step(
        bin_edges_1, np.append(height_eos_1, height_eos_1[-1]),
        where='post', color='cornflowerblue', linewidth=3, 
        label=f'1) cSR: {csr_1}, cVR: {cvr_1}'
    )

    # ax.bar(bin_edges_1[:-1], height_eos_1, width=np.diff(bin_edges_1), align='edge', 
    #     color='white', edgecolor='blue', hatch='\\\\', linewidth=2, 
    #     label=f'Tau Decay - cSR: {csr_1}, cVR: {cvr_1}')
    
    # ax.vlines(x_unq_1[0] - w/2, 0, height_eos_1[0], color='cornflowerblue', linewidth=3)
    # ax.vlines(x_unq_1[-1] + w/2, 0, height_eos_1[-1], color='cornflowerblue', linewidth=3)

    # Correct the bin edges for the second dataset
    bin_edges_2 = np.append(x_unq_2 - np.diff(x_unq_2)[0] / 2, x_unq_2[-1] + np.diff(x_unq_2)[0] / 2)
    ax.step(bin_edges_2, np.append(height_eos_2, height_eos_2[-1]),
        where='post', color='orange', linewidth=3, 
        label=f'2) cSR: {csr_2}, cVR: {cvr_2}'
    )

    ax.vlines(x_unq_2[0] - w/2, 0, height_eos_2[0], color='gold', linewidth=3)
    ax.vlines(x_unq_2[-1] + w/2, 0, height_eos_2[-1], color='gold', linewidth=3)

    # ax.bar(bin_edges_2[:-1], height_eos_2, width=np.diff(bin_edges_2), align='edge', 
    #     color='none', edgecolor='red', hatch='//', linewidth=2, 
    #     label=f'Tau Decay - cSR: {csr_2}, cVR: {cvr_2}')

    # # Correct the bin edges for the third dataset
    bin_edges_3 = np.append(x_unq_3 - np.diff(x_unq_3)[0] / 2, x_unq_3[-1] + np.diff(x_unq_3)[0] / 2)
    ax.step(
        bin_edges_3, np.append(height_eos_3, height_eos_3[-1]),
        where='post', color='green', linewidth=3, 
        label=f'3) cSR: {csr_3}, cVR: {cvr_3}'
    )
    ax.vlines(x_unq_3[0] - w/2, 0, height_eos_3[0], color='green', linewidth=3)
    ax.vlines(x_unq_3[-1] + w/2, 0, height_eos_3[-1], color='green', linewidth=3)

    # ax.bar(bin_edges_3[:-1], height_eos_3, width=np.diff(bin_edges_3), align='edge', 
    # color='none', edgecolor='gold', hatch='|', linewidth=2, 
    # label=f'Tau Decay - cSR: {csr_3}, cVR: {cvr_3}')

    # Add labels and title for the subplot
    ax.set_xlabel(f'{lbl[i]}', fontsize=35)
    ax.set_ylabel('Integrated Bin Heights', fontsize=35)
    ax.set_title(f'{lbl[i]} Integrated Bin Heights')

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0.2, 1.5))  # Forces scientific notation with x10^4
    ax.yaxis.set_major_formatter(formatter)
    # ax.grid(True)

    ax.set_xlim((np.min([x_unq_1, x_unq_2, x_unq_3]) - w, max(x_unq_1)) + w/2)

    # Add minor ticks
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    # ax.grid(which='both', linestyle='--', linewidth=0.5)
    # ax.grid(which='minor', linestyle=':', linewidth=0.5)

    ax.tick_params(axis='both',which='major',direction='in',right = True, top=True, length=12, width=1.5)
    ax.tick_params(axis='both', which='minor', direction='in', right = True, top=True, length=6,width=1.5)

    if i == 0:
        handles, labels = ax.get_legend_handles_labels()

# Add legend
fig.legend(
    handles=handles[:3],  # Include the first three entries
    labels=labels[:3],    # Include the first three labels
    loc='lower center',   # Position the legend at the bottom center
    fontsize=30,
    ncol=3,               # Arrange legend entries in a single row
    bbox_to_anchor=(0.5, -0.065)  # Adjust position (x=center, y=slightly below the figure)
)

fig.suptitle('Integrated Observable Distributions \n cVR Impact, Tau Decay')
# Adjust layout and show the plot
plt.tight_layout()  # Leave space for the global title
plt.show()

# %%
