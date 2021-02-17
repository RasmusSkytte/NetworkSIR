from os import stat
import numpy as np
import pandas as pd
from datetime import datetime


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

from tqdm import tqdm
from pathlib import Path

from importlib import reload
from src.utils import utils
from src import plot
from src import file_loaders
from src import rc_params

from src.analysis.helpers import *


# Define the subset to plot on
#subset = None
subset = {"contact_matrices_name" : "2021_fase1"}
fig_name = Path("Figures/" + subset["contact_matrices_name"] + ".png")

# Number of plots to keep
N = 89

start_date = datetime(2020, 12, 21)


def plot_simulation(I_tot_scaled, f, start_date, axes) :

    # Create the plots
    tmp_handles_0 = axes[0].plot(pd.date_range(start=start_date, periods = len(I_tot_scaled), freq="D"),     I_tot_scaled, lw = 4, c = "k")[0]
    tmp_handles_2 = axes[1].plot(pd.date_range(start=start_date, periods = len(f),            freq="W-SUN"), f,            lw = 4, c = "k")[0]

    return [tmp_handles_0, tmp_handles_2]

rc_params.set_rc_params()

reload(plot)
reload(file_loaders)

# Prepare output file
utils.make_sure_folder_exist(fig_name)


logK, logK_sigma, covid_index_offset, beta = load_covid_index(start_date)
fraction, fraction_sigma, fraction_offset = load_b117_fraction()



# Load the ABM simulations
abm_files = file_loaders.ABM_simulations(base_dir="Output/ABM", subset=subset, verbose=True)

if len(abm_files.all_filenames) == 0 :
    raise ValueError(f"No files loaded with subset: {subset}")

plot_handles = []
lls     = []

# Prepare figure
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 12))
axes = axes.flatten()

print("Plotting the individual ABM simulations. Please wait", flush=True)
for filename in tqdm(
    abm_files.iter_all_files(),
    total=len(abm_files.all_filenames)) :

    # Load
    I_tot_scaled, f = load_from_file(filename)

    # Plot
    h = plot_simulation(I_tot_scaled, f, start_date, axes)

    # Evaluate
    ll =  0.5 * compute_loglikelihood(I_tot_scaled, (logK,         logK_sigma, covid_index_offset), transformation_function = lambda x : np.log(x) - beta * np.log(80_000))
    ll += 0.5 * compute_loglikelihood(f,            (fraction, fraction_sigma, fraction_offset))

    # Store the plot handles and loglikelihoods
    plot_handles.append(h)
    lls.append(ll)


# Filter out "bad" runs
lls = np.array(lls)
ulls = lls[~np.isnan(lls)] # Only non-nans
ulls = np.unique(ulls)     # Only unique values
ulls = sorted(ulls)[-N:]   # Keep N best
lls_best = np.array(ulls)

for i in reversed(range(len(lls))) :
    if lls[i] in ulls :
        ulls.remove(lls[i])
    else :
        for handle in plot_handles[i] :
            handle.remove()
        plot_handles.pop(i)

# Rescale lls for plotting
lls_best -= np.min(lls_best)
lls_best /= np.max(lls_best) / 0.8

# Color according to lls
for ll, handles in zip(lls_best, plot_handles) :
    for line in handles :
        line.set_alpha(1-ll)


# Plot the covid index
t = pd.date_range(start = datetime(2021, 1, 1), periods = len(logK), freq = "D")

m  = np.exp(logK) * (80_000 ** beta)
ub = np.exp(logK + logK_sigma) * (80_000 ** beta) - m
lb = m - np.exp(logK - logK_sigma) * (80_000 ** beta)
s  = np.stack((lb.to_numpy(), ub.to_numpy()))

axes[0].errorbar(t, m, yerr=s, fmt='o', lw=2)


# Plot the WGS B.1.1.7 fraction
t = pd.date_range(start = datetime(2020, 12, 28), periods = len(fraction), freq = "W-SUN")
axes[1].errorbar(t, fraction, yerr=fraction_sigma, fmt='s', lw=2)





#axes[0].set_ylim(0, 2000)
axes[0].set_ylabel('Daglige positive')


#axes[1].set_ylim(0, 1)
axes[1].set_ylabel('frac. B.1.1.7')



months     = mdates.MonthLocator()
months_fmt = mdates.DateFormatter('%b')

axes[1].xaxis.set_major_locator(months)
axes[1].xaxis.set_major_formatter(months_fmt)
axes[1].set_xlim([datetime(2021, 1, 1), datetime(2021, 3, 1)])

plt.savefig(fig_name)