import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm

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


# Define the subset to plot on
subset = None
#subset = {"beta" : 0.0125, "N_init" : 210}

# Number of plots to keep
N = 25

def plot_with_loglikelihood(cfg, abm_files,
                            covid_index, covid_index_sigma, covid_index_offset,
                            fraction, fraction_sigma, fraction_offset,
                            axes):


    # Store the loglikelihood and function handles
    lls = []
    lines = []

    # Loop over repetitions of the cfg
    for filename in abm_files.cfg_to_filenames(cfg) :

        # Load the csv summery file
        df = file_loaders.pandas_load_file(filename)

        # Extract the values
        I = np.array(df["I"]) / 2.7 * (5_800_000 / cfg.network.N_tot)

        # Get the average number of infected per day
        # (There are 10 clicks per day)
        days = int(len(I) / 10)
        for k in range(days) :
            I[k] = np.mean(I[k*10:(k+1)*10])
        I_m = I[:days]

        # Plot the simulation prediction
        lines.extend(axes[0].plot(pd.date_range(start=datetime(2020, 12, 8), periods = len(I_m), freq="D"), I_m, lw = 4, c = "k"))

        if days >= len(covid_index) + covid_index_offset :

            # Get the range corresponding to the tests
            I_model = I_m[covid_index_offset:covid_index_offset+len(covid_index)]

            # Model input is number of infected. We assume 80.000 daily tests in the model
            logK_model = np.log(I_model) - beta * np.log(80_000)

            # Calculate (log) proability for every point
            log_prop = norm.logpdf(logK_model, loc=covid_index, scale=covid_index_sigma)

            # Determine the log likelihood
            lls.append(np.sum(log_prop))

        else :
            lls.append(np.nan)

    return (lls, lines)


rc_params.set_rc_params()

reload(plot)
reload(file_loaders)

# Prepare output file
fig_name = Path(f"Figures/data_analysis_rasmus_HEP.png")
utils.make_sure_folder_exist(fig_name)

# Load the covid index data
df_index = pd.read_feather("Data/covid_index.feather")

# Get the beta value (Here, scaling parameter for the index cases)
beta       = df_index["beta"][0]
beta_simga = df_index["beta_sd"][0]

# Find the index for the starting date
ind = np.where(df_index["date"] == datetime(2021, 1, 1).date())[0][0]

# Only fit to data after this date
logK       = df_index["logI"][ind:]     # Renaming the index I to index K to avoid confusion with I state in SIR model
logK_sigma = df_index["logI_sd"][ind:]

# Determine the covid_index_offset
covid_index_offset = (datetime(2021, 1, 1) - datetime(2020, 12, 8)).days

# Load the ABM simulations
abm_files = file_loaders.ABM_simulations(base_dir="Output/ABM", subset=None, verbose=True)



handles = []
lls     = []

# Prepare figure
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 12))

print("Plotting the individual ABM simulations. Please wait", flush=True)
for cfg in tqdm(
    abm_files.iter_cfgs(),
    total=len(abm_files.cfgs)) :

    # Plot and compute loglikelihoods
    tmp_lls, tmp_handles = plot_with_loglikelihood(cfg, abm_files, logK, logK_sigma, covid_index_offset, 0, 0, 0, axes)

    # Store the plot handles and loglikelihoods
    lls.extend(tmp_lls)
    handles.extend(tmp_handles)


cfgs = [cfg for cfg in abm_files.iter_cfgs()]
cfg = cfgs[np.nanargmax(lls)]
print("--- Best parameters ---")
print(f"loglikelihood : {lls[np.nanargmax(lls)]:.3f}")
print(f"beta : {cfg.beta:.5f}")
print(f"beta_UK_multiplier : {cfg.beta_UK_multiplier:.3f}")
print(f"N_init : {cfg.N_init:.0f}")
print(f"N_init_UK_frac : {cfg.N_init_UK_frac:.3f}")


# Filter out "bad" runs
lls = np.array(lls)
ulls = lls[~np.isnan(lls)] # Only non-nans
ulls = np.unique(ulls)     # Only unique values
ulls = sorted(ulls)[-N:]   # Keep N best
lls_new = np.array(ulls)

for i in reversed(range(len(lls))) :
    if lls[i] in ulls :
        ulls.remove(lls[i])
    else :
        handles.pop(i).remove()

lls = lls_new
# Rescale lls for plotting
lls -= np.min(lls)
lls /= np.max(lls) / 0.8

# Color according to lls
for ll, line in zip(lls, handles) :
    line.set_alpha(1-ll)



t = pd.date_range(start = datetime(2021, 1, 1), periods = len(logK), freq = "D")

m  = np.exp(logK) * (80_000 ** beta)
ub = np.exp(logK + logK_sigma) * (80_000 ** beta)
lb = np.exp(logK - logK_sigma) * (80_000 ** beta)
s  = np.stack((lb.to_numpy(), ub.to_numpy()))

axes[0].errorbar(t, m, yerr=s, fmt='o', lw=2)



months = mdates.MonthLocator()  # every month
months_fmt = mdates.DateFormatter('%b')

axes[1].xaxis.set_major_locator(months)
axes[1].xaxis.set_major_formatter(months_fmt)


plt.ylim(0, 4000)
plt.ylabel('Daglige positive tests')

axes[1].set_xlim([datetime(2021, 1, 1), datetime(2021, 3, 1)])

plt.savefig(fig_name)
plt.show()
