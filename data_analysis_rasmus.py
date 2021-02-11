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
subset = {"beta" : 0.0125, "N_init" : 210}

# Number of plots to keep
N = 625

def plot_with_loglikelihood(cfg, abm_files, covid_index, covid_index_sigma, covid_index_offset, axes):

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
        lines.extend(axes.plot(pd.date_range(start=datetime(2020, 12, 8), periods = len(I_m), freq="D"), I_m, lw = 4, c = "k"))

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

#%%

reload(plot)
reload(file_loaders)

# Prepare output file
pdf_name = Path(f"Figures/data_analysis_rasmus_HEP.pdf")
utils.make_sure_folder_exist(pdf_name)

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
fig = plt.figure()
axes = plt.gca()

for cfg in tqdm(
    abm_files.iter_cfgs(),
    desc="Plotting the individual ABM simulations",
    total=len(abm_files.cfgs)) :

    # Plot and compute loglikelihoods
    tmp_lls, tmp_handles = plot_with_loglikelihood(cfg, abm_files, logK, logK_sigma, covid_index_offset, axes)

    # Store the plot handles and loglikelihoods
    lls.extend(tmp_lls)
    handles.extend(tmp_handles)

# Rescale lls for plotting
lls -= np.nanmin(lls)
lls /= np.nanmax(lls) / 0.8
lls += 0.2

# Color according to lls
for ll, line in zip(lls, handles) :
    line.set_alpha(ll)

# Keep only the N best lines
for ind in sorted(lls.argsort()[:-N], reverse = True) :
    handles.pop(ind).remove()


months = mdates.MonthLocator()  # every month
months_fmt = mdates.DateFormatter('%b')

axes.xaxis.set_major_locator(months)
axes.xaxis.set_major_formatter(months_fmt)

plt.ylim(0, 5000)

print(lls)
cfgs = [cfg for cfg in abm_files.iter_cfgs()]
cfg = cfgs[np.nanargmax(lls)]
print(cfg.beta)
print(cfg.beta_UK_multiplier)
print(cfg.N_init)

#plt.show()
plt.savefig(pdf_name, dpi=100)
plt.close("all")