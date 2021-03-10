import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from tqdm import tqdm
from pathlib import Path

from src.utils import utils
from src.utils import file_loaders
from src import rc_params

from src.analysis.helpers import *


# Define the subset to plot on
#subset = None
#fig_name = Path("Figures/all.png")

subsets = [ {"Intervention_contact_matrices_name" : ["ned2021jan", "2021_fase1"]}]

# subsets = [{"Intervention_contact_matrices_name" : ["ned2021jan", "2021_fase1"]},
#            {"Intervention_contact_matrices_name" : ["ned2021jan", "2021_fase1", "2021_fase2_sce1"]},
#            {"Intervention_contact_matrices_name" : ["ned2021jan", "2021_fase1", "2021_fase2_sce2"]},
#            {"Intervention_contact_matrices_name" : ["ned2021jan", "2021_fase1", "2021_fase2_sce3"]},
#            {"Intervention_contact_matrices_name" : ["ned2021jan", "2021_fase1", "2021_fase2_sce4"]},
#            {"Intervention_contact_matrices_name" : ["ned2021jan", "2021_fase1", "2021_fase2_sce5"]},
#            {"Intervention_contact_matrices_name" : ["ned2021jan", "2021_fase1", "2021_fase2_sce6"]},
#            {"Intervention_contact_matrices_name" : ["ned2021jan", "2021_fase1", "2021_fase2_sce7_marts", "2021_fase2_sce7_april", "2021_fase2_sce7_maj"]},
#            {"Intervention_contact_matrices_name" : ["ned2021jan", "2021_fase1", "2021_fase2_sce8_marts", "2021_fase2_sce8_april", "2021_fase2_sce8_maj"]},
#            {"Intervention_contact_matrices_name" : ["ned2021jan", "2021_fase1", "2021_fase2_sce9"]}]

for subset in subsets :
    fig_name = Path("Figures/" + subset["Intervention_contact_matrices_name"][-1] + ".png")

    # Number of plots to keep
    N = 25

    start_date = datetime.datetime(2020, 12, 21)


    def plot_simulation(I_tot_scaled, f, start_date, axes) :

        # Create the plots
        tmp_handles_0 = axes[0].plot(pd.date_range(start=start_date, periods = len(I_tot_scaled), freq="D"),     I_tot_scaled, lw = 4, c = "k")[0]
        tmp_handles_1 = axes[1].plot(pd.date_range(start=start_date, periods = len(f),            freq="W-SUN"), f,            lw = 4, c = "k")[0]

        return [tmp_handles_0, tmp_handles_1]

    rc_params.set_rc_params()


    # Prepare output file
    file_loaders.make_sure_folder_exist(fig_name)


    logK, logK_sigma, beta, covid_index_offset, t_index   = load_covid_index(start_date.date())
    fraction, fraction_sigma, fraction_offset, t_fraction = load_b117_fraction()


    # Load the ABM simulations
    abm_files = file_loaders.ABM_simulations(base_dir="Output/ABM", subset=subset, verbose=True)

    if len(abm_files.all_filenames) == 0 :
        raise ValueError(f"No files loaded with subset: {subset}")

    plot_handles = []
    lls     = []

    # Prepare figure
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 12))
    axes = axes.flatten()

    print("Plotting the individual ABM simulations. Please wait", flush=True)
    for filename in tqdm(
        abm_files.iter_all_files(),
        total=len(abm_files.all_filenames)) :

        # Load
        I_tot_scaled, f, _ = load_from_file(filename)

        # Plot
        h  = plot_simulation(I_tot_scaled, f, start_date, axes)

        # Evaluate
        ll =  compute_loglikelihood(I_tot_scaled, (logK,         logK_sigma, covid_index_offset), transformation_function = lambda x : np.log(x) - beta * np.log(80_000))
        ll += compute_loglikelihood(f,            (fraction, fraction_sigma, fraction_offset))

        # Store the plot handles and loglikelihoods
        plot_handles.append(h)
        lls.append(ll)

    lls = np.array(lls)

    # Filter out "bad" runs
    ulls = lls[~np.isnan(lls)] # Only non-nans
    ulls = np.unique(ulls)     # Only unique values
    ulls = sorted(ulls)[-N:]   # Keep N best
    lls = lls.tolist()

    if len(ulls) > 1 :

        for i in reversed(range(len(lls))) :
            if lls[i] in ulls :
                ulls.remove(lls[i])
            else :
                for handle in plot_handles[i] :
                    handle.remove()
                plot_handles.pop(i)
                lls.pop(i)

        lls_best = np.array(lls)
        # Rescale lls for plotting
        lls_best -= np.min(lls_best)
        lls_best /= np.max(lls_best)

        # Color according to lls
        for ll, handles in zip(lls_best, plot_handles) :
            for line in handles :
                line.set_alpha(0.05 + 0.95*ll)

    # Plot the covid index
    m  = np.exp(logK) * (80_000 ** beta)
    ub = np.exp(logK + logK_sigma) * (80_000 ** beta) - m
    lb = m - np.exp(logK - logK_sigma) * (80_000 ** beta)
    s  = np.stack((lb.to_numpy(), ub.to_numpy()))

    axes[0].errorbar(t_index, m, yerr=s, fmt='o', lw=2)


    # Plot the WGS B.1.1.7 fraction
    axes[1].errorbar(t_fraction, fraction, yerr=fraction_sigma, fmt='s', lw=2)


    # Get restriction_thresholds from a cfg
    restriction_thresholds = abm_files.cfgs[0].restriction_thresholds

    axes[0].set_ylim(0, 10000)
    axes[0].set_ylabel('Daglige positive')


    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('frac. B.1.1.7')


    fig.canvas.draw()

    ylims = [ax.get_ylim() for ax in axes]

    # Get the transition dates
    restiction_days = restriction_thresholds[1::2]

    for day in restiction_days :
        restiction_date = start_date + datetime.timedelta(days=day)
        for ax, lim in zip(axes, ylims) :
            ax.plot([restiction_date, restiction_date], lim, '--', color="k", linewidth=2)



    months     = mdates.MonthLocator()
    months_fmt = mdates.DateFormatter('%b')

    axes[1].xaxis.set_major_locator(months)
    axes[1].xaxis.set_major_formatter(months_fmt)
    axes[1].set_xlim([datetime.datetime(2021, 1, 1), datetime.datetime(2021, 5, 15)])


    for ax, lim in zip(axes, ylims) :
        ax.set_ylim(lim[0], lim[1])


    fig.savefig(fig_name)