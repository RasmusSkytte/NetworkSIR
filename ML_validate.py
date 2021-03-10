import numpy as np
import pandas as pd
import scipy
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from tqdm import tqdm
from pathlib import Path
import os

from src.utils import file_loaders
from src import rc_params

from src.analysis.helpers import *

# Define the subset to plot on
subsets = [ {'Intervention_contact_matrices_name' : ['ned2021jan', '2021_fase1']}]


for subset in subsets :
    fig_name = Path('Figures/' + subset['Intervention_contact_matrices_name'][-1] + '.png')

    # Number of plots to keep
    N = 25

    def plot_simulation(total_tests, f, t_tests, t_f, axes) :

        # Create the plots
        tmp_handles_0 = axes[0].plot(t_tests, total_tests,  lw=4, c='k')[0]
        tmp_handles_1 = axes[1].plot(t_f,     f,            lw=4, c='k')[0]

        return [tmp_handles_0, tmp_handles_1]

    def plot_simulation_category(tests_by_category, t, axes) :

        tmp_handles = []
        # Create the plots
        for i in range(np.size(tests_by_category, 1)) :
            tmp_handle = axes[i].plot(t, tests_by_category[:, i], lw=4, c=plt.cm.tab10(i))[0]
            tmp_handles.append(tmp_handle)

        return tmp_handles

    def plot_simulation_growth_rates(tests_by_variant, t, axes) :

        # Add the total tests also
        tests_by_variant = np.concatenate((np.sum(tests_by_variant, axis=1).reshape(-1, 1), tests_by_variant), axis=1)

        t += datetime.timedelta(days=0.5)
        tmp_handles = []

        for i in range(tests_by_variant.shape[1]) :

            y = tests_by_variant[:, i]

            if np.all(y == 0) :
                continue

            window_size = 7 # days
            t_w = np.arange(window_size)
            R_w = []

            t_max = len(y)-window_size
            if np.any(y == 0) :
                t_max = min(t_max, np.where(y > 0)[0][-1])

            for j in range(t_max) :
                y_w = y[j:(j+window_size)]
                res, _ = scipy.optimize.curve_fit(lambda t, a, r: a * np.exp(r * t), t_w, y_w, p0=(np.max(y_w), 0))
                R_w.append(1 + 4.7 * res[1])

            t_w = t[window_size:(window_size+t_max)]
            tmp_handles.append(axes[i].plot(t_w, R_w, lw=4, c='k')[0])

        return tmp_handles


    rc_params.set_rc_params()

    # Load the ABM simulations
    abm_files = file_loaders.ABM_simulations(base_dir='Output/ABM', subset=subset, verbose=True)

    if len(abm_files.all_filenames) == 0 :
        raise ValueError(f'No files loaded with subset: {subset}')


    # Get a cfg out
    cfg = abm_files.cfgs[0]
    start_date = datetime.datetime(2020, 12, 28) + datetime.timedelta(days=cfg.start_date_offset)
    end_date   = start_date + datetime.timedelta(days=cfg.day_max)

    t_tests, t_f = parse_time_ranges(start_date, end_date)

    logK, logK_sigma, beta, t_index      = load_covid_index()
    fraction, fraction_sigma, t_fraction = load_b117_fraction()

    # Prepare output file
    file_loaders.make_sure_folder_exist(fig_name)

    plot_handles = []
    lls          = []

    # Prepare figure
    fig1, axes1 = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 12))
    axes1 = axes1.flatten()

    fig2, axes2 = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 12))
    axes2 = axes2.flatten()

    fig3, axes3 = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(12, 12))
    axes3 = axes3.flatten()

    fig4, axes4 = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 12))
    axes4 = axes4.flatten()


    print('Plotting the individual ABM simulations. Please wait', flush=True)
    for filename in tqdm(
        abm_files.iter_all_files(),
        total=len(abm_files.all_filenames)) :

        # Load
        total_tests, f, tests_per_age_group, tests_by_variant, tests_by_region = load_from_file(filename)

        # Plot
        h  = plot_simulation(total_tests, f, t_tests, t_f, axes1)
        h2 = plot_simulation_growth_rates(tests_by_variant, t_tests, axes2)
        h3 = plot_simulation_category(tests_per_age_group, t_tests, axes3)
        h4 = plot_simulation_category(tests_by_region, t_tests, axes4)

        h.extend(h2)
        h.extend(h3)
        h.extend(h4)

        # Evaluate
        ll =  compute_loglikelihood((total_tests, t_tests), (logK,         logK_sigma, t_index), transformation_function = lambda x : np.log(x) - beta * np.log(ref_tests))
        ll += compute_loglikelihood((f, t_f),               (fraction, fraction_sigma, t_fraction))

        # Store the plot handles and loglikelihoods
        plot_handles.append(h)
        lls.append(ll)

    lls = np.array(lls)

    # Filter out 'bad' runs
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





     ######     ###     ######  ########  ######       ####       ########     ##      ##   ########
    ##    ##   ## ##   ##    ## ##       ##    ##     ##  ##      ##     ##  ####    ####   ##    ##
    ##        ##   ##  ##       ##       ##            ####       ##     ##    ##      ##       ##
    ##       ##     ##  ######  ######    ######      ####        ########     ##      ##      ##
    ##       #########       ## ##             ##    ##  ## ##    ##     ##    ##      ##     ##
    ##    ## ##     ## ##    ## ##       ##    ##    ##   ##      ##     ##    ##      ##     ##
    ######  ##     ##  ######  ########  ######      ####  ##    ########   ######  ######   ##

    # Plot the covid index
    m  = np.exp(logK) * (ref_tests ** beta)
    ub = np.exp(logK + logK_sigma) * (ref_tests ** beta) - m
    lb = m - np.exp(logK - logK_sigma) * (ref_tests ** beta)
    s  = np.stack((lb.to_numpy(), ub.to_numpy()))

    axes1[0].errorbar(t_index, m, yerr=s, fmt='o', lw=2)



    # Plot the WGS B.1.1.7 fraction
    axes1[1].errorbar(t_fraction, fraction, yerr=fraction_sigma, fmt='s', lw=2)


    # Get restriction_thresholds from a cfg
    restriction_thresholds = abm_files.cfgs[0].restriction_thresholds

    axes1[0].set_ylim(0, 2000)
    axes1[0].set_ylabel('Daglige positive')


    axes1[1].set_ylim(0, 1)
    axes1[1].set_ylabel('frac. B.1.1.7')


    fig1.canvas.draw()

    ylims = [ax.get_ylim() for ax in axes1]

    # Get the transition dates
    restiction_days = restriction_thresholds[1::2]

    for day in restiction_days :
        restiction_date = start_date + datetime.timedelta(days=day)
        for ax, lim in zip(axes1, ylims) :
            ax.plot([restiction_date, restiction_date], lim, '--', color='k', linewidth=2)



    months     = mdates.MonthLocator()
    months_fmt = mdates.DateFormatter('%b')

    axes1[1].xaxis.set_major_locator(months)
    axes1[1].xaxis.set_major_formatter(months_fmt)
    axes1[1].set_xlim([start_date, end_date])


    for ax, lim in zip(axes1, ylims) :
        ax.set_ylim(lim[0], lim[1])


    fig1.savefig(fig_name)







    ########          ########
    ##     ##            ##
    ##     ##            ##
    ########             ##
    ##   ##              ##
    ##    ##             ##
    ##     ## #######    ##

    R_t = np.array([1, 0.75, 0.75*1.55])
    for i in range(len(axes2)) :

        axes2[i].plot([start_date, end_date], [R_t[i], R_t[i]], 'b--', lw=2)

        axes2[i].set_xlim([start_date, end_date])
        axes2[i].set_ylim(0, 2)

        axes2[i].xaxis.set_major_locator(months)
        axes2[i].xaxis.set_major_formatter(months_fmt)

        if i == 0 :
            axes2[i].set_title(f'All variants', fontsize=24, pad=5)
        else :
            axes2[i].set_title(f'Variant {i}', fontsize=24, pad=5)

        axes2[i].tick_params(axis='x', labelsize=24)
        axes2[i].tick_params(axis='y', labelsize=24)


    fig2.savefig(os.path.splitext(fig_name)[0] + '_growth_rates.png')







       ###     ######   ########     ######   ########   #######  ##     ## ########   ######
      ## ##   ##    ##  ##          ##    ##  ##     ## ##     ## ##     ## ##     ## ##    ##
     ##   ##  ##        ##          ##        ##     ## ##     ## ##     ## ##     ## ##
    ##     ## ##   #### ######      ##   #### ########  ##     ## ##     ## ########   ######
    ######### ##    ##  ##          ##    ##  ##   ##   ##     ## ##     ## ##              ##
    ##     ## ##    ##  ##          ##    ##  ##    ##  ##     ## ##     ## ##        ##    ##
    ##     ##  ######   ########     ######   ##     ##  #######   #######  ##         ######

    # Load tests per age group
    t, positive_per_age_group = load_infected_per_category(beta, category='AgeGr')

    for i in range(len(axes3)) :

        # Delete empty axes
        if i == 8 :
            for ax in axes3[i:] :
                ax.remove()
            break

        axes3[i].scatter(t, positive_per_age_group[:, i], color='k', s=10, zorder=100)

        axes3[i].set_xlim([start_date, end_date])
        axes3[i].set_ylim(0, 300)

        axes3[i].xaxis.set_major_locator(months)
        axes3[i].xaxis.set_major_formatter(months_fmt)

        axes3[i].set_title(f'{10*i}-{10*(i+1)-1}', fontsize=24, pad=5)

        axes3[i].tick_params(axis='x', labelsize=24)
        axes3[i].tick_params(axis='y', labelsize=24)

    # Adjust the last title
    #axes3[-1].set_title(f'{10*i}+', fontsize=24, pad=5)


    fig3.savefig(os.path.splitext(fig_name)[0] + '_age_groups.png')







    ########  ########  ######   ####  #######  ##    ##  ######
    ##     ## ##       ##    ##   ##  ##     ## ###   ## ##    ##
    ##     ## ##       ##         ##  ##     ## ####  ## ##
    ########  ######   ##   ####  ##  ##     ## ## ## ##  ######
    ##   ##   ##       ##    ##   ##  ##     ## ##  ####       ##
    ##    ##  ##       ##    ##   ##  ##     ## ##   ### ##    ##
    ##     ## ########  ######   ####  #######  ##    ##  ######

    # Load tests per region
    t, positive_per_region = load_infected_per_category(beta, category='Region')
    positive_per_region = positive_per_region[:, [3, 0, 2, 1, 4]]

    for i in range(len(axes4)) :

        # Delete empty axes
        if i == len(cfg['label_names']) :
            for ax in axes4[i:] :
                ax.remove()
            break

        axes4[i].scatter(t, positive_per_region[:, i], color='k', s=10, zorder=100)

        axes4[i].set_xlim([start_date, end_date])
        axes4[i].set_ylim(0, 500)

        axes4[i].set_title(cfg['label_names'][i], fontsize=24, pad=5)

        axes4[i].xaxis.set_major_locator(months)
        axes4[i].xaxis.set_major_formatter(months_fmt)

        axes4[i].tick_params(axis='x', labelsize=24)
        axes4[i].tick_params(axis='y', labelsize=24)



    fig4.savefig(os.path.splitext(fig_name)[0] + '_regions.png')



