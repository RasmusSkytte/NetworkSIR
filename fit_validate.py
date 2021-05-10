import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
from numpy.core.defchararray import multiply

from tqdm import tqdm
from pathlib import Path
import os

from src.utils import file_loaders

from src.analysis.helpers  import *
from src.analysis.plotters import *

# Define the subset to plot on
subsets = [ {'Intervention_contact_matrices_name' : ['fase3/S2_1', 'fase3/S2_2', 'fase3/S2_3', 'fase3/S2_4', 'fase3/S2_5', 'fase3/S2_6', 'fase3/S2_7', 'fase3/S2_8']} ]
#subsets = [ {'Intervention_contact_matrices_name' : ['fase3/S2_1', 'fase3/S2_2', 'fase3/S2_3', 'fase3/S2_4', 'fase3/S2_5', 'fase3/S2_6']} ]

for subset in subsets :
    fig_name = Path('Figures/' + subset['Intervention_contact_matrices_name'][-1].replace('/','_') + '.png')

    # Number of plots to keep
    N = 25

    # Load the ABM simulations
    abm_files = file_loaders.ABM_simulations(subset=subset, verbose=True)

    if len(abm_files.filenames) == 0 :
        raise ValueError(f'No files loaded with subset: {subset}')


    # Get a cfg out
    cfg = abm_files.cfgs[0]
    start_date = datetime.datetime(2020, 12, 28) + datetime.timedelta(days=cfg.start_date_offset)
    end_date   = start_date + datetime.timedelta(days=cfg.day_max)

    t_day, t_week = parse_time_ranges(start_date, end_date)

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

    if cfg.stratified_labels == 'region' :
        fig4, axes4 = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 12))
        axes4 = axes4.flatten()
    elif cfg.stratified_labels == 'landsdel' :
        fig4, axes4 = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(12, 12))
        axes4 = axes4.flatten()

    fig5, axes5 = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(12, 12))
    axes5 = axes5.flatten()

    fig6, axes6 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(12, 12))
    axes6 = axes6.flatten()

    print('Plotting the individual ABM simulations. Please wait', flush=True)
    for (filename, network_filename) in tqdm(
        zip(abm_files.iter_files(), abm_files.iter_network_files()),
        total=len(abm_files.filenames)) :

        # Load
        ( positve_tests,
          f,
          positive_per_age_group,
          positive_by_variant,
          positive_by_region,
          vaccinations_by_age_group,
          daily_tests) = load_from_file(filename, network_filename, start_date)

        # Plot
        h  = plot_simulation_cases_and_variant_fraction(positve_tests, f, t_day, t_week, axes1)
        h2 = plot_simulation_growth_rates(positive_by_variant, t_day, axes2)
        h3 = plot_simulation_category(positive_per_age_group, t_day, axes3)
        h4 = plot_simulation_category(positive_by_region, t_day, axes4)
        h5 = plot_simulation_category(vaccinations_by_age_group, t_day, axes5)
        h6 = plot_simulation_category(daily_tests, t_day, axes6)

        h.extend(h2)
        h.extend(h3)
        h.extend(h4)
        h.extend(h5)
        h.extend(h6)

        # Evaluate
        ll =  compute_loglikelihood((positve_tests, t_day), (logK,         logK_sigma, t_index), transformation_function = lambda x : np.log(x) - beta * np.log(ref_tests))
        ll += compute_loglikelihood((f, t_week),          (fraction, fraction_sigma, t_fraction))

        # Store the plot handles and loglikelihoods
        plot_handles.append(h)
        lls.append(ll)

    lls = np.array(lls)
    lls[np.isneginf(lls)] = np.nan

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
    planned_restriction_dates = abm_files.cfgs[0].planned_restriction_dates

    axes1[0].set_ylim(0, 4000)
    axes1[0].set_ylabel('Daglige positive')


    axes1[1].set_ylim(0, 1)
    axes1[1].set_ylabel('frac. B.1.1.7')


    fig1.canvas.draw()

    ylims = [ax.get_ylim() for ax in axes1]

    # Get the transition dates
    restiction_days = planned_restriction_dates[1::2]

    for day in restiction_days :
        restiction_date = start_date + datetime.timedelta(days=day)
        for ax, lim in zip(axes1, ylims) :
            ax.plot([restiction_date, restiction_date], lim, '--', color='k', linewidth=2)


    set_date_xaxis(axes1[1], start_date, end_date)



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

    R_t = np.array([1.0, 1.0, 1.0])
    for i in range(len(axes2)) :

        axes2[i].plot([start_date, end_date], [R_t[i], R_t[i]], 'b--', lw=2)

        axes2[i].set_ylim(0, 2)

        set_date_xaxis(axes2[i], start_date, end_date)

        if i == 0 :
            axes2[i].set_title(f'All variants', fontsize=24, pad=5)
        else :
            axes2[i].set_title(f'Variant {i}', fontsize=24, pad=5)

        axes2[i].tick_params(axis='x', labelsize=24)
        axes2[i].tick_params(axis='y', labelsize=24)

        # Add the dates of restrictions
        for day in restiction_days :
            restiction_date = start_date + datetime.timedelta(days=day)
            for ax, lim in zip(axes1, ylims) :
                ax.plot([restiction_date, restiction_date], lim, '--', color='k', linewidth=2)




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

            # Adjust the last title
            axes3[i-1].set_title(f'{10*(i-1)}+', fontsize=24, pad=5)
            break

        axes3[i].scatter(t, positive_per_age_group[:, i], color='k', s=10, zorder=100)

        axes3[i].set_ylim(0, 300)

        set_date_xaxis(axes3[i], start_date, end_date, interval=2)

        axes3[i].set_title(f'{10*i}-{10*(i+1)-1}', fontsize=24, pad=5)

        axes3[i].tick_params(axis='x', labelsize=24)
        axes3[i].tick_params(axis='y', labelsize=24)




    fig3.savefig(os.path.splitext(fig_name)[0] + '_age_groups.png')







    ########  ########  ######   ####  #######  ##    ##  ######
    ##     ## ##       ##    ##   ##  ##     ## ###   ## ##    ##
    ##     ## ##       ##         ##  ##     ## ####  ## ##
    ########  ######   ##   ####  ##  ##     ## ## ## ##  ######
    ##   ##   ##       ##    ##   ##  ##     ## ##  ####       ##
    ##    ##  ##       ##    ##   ##  ##     ## ##   ### ##    ##
    ##     ## ########  ######   ####  #######  ##    ##  ######

    raw_label_map = pd.read_csv('Data/label_map.csv')


    stratification = cfg.stratified_labels
    label_map = {'stratification_idx_to_stratification' : raw_label_map[[stratification + '_idx',  stratification ]].drop_duplicates().set_index(stratification + '_idx')[stratification],
                 'kommune_to_stratification_idx' : raw_label_map[[stratification + '_idx',  'kommune' ]].drop_duplicates().set_index('kommune')[stratification + '_idx']}


    N_stratifications = len(label_map['stratification_idx_to_stratification'])
    t, _, _, cases_adjusted_per_label = file_loaders.load_label_data('newest', label_map['kommune_to_stratification_idx'], test_reference = cfg.test_reference, beta = cfg.testing_exponent)


    for i in range(len(axes4)) :

        # Delete empty axes
        if i == N_stratifications :
            for ax in axes4[i:] :
                ax.remove()
            break

        axes4[i].scatter(t, cases_adjusted_per_label[:, i], color='k', s=10, zorder=100)

        axes4[i].set_ylim(0, 500)

        if cfg.stratified_labels == 'region' :
            interval = 2
            fontsize = 24
        elif cfg.stratified_labels == 'landsdel' :
            interval = 3
            fontsize = 18

        set_date_xaxis(axes4[i], start_date, end_date, interval=interval)

        axes4[i].set_title(label_map['stratification_idx_to_stratification'][i], fontsize=fontsize, pad=5)

        axes4[i].tick_params(axis='x', labelsize=24)
        axes4[i].tick_params(axis='y', labelsize=24)


    fig4.savefig(os.path.splitext(fig_name)[0] + '_regions.png')






    ##     ##    ###     ######   ######  #### ##    ##    ###    ######## ####  #######  ##    ##  ######
    ##     ##   ## ##   ##    ## ##    ##  ##  ###   ##   ## ##      ##     ##  ##     ## ###   ## ##    ##
    ##     ##  ##   ##  ##       ##        ##  ####  ##  ##   ##     ##     ##  ##     ## ####  ## ##
    ##     ## ##     ## ##       ##        ##  ## ## ## ##     ##    ##     ##  ##     ## ## ## ##  ######
     ##   ##  ######### ##       ##        ##  ##  #### #########    ##     ##  ##     ## ##  ####       ##
      ## ##   ##     ## ##    ## ##    ##  ##  ##   ### ##     ##    ##     ##  ##     ## ##   ### ##    ##
       ###    ##     ##  ######   ######  #### ##    ## ##     ##    ##    ####  #######  ##    ##  ######

    for i in range(len(axes5)) :

        set_date_xaxis(axes5[i], start_date, end_date, interval=2)

        axes5[i].set_title(f'{10*i}-{10*(i+1)-1}', fontsize=24, pad=5)

        axes5[i].tick_params(axis='x', labelsize=24)
        axes5[i].tick_params(axis='y', labelsize=24)

    # Adjust the last title
    axes5[-2].set_title(f'{10*(i-1)}+', fontsize=24, pad=5)
    axes5[-1].set_title('all', fontsize=24, pad=5)


    fig5.savefig(os.path.splitext(fig_name)[0] + '_vaccinations.png')




    ######## ########  ######  ######## #### ##    ##  ######
       ##    ##       ##    ##    ##     ##  ###   ## ##    ##
       ##    ##       ##          ##     ##  ####  ## ##
       ##    ######    ######     ##     ##  ## ## ## ##   ####
       ##    ##             ##    ##     ##  ##  #### ##    ##
       ##    ##       ##    ##    ##     ##  ##   ### ##    ##
       ##    ########  ######     ##    #### ##    ##  ######¨

    t, tests_per_label, _, _ = file_loaders.load_label_data('newest', label_map['kommune_to_stratification_idx'], test_reference = cfg.test_reference, beta = cfg.testing_exponent)

    fig6.savefig(os.path.splitext(fig_name)[0] + '_testing.png')
