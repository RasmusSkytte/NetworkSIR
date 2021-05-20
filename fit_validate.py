import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
import os

from src.utils import file_loaders

from src.analysis.helpers  import *
from src.analysis.plotters import *

dim = True

# Define the subset to plot on
subsets = [ {'Intervention_contact_matrices_name' : ['fase3/S2_1', 'fase3/S2_2', 'fase3/S2_3', 'fase3/S2_4', 'fase3/S2_5', 'fase3/S2_6', 'fase3/S2_7', 'fase3/S2_8']} ]
#subsets = [ {'Intervention_contact_matrices_name' : ['fase3/S2_5']} ]

for subset in subsets :
    fig_name = Path('Figures/' + subset['Intervention_contact_matrices_name'][-1].replace('/','_') + '.png')

    # Number of plots to keep
    N = 15

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

    fig2, axes2 = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 12))
    axes2 = axes2.flatten()

    fig3, axes3 = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 12))
    axes3 = axes3.flatten()

    fig4, axes4 = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(12, 12))
    axes4 = axes4.flatten()

    if cfg.stratified_labels == 'region' :
        fig5, axes5 = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 12))
        axes5 = axes5.flatten()
    elif cfg.stratified_labels == 'landsdel' :
        fig5, axes5 = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(12, 12))
        axes5 = axes5.flatten()

    fig6, axes6 = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(12, 12))
    axes6 = axes6.flatten()

    fig7, axes7 = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(12, 12))

    print('Plotting the individual ABM simulations. Please wait', flush=True)
    for k, (filename, network_filename) in tqdm(
        enumerate(zip(abm_files.iter_files(), abm_files.iter_network_files())),
        total=len(abm_files.filenames)) :

        # Load
        ( positve_tests,
          f,
          positive_per_age_group,
          positive_by_variant,
          positive_by_region,
          vaccinations_by_age_group,
          daily_tests,
          total_infections) = load_from_file(filename, network_filename, start_date)

        # Plot
        h  =      plot_simulation_cases_and_variant_fraction(positve_tests, f, total_infections, daily_tests, t_day, t_week, axes1, color=plt.cm.tab10(k) )
        h.extend( plot_simulation_cases_and_variant_fraction(positve_tests, f, total_infections, daily_tests, t_day, t_week, axes2, color=plt.cm.tab10(k)) )
        h.extend( plot_simulation_growth_rates(positive_by_variant, t_day, axes3) )
        h.extend( plot_simulation_category(positive_per_age_group, t_day, axes4) )
        h.extend( plot_simulation_category(positive_by_region, t_day, axes5) )
        h.extend( plot_simulation_category(vaccinations_by_age_group, t_day, axes6) )
        h.extend( plot_simulation_category(daily_tests, t_day, [axes7]) )

        plot_handles.append(h)

        if dim :
            # Evaluate loglikelihood
            I = min(len(logK), len(daily_tests))
            ll =  compute_loglikelihood((positve_tests[:I], t_day[:I]), (logK[:I],         logK_sigma[:I], t_index[:I]), transformation_function = lambda x : np.log(x) - beta * np.log(daily_tests[:I].flatten()))
            ll += compute_loglikelihood((f, t_week),          (fraction, fraction_sigma, t_fraction))

            # Store the plot handles and loglikelihoods
            lls.append(ll)


    if dim :
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
            for k, (ll, handles) in enumerate(zip(lls_best, plot_handles)) :
                for line in handles :
                    line.set_alpha(0.05 + 0.95*ll)
                    line.set_color(plt.cm.tab10(k))





     ######     ###     ######  ########  ######       ####       ########     ##      ##   ########
    ##    ##   ## ##   ##    ## ##       ##    ##     ##  ##      ##     ##  ####    ####   ##    ##
    ##        ##   ##  ##       ##       ##            ####       ##     ##    ##      ##       ##
    ##       ##     ##  ######  ######    ######      ####        ########     ##      ##      ##
    ##       #########       ## ##             ##    ##  ## ##    ##     ##    ##      ##     ##
    ##    ## ##     ## ##    ## ##       ##    ##    ##   ##      ##     ##    ##      ##     ##
    ######  ##     ##  ######  ########  ######      ####  ##    ########   ######  ######   ##

    raw_label_map = pd.read_csv('Data/label_map.csv')

    stratification = 'land'
    label_map = {'stratification_idx_to_stratification' : raw_label_map[[stratification + '_idx',  stratification ]].drop_duplicates().set_index(stratification + '_idx')[stratification],
                 'kommune_to_stratification_idx' : raw_label_map[[stratification + '_idx',  'kommune' ]].drop_duplicates().set_index('kommune')[stratification + '_idx']}


    N_stratifications = len(label_map['stratification_idx_to_stratification'])
    t, _, cases, _ = file_loaders.load_label_data('newest', label_map['kommune_to_stratification_idx'], test_reference = cfg.test_reference, beta = cfg.testing_exponent)


    # Plot the covid index on axes 2
    axes1[0].scatter(t, cases.flatten(), color='k', s=10, zorder=100)

    # Plot the WGS B.1.1.7 fraction
    axes1[1].errorbar(t_fraction, fraction, yerr=fraction_sigma, fmt='s', lw=2)


    axes1[0].set_ylim(0, 4000)
    axes1[0].set_ylabel('Daglige positive')

    axes1[1].set_ylim(0, 1)
    axes1[1].set_ylabel('frac. B.1.1.7')

    set_date_xaxis(axes1[1], start_date, end_date)

    fig1.savefig(fig_name)



    # Plot the covid index on axes 2
    ref_tests = file_loaders.load_daily_tests(utils.DotDict({'day_max' : len(logK)-1, 'start_date_offset' : cfg.start_date_offset, 'network':utils.DotDict({'N_tot' : 5_800_000})}))
    m  = np.exp(logK) * (ref_tests ** beta)
    ub = np.exp(logK + logK_sigma) * (ref_tests ** beta) - m
    lb = m - np.exp(logK - logK_sigma) * (ref_tests ** beta)
    s  = np.stack((lb.to_numpy(), ub.to_numpy()))

    axes2[0].errorbar(t_index, m, yerr=s, fmt='o', lw=2)


    # Plot the WGS B.1.1.7 fraction
    axes2[1].errorbar(t_fraction, fraction, yerr=fraction_sigma, fmt='s', lw=2)


    # Get restriction_thresholds from a cfg
    planned_restriction_dates = abm_files.cfgs[0].planned_restriction_dates
    planned_restriction_types = abm_files.cfgs[0].planned_restriction_types

    axes2[0].set_ylim(0, 4000)
    axes2[0].set_ylabel('Daglige positive')


    axes2[1].set_ylim(0, 1)
    axes2[1].set_ylabel('frac. B.1.1.7')


    fig1.canvas.draw()

    ylims = [ax.get_ylim() for ax in axes2]

    # Get the transition dates
    restiction_days = planned_restriction_dates

    for day in restiction_days :
        restiction_date = start_date + datetime.timedelta(days=day)
        for ax, lim in zip(axes2, ylims) :
            ax.plot([restiction_date, restiction_date], lim, '--', color='k', linewidth=2)


    set_date_xaxis(axes2[1], start_date, end_date)



    for ax, lim in zip(axes2, ylims) :
        ax.set_ylim(lim[0], lim[1])


    fig2.savefig(os.path.splitext(fig_name)[0] + '_likelihood.png')







    ########          ########
    ##     ##            ##
    ##     ##            ##
    ########             ##
    ##   ##              ##
    ##    ##             ##
    ##     ## #######    ##

    R_t = np.array([1.0, 1.0, 1.0])
    for i in range(len(axes3)) :

        axes3[i].plot([start_date, end_date], [R_t[i], R_t[i]], 'b--', lw=2)

        axes3[i].set_ylim(0, 2)

        set_date_xaxis(axes3[i], start_date, end_date)

        if i == 0 :
            axes3[i].set_title(f'All variants', fontsize=24, pad=5)
        else :
            axes3[i].set_title(f'Variant {i}', fontsize=24, pad=5)

        axes3[i].tick_params(axis='x', labelsize=24)
        axes3[i].tick_params(axis='y', labelsize=24)

        # Add the dates of restrictions
        for day in restiction_days :
            restiction_date = start_date + datetime.timedelta(days=day)
            for ax, lim in zip(axes1, ylims) :
                ax.plot([restiction_date, restiction_date], lim, '--', color='k', linewidth=2)




    fig3.savefig(os.path.splitext(fig_name)[0] + '_growth_rates.png')







       ###     ######   ########     ######   ########   #######  ##     ## ########   ######
      ## ##   ##    ##  ##          ##    ##  ##     ## ##     ## ##     ## ##     ## ##    ##
     ##   ##  ##        ##          ##        ##     ## ##     ## ##     ## ##     ## ##
    ##     ## ##   #### ######      ##   #### ########  ##     ## ##     ## ########   ######
    ######### ##    ##  ##          ##    ##  ##   ##   ##     ## ##     ## ##              ##
    ##     ## ##    ##  ##          ##    ##  ##    ##  ##     ## ##     ## ##        ##    ##
    ##     ##  ######   ########     ######   ##     ##  #######   #######  ##         ######

    # Load tests per age group
    t, positive_per_age_group = load_infected_per_category(beta, category='AgeGr', test_adjust=False)

    for i in range(len(axes4)) :

        # Delete empty axes
        if i == 8 :
            for ax in axes4[i:] :
                ax.remove()

            # Adjust the last title
            axes4[i-1].set_title(f'{10*(i-1)}+', fontsize=24, pad=5)
            break

        axes4[i].scatter(t, positive_per_age_group[:, i], color='k', s=10, zorder=100)

        axes4[i].set_ylim(0, 300)

        set_date_xaxis(axes4[i], start_date, end_date, interval=2)

        axes4[i].set_title(f'{10*i}-{10*(i+1)-1}', fontsize=24, pad=5)

        axes4[i].tick_params(axis='x', labelsize=24)
        axes4[i].tick_params(axis='y', labelsize=24)




    fig4.savefig(os.path.splitext(fig_name)[0] + '_age_groups.png')







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
    t, _, cases_per_label, _ = file_loaders.load_label_data('newest', label_map['kommune_to_stratification_idx'], test_reference = cfg.test_reference, beta = cfg.testing_exponent)


    for i in range(len(axes5)) :

        # Delete empty axes
        if i == N_stratifications :
            for ax in axes5[i:] :
                ax.remove()
            break

        axes5[i].scatter(t, cases_per_label[:, i], color='k', s=10, zorder=100)

        axes5[i].set_ylim(0, 500)

        if cfg.stratified_labels == 'region' :
            interval = 2
            fontsize = 24
        elif cfg.stratified_labels == 'landsdel' :
            interval = 3
            fontsize = 18

        set_date_xaxis(axes5[i], start_date, end_date, interval=interval)

        axes5[i].set_title(label_map['stratification_idx_to_stratification'][i], fontsize=fontsize, pad=5)

        axes5[i].tick_params(axis='x', labelsize=24)
        axes5[i].tick_params(axis='y', labelsize=24)


    fig5.savefig(os.path.splitext(fig_name)[0] + '_regions.png')






    ##     ##    ###     ######   ######  #### ##    ##    ###    ######## ####  #######  ##    ##  ######
    ##     ##   ## ##   ##    ## ##    ##  ##  ###   ##   ## ##      ##     ##  ##     ## ###   ## ##    ##
    ##     ##  ##   ##  ##       ##        ##  ####  ##  ##   ##     ##     ##  ##     ## ####  ## ##
    ##     ## ##     ## ##       ##        ##  ## ## ## ##     ##    ##     ##  ##     ## ## ## ##  ######
     ##   ##  ######### ##       ##        ##  ##  #### #########    ##     ##  ##     ## ##  ####       ##
      ## ##   ##     ## ##    ## ##    ##  ##  ##   ### ##     ##    ##     ##  ##     ## ##   ### ##    ##
       ###    ##     ##  ######   ######  #### ##    ## ##     ##    ##    ####  #######  ##    ##  ######

    for i in range(len(axes6)) :

        set_date_xaxis(axes6[i], start_date, end_date, interval=2)

        axes6[i].set_title(f'{10*i}-{10*(i+1)-1}', fontsize=24, pad=5)

        axes6[i].tick_params(axis='x', labelsize=24)
        axes6[i].tick_params(axis='y', labelsize=24)

    # Adjust the last title
    axes6[-2].set_title(f'{10*(i-1)}+', fontsize=24, pad=5)
    axes6[-1].set_title('all', fontsize=24, pad=5)


    fig6.savefig(os.path.splitext(fig_name)[0] + '_vaccinations.png')




    ######## ########  ######  ######## #### ##    ##  ######
       ##    ##       ##    ##    ##     ##  ###   ## ##    ##
       ##    ##       ##          ##     ##  ####  ## ##
       ##    ######    ######     ##     ##  ## ## ## ##   ####
       ##    ##             ##    ##     ##  ##  #### ##    ##
       ##    ##       ##    ##    ##     ##  ##   ### ##    ##
       ##    ########  ######     ##    #### ##    ##  ######¨

    t, tests_per_label, _, _ = file_loaders.load_label_data('newest', label_map['kommune_to_stratification_idx'], test_reference = cfg.test_reference, beta = cfg.testing_exponent)

    fig7.savefig(os.path.splitext(fig_name)[0] + '_testing.png')
