from contexttimer import Timer

from tqdm import tqdm

import datetime

from src.utils      import utils
from src.simulation import simulation
from src.utils      import file_loaders

from src.analysis.helpers  import *
from src.analysis.plotters import *


if utils.is_local_computer():
    f = 0.1
    n_steps = 1
    num_cores_max = 3
    N_runs = 1
else :
    f = 0.5
    n_steps = 1
    num_cores_max = 15
    N_runs = 3


if num_cores_max == 1 :
    verbose = True
else :
    verbose = False

params, start_date = utils.load_params('cfg/analyzers/vaccinations.yaml', f)



N_files_total = 0
if __name__ == '__main__':
    with Timer() as t:

        N_files_total +=  simulation.run_simulations(params, N_runs=N_runs, num_cores_max=num_cores_max, verbose=verbose)

    print(f'\n{N_files_total:,} files were generated, total duration {utils.format_time(t.elapsed)}')
    print('Finished simulating!')



    # Load the simulations
    subset = {'matrix_labels' : 'land', 'stratified_labels' : 'land', 'incidence_interventions_to_apply' : [0]}
    data = file_loaders.ABM_simulations(subset=subset)

    if len(data.filenames) == 0 :
        raise ValueError(f'No files loaded with subset: {subset}')


    # Get a cfg out
    cfg = data.cfgs[0]
    end_date   = start_date + datetime.timedelta(days=cfg.day_max)

    t_day, _ = parse_time_ranges(start_date, end_date)

    # Prepare output file
    fig_names = ['Figures/vaccination_effect.png', 'Figures/vaccination_breakdown.png']

    for fig_name in fig_names :
        file_loaders.make_sure_folder_exist(fig_name)


    # Prepare figures
    fig1 = plt.figure(figsize=(12, 12))
    axes1 = plt.gca()

    fig2, axes2 = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(12, 12))
    axes2 = axes2.flatten()

    print('Plotting the individual ABM simulations. Please wait', flush=True)
    for (filename, network_filename) in tqdm(zip(data.iter_files(), data.iter_network_files()), total=len(data.filenames)) :

        cfg = file_loaders.filename_to_cfg(filename)

        # Load
        total_tests, _, _, _, _, vaccinations_by_age_group, _ = load_from_file(filename, network_filename, start_date)

        # Create the plots
        if cfg.start_date_offset == 4 :
            color = plt.cm.tab10(0)
        elif cfg.start_date_offset == 187 :
            color = plt.cm.tab10(1)
        else :
            color = plt.cm.tab10(2)



        if 5 in cfg.continuous_interventions_to_apply :
            linestyle = '-'
            label = f'+ V (start = {cfg.start_date_offset})'
        else :
            linestyle = '--'
            label = f'- V (start = {cfg.start_date_offset})'

        plot_simulation_cases(total_tests, t_day, axes1, color=color, linestyle=linestyle, label=label)

        #plot_simulation_category(vaccinations_by_age_group, t_day, axes2)


    axes1.set_ylim(0, 8000)
    axes1.set_ylabel('Daglige positive')

    set_date_xaxis(axes1, start_date, end_date)

    fig1.legend(bbox_to_anchor=(0.95, 0.9), loc='upper left')

    fig1.savefig(fig_names[0])





    # for i in range(len(axes2)) :

    #     set_date_xaxis(axes2[i], start_date, end_date, interval=2)

    #     axes2[i].set_title(f'{10*i}-{10*(i+1)-1}', fontsize=24, pad=5)
    #     axes2[i].set_ylim(0, 1)

    #     axes2[i].tick_params(axis='x', labelsize=24)
    #     axes2[i].tick_params(axis='y', labelsize=24)


    # # Adjust the last title
    # axes2[-2].set_title(f'{10*(i-1)}+', fontsize=24, pad=5)
    # axes2[-1].set_title('all', fontsize=24, pad=5)


    # fig2.savefig(fig_names[1])
