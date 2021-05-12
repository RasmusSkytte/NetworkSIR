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

params, start_date = utils.load_params('cfg/analyzers/incidence_lockdowns.yaml', f)


N_files_total = 0
if __name__ == '__main__':
    with Timer() as t:

        N_files_total +=  simulation.run_simulations(params, N_runs=N_runs, num_cores_max=num_cores_max, verbose=verbose)

    print(f'\n{N_files_total:,} files were generated, total duration {utils.format_time(t.elapsed)}')
    print('Finished simulating!')



    # Load the simulations
    subset = {'matrix_labels' : 'land', 'stratified_labels' : 'land'}
    data = file_loaders.ABM_simulations(subset=subset)

    if len(data.filenames) == 0 :
        raise ValueError(f'No files loaded with subset: {subset}')


    # Get a cfg out
    cfg = data.cfgs[0]
    end_date   = start_date + datetime.timedelta(days=cfg.day_max)

    t_day, _ = parse_time_ranges(start_date, end_date)

    # Prepare output file
    fig_name = 'Figures/incidence_lockdowns.png'
    file_loaders.make_sure_folder_exist(fig_name)


    # Prepare figure
    fig = plt.figure(figsize=(12, 12))
    axes = plt.gca()

    handles = []

    print('Plotting the individual ABM simulations. Please wait', flush=True)
    for (filename, network_filename) in tqdm(zip(data.iter_files(), data.iter_network_files()), total=len(data.filenames)) :

        cfg = file_loaders.filename_to_cfg(filename)

        # Load
        total_tests, _, _, _, _, _, _ = load_from_file(filename, network_filename, start_date)

        # Create the plots
        d = np.argmax(np.array(params['incidence_threshold']).flatten() == cfg['incidence_threshold'][0])
        I_tresh = cfg['incidence_threshold'][0][0]

        if 1 in cfg.incidence_interventions_to_apply :
            linestyle = '-'
            label = f'+ lockdown (I = {I_tresh})'
        else :
            linestyle = '--'
            label = f'- lockdown (I = {I_tresh})'

        handles.append(plot_simulation_cases(total_tests, t_day, axes, color=plt.cm.tab10(d), linestyle=linestyle, label=label))


    axes.set_ylim(0, 10_000)
    axes.set_ylabel('Daglige positive')

    set_date_xaxis(axes, start_date, end_date)

    plt.legend()

    plt.savefig(fig_name)
