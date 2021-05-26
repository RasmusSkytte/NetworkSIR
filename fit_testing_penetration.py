import numpy as np

from src.utils import utils
from src.simulation import simulation
from src.analysis.helpers import *

import os

from tqdm import tqdm
from contexttimer import Timer


if utils.is_local_computer():
    f = 0.1
    n_steps = 10
    num_cores_max = 1
    N_runs = 1
else :
    f = 0.1
    n_steps = 10
    num_cores_max = 3
    N_runs = 3


verbose = False

increment = 0.05


# load starting parameters
params, start_date = utils.load_params('cfg/simulation_parameters_local_lockdowns.yaml', f)
params['day_max'] = 60


# Run iterative algorithm
for n in range(n_steps) :

    # Run simulation with the next parameter set
    if __name__ == '__main__':
        with Timer() as t:
            simulation.run_simulations(params, N_runs=N_runs, num_cores_max=num_cores_max, verbose=verbose)


    # Load the latest simulation data
    subset = {'Intervention_contact_matrices_name' : params['Intervention_contact_matrices_name'][0], 'testing_penetration' : params['testing_penetration'][0]}

    if __name__ == '__main__':
        # Load the ABM simulations
        abm_files = file_loaders.ABM_simulations(subset=subset, verbose=True)

        if len(abm_files.cfgs) == 0 :
            raise ValueError('No files found')

        # Define delta
        delta = np.zeros_like(params['testing_penetration'][0])

        for cfg in abm_files.iter_cfgs() :

            for filename, network_filename in zip(abm_files.cfg_to_filenames(cfg), abm_files.cfg_to_filenames(cfg, datatype='networks')) :

                # Load
                _, _, P_age_groups, _, _, _, _, _, _= load_from_file(filename, network_filename, start_date)

                start_date = datetime.datetime(2020, 12, 28) + datetime.timedelta(days=cfg.start_date_offset)
                end_date   = start_date + datetime.timedelta(days=cfg.day_max)

                t_tests, _ = parse_time_ranges(start_date, end_date)


                t, positive_per_age_group = load_infected_per_category(cfg.testing_exponent, category='AgeGr', test_adjust=False)

                positive_per_age_group = positive_per_age_group[t >= start_date, :]
                positive_per_age_group = positive_per_age_group[:cfg['day_max']+1, :]

                # Compute the difference between simulations and data
                delta += np.sum(positive_per_age_group - P_age_groups, axis=0)


    # Dermine which age group needs adjustment
    I = np.argmax(delta)
    print(delta)
    print(I)
    params['testing_penetration'][0][I] -= np.sign(delta[I]) * increment

