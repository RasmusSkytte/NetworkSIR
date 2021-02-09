from datetime import datetime
import numpy as np
# from tqdm import tqdm
# import multiprocessing as mp
# from pathlib import Path
# from importlib import reload
from src.utils import utils
from src.simulation import simulation
# from functools import partial
# import yaml
from contexttimer import Timer


N_tot_max = False
dry_run = False
force_rerun = False


if utils.is_local_computer():

    N_runs = 1

    # Fraction of population to simulate
    f = 0.01

    noise = lambda d : 0
    #noise = lambda d : np.linspace(-d, d, 3)

    verbose = False
    num_cores_max = 2

else :

    N_runs = 1

    # Fraction of population to simulate
    f = 0.1

    noise = lambda d : np.linspace(-d, d, 5)

    verbose = True
    num_cores_max = 20


all_simulation_parameters = [
    {
        "N_tot": int(5_800_000 * f),
        "rho": 0.1,
        #"epsilon_rho": 1,
        "contact_matrices_name": "2021_fase1_sce1",         # The target activity in the society
        "Intervention_contact_matrices_name": "ned2021jan", # Current activity in society
        #
        "burn_in": 20,
        "start_date_offset" : (datetime(2020, 12, 28) - datetime(2020, 12, 28)).days,    # Simulation start date - vaccination start date
        "day_max": 90,
        #
        "beta": 0.0125 + noise(0.005),
        "beta_UK_multiplier": 1.5 + noise(0.2),
        "lambda_I": 4 / 2.52,
        "lambda_E": 4 / 2.5,
        #
        "N_init": (4600 + noise(200)) * f,
        "N_init_UK_frac": 0.03 + noise(0.01),
        #
        "weighted_random_initial_infections": True,
        #
        "do_interventions": True,
        "threshold_interventions_to_apply": [[3]],
        "restriction_thresholds": [[1,5]],
        "continuous_interventions_to_apply":  [[1,2,3,4,5]],
        "intervention_removal_delay_in_clicks": [20],
        "make_restrictions_at_kommune_level": [False],
        "tracking_delay": [10],
    },
]


N_files_total = 0

if __name__ == "__main__":
    with Timer() as t:

        if dry_run:
            print("\n\nRunning a dry run, nothing will actually be simulated.!!!\n\n")

        if force_rerun:
            print("Notice: forced rerun is set to True")

        for d_simulation_parameters in all_simulation_parameters:
            # break
            N_files = simulation.run_simulations(
                d_simulation_parameters,
                N_runs=N_runs,
                num_cores_max=num_cores_max,
                N_tot_max=N_tot_max,
                verbose=verbose,
                force_rerun=force_rerun,
                dry_run=dry_run,
                save_csv=True,
            )

        N_files_total += N_files

    print(f"\n{N_files_total:,} files were generated, total duration {utils.format_time(t.elapsed)}")
    print("Finished simulating!")
