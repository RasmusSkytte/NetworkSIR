import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
from importlib import reload
from src.utils import utils
from src.simulation import simulation
from functools import partial
import yaml
from contexttimer import Timer


N_tot_max = False

num_cores_max = 4
N_runs = 1

dry_run = False
force_rerun = True
verbose = True

#%%


if utils.is_local_computer():

    all_simulation_parameters = [
        {
             "N_tot": 580_000*2,
            #"N_tot": 6_000,
            # "make_random_initial_infections": True,
            # "weighted_random_initial_infections": True,
            # "test_delay_in_clicks": [0, 0, 25],
            #"results_delay_in_clicks": [20, 20, 20],
            #"tracking_delay": [0],
             "weighted_random_initial_infections": True,
             "lambda_I": 0.5,
            # "masking_rate_reduction": [[0.0, 0.5, 0.6]] ,
             "threshold_info": [[[1,2], [50, 30], [10, 10]],
                                [[1,2], [100, 100], [10, 10]],
                                [[1,2], [200, 200], [10, 10]],
                                [[1,2], [200, 50], [10, 10]],
                                [[1,2], [200, 100], [10, 10]],
                                [[1,2], [400, 50], [10, 10]],
                                [[1,2], [400, 200], [10, 10]],
                                [[1,2], [400, 400], [10, 10]],
                                [[1,2], [50, 30], [20, 20]],
                                [[1,2], [100, 100], [20, 20]],
                                [[1,2], [200, 200], [20, 20]],
                                [[1,2], [200, 50], [20, 20]],
                                [[1,2], [200, 100], [20, 20]],
                                [[1,2], [400, 50], [20, 20]],
                                [[1,2], [400, 200], [20, 20]],
                                [[1,2], [400, 400], [20, 20]],
                                [[1,2], [50, 30], [30, 30]],
                                [[1,2], [100, 100], [30, 30]],
                                [[1,2], [200, 200], [30, 30]],
                                [[1,2], [200, 50], [30, 30]],
                                [[1,2], [200, 100], [30, 30]],
                                [[1,2], [400, 50], [30, 30]],
                                [[1,2], [400, 200], [30, 30]],
                                [[1,2], [400, 400], [30, 30]],
                                [[1,2], [100, 100], [50, 50]],
                                [[1,2], [200, 200], [50, 50]],
                                [[1,2], [200, 50], [50, 50]],
                                [[1,2], [200, 100], [50, 50]],
                                [[1,2], [400, 50], [50, 50]],
                                [[1,2], [400, 100], [50, 50]],
                                [[1,2], [400, 200], [50, 50]],
                                [[1,2], [400, 400], [50, 50]],
                                ],
            # "results_delay_in_clicks": [20, 20, 20],
            # "tracking_delay": 15
            # "N_contacts_max": 100,
            # "work_other_ratio": 0.5,0.6These
            # "N_init": [100, 1000]
            # "rho": 0.1,
             "beta": [0.007],
             #"make_initial_infections_at_kommune": False,
            # "interventions_to_apply": [[1, 2, 3, 4, 5, 6]],
             "intervention_removal_delay_in_clicks": [20],
             "make_restrictions_at_kommune_level": [True],
            # "N_events": 1000,
            # "mu": 20,
            #"tracking_rates": [1.0, 0.5,0.1]            
            "tracking_delay": [10,30],
             "day_max": 183,
            # "event_size_max": 50,
        },
    ]

else:
    all_simulation_parameters = utils.get_simulation_parameters()


# all_simulation_parameters = utils.get_simulation_parameters()
# x = x

#%%

N_runs = 1 if utils.is_local_computer() else N_runs

N_files_total = 0


# if __name__ == "__main__":

with Timer() as t:

    if dry_run:
        print("\n\nRunning a dry run, nothing will actually be simulated.!!!\n\n")

    if force_rerun:
        print("Notice: forced rerun is set to True")

    for d_simulation_parameters in all_simulation_parameters:
        # break
        verbose = True
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

# %%
