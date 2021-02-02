# import numpy as np TODO: Delete line
# from tqdm import tqdm TODO: Delete line
# import multiprocessing as mp TODO: Delete line
# from pathlib import Path TODO: Delete line
# from importlib import reload TODO: Delete line
from src.utils import utils
from src.simulation import simulation
# from functools import partial TODO: Delete line
# import yaml TODO: Delete line
from contexttimer import Timer

N_tot_max = False

num_cores_max = 1
N_runs = 1


dry_run = False
force_rerun = True
verbose = True

#%%


if utils.is_local_computer():

    all_simulation_parameters = [
        {

            "N_tot": 580000*2,
            #"N_tot": 6_000,
            # "make_random_initial_infections": True,
            # "weighted_random_initial_infections": True,
            # "test_delay_in_clicks": [0, 0, 25],
            #"results_delay_in_clicks": [20, 20, 20],
            #"tracking_delay": [0],
             "weighted_random_initial_infections": True,
             "lambda_I": 4/2.52,
             "lambda_E": 4/2.5,
             "threshold_info": [[[2,7], [150000, 150000], [200,200]]],
            # "tracking_delay": 15
            # "N_contacts_max": 100,
            # "work_other_ratio": 0.5, 0.6
            # "N_init": [100, 1000]
            "rho": 0.0,
            #"epsilon_rho": 1,
            # "make_initial_infections_at_kommune": False,
            # "interventions_to_apply": [[1, 2, 3, 4, 5, 6]],
            "intervention_removal_delay_in_clicks": [21],
            "make_restrictions_at_kommune_level": [False],
            "burn_in": 0,
            # "N_tot": [58_000],
            # "make_random_initial_infections": True,
            # "weighted_random_initial_infections": True,
            # "test_delay_in_clicks": [0, 0, 25],
            # "results_delay_in_clicks": [[20, 20, 20]],
            # "tracking_delay": [0, 5, 10, 15, 20, 25, 30],
            # "weighted_random_initial_infections": True,
            "do_interventions": True,
            "interventions_to_apply": [[3, 4, 5, 6, 7]],
            # "results_delay_in_clicks": [20, 20, 20],
            # "tracking_delay": 15
            # "N_contacts_max": 100,
            # "work_other_ratio": 0.5,
            "N_init": [1800,2000],
            # "N_init": [1000],
            "N_init_UK": [100],
            #"work_other_ratio": 0.95,  # "algo 1"
            # "rho": 0.1,
            # "beta": [0.004],
            "beta": [0.0125],
            # "beta": [0.016, 0.018],
            "beta_UK_multiplier": [1.5],
            # "outbreak_position_UK": ["københavn", "nordjylland"],
            "outbreak_position_UK": ["københavn"],
            # "N_daily_vaccinations": [0],
            # "N_events": 1000,
            # "mu": 20,
            # "tracking_rates": [1.0, 0.5,0.1]
            "tracking_delay": [10],
            "day_max": 100,
            "days_of_vacci_start": 0 # number of days after vaccinations calender start. 31 = 1-st of feb.

            #"verbose":True,
            # "event_size_max": 50,
            #"vaccinations": True,
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
            save_initial_network=False,
        )

        N_files_total += N_files

print(f"\n{N_files_total:,} files were generated, total duration {utils.format_time(t.elapsed)}")
print("Finished simulating!")

# %%
