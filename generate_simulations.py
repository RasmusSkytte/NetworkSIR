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


num_cores_max = 1
N_runs = 1


dry_run = False
force_rerun = False
verbose = True

#%%


if utils.is_local_computer():

    # Fraction of population to simulate
    f = 0.01

    all_simulation_parameters = [
        {
            "N_tot": int(5_800_000 * f),
            "weighted_random_initial_infections": True,
            "lambda_I": 4 / 2.52,
            "lambda_E": 4 / 2.5,
            "rho": 0.0,
            #"epsilon_rho": 1,
            "intervention_removal_delay_in_clicks": [20],
            "make_restrictions_at_kommune_level": [False],
            "burn_in": 0,
            "start_date_offset" : (datetime(2020, 12, 28) - datetime(2020, 12, 28)).days,    # Simulation start date - vaccination start date 
            # "N_tot": [58_000],
            # "make_random_initial_infections": True,
            # "weighted_random_initial_infections": True,
            # "test_delay_in_clicks": [0, 0, 25],
            # "results_delay_in_clicks": [[20, 20, 20]],
            # "tracking_delay": [0, 5, 10, 15, 20, 25, 30],
            # "weighted_random_initial_infections": True,
            "do_interventions": True,
            "threshold_interventions_to_apply": [[3]],
            "restriction_thresholds": [[1,5]],
            "continuous_interventions_to_apply":  [[1,2,3,4,5]],
            # "results_delay_in_clicks": [20, 20, 20],
            # "tracking_delay": 15
            # "N_contacts_max": 100,
            # "work_other_ratio": 0.5,
            "N_init": np.array([4500, 4600, 4700]) * f,
            # "N_init": [1000],
            "N_init_UK_frac": [0.03],
            #"work_other_ratio": 0.95,  # "algo 1"
            # "rho": 0.1,
            # "beta": [0.004],
            "beta": [0.0125],
            # "beta": [0.016, 0.018],
            "beta_UK_multiplier": [1.5],
            # "N_daily_vaccinations": [0],
            # "N_events": 1000,
            # "mu": 20,
            # "tracking_rates": [1.0, 0.5,0.1]
            "tracking_delay": [10],
            "day_max": 100,
            "days_of_vacci_start": 0, # number of days after vaccinations calender start. 31 = 1-st of feb.  # TODO: Try to properly include dates in the simulations
            #"verbose":True,
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

    if __name__ == "__main__" : # Needed for windows multiprocessing

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
                save_initial_network=True,
            )

        N_files_total += N_files

print(f"\n{N_files_total:,} files were generated, total duration {utils.format_time(t.elapsed)}")
print("Finished simulating!")

# %%
