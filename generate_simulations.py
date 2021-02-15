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

start_date = datetime(2020, 12, 21)
end_date   = datetime(2021, 3, 1)

if utils.is_local_computer():

    N_runs = 1

    # Fraction of population to simulate
    f = 0.1

    noise = lambda m, d : m
    #noise = lambda m, d : np.round(m + np.linspace(-d, d, 3), 5)
    linspace = lambda start, stop : np.round(np.linspace(start, stop, 3), 5)

    verbose = False
    num_cores_max = 1

else :

    N_runs = 3

    # Fraction of population to simulate
    f = 0.1

    noise = lambda m, d : np.round(m + np.linspace(-d, d, 5), 5)
    linspace = lambda start, stop : np.round(np.linspace(start, stop, 3), 5)

    verbose = False
    num_cores_max = 30


# all_simulation_parameters = [
#     {
#         "N_tot": int(5_800_000 * f),
#         "rho": 0.1,
#         #"epsilon_rho": 1,
#         "contact_matrices_name": "2021_fase1_sce1",                  # The target activity in the society
#         #
#         "Intervention_contact_matrices_name": [["ned2021jan"]],         # Nedlukningen i januar
#         #
#         "restriction_thresholds": [[ 0, (datetime(2021, 2, 8) - start_date).days]],
#         #
#         "threshold_interventions_to_apply": [[3]],          # 3: Matrix intervention
#         #
#         "start_date_offset" : (start_date - datetime(2020, 12, 28)).days,    # Simulation start date - vaccination start date
#         "day_max": (end_date - start_date).days,
#         #
#         "beta": noise(0.0095, 0.0005),
#         #"beta": [0.009, 0.0095, 0.01, 0.0105, 0.011],
#         "beta_UK_multiplier": noise(1.3, 0.2),
#         #"beta_UK_multiplier": [1.1, 1.3, 1.5, 1.7],
#         "lambda_I": 4 / 2.52,
#         "lambda_E": 4 / 2.5,
#         #
#         "N_init": noise(30000 * f, 2000 * f),
#         "N_init_UK_frac": noise(0.02, 0.01),
#         #"N_init_UK_frac": [0.00, 0.01, 0.02, 0.03],
#         #
#         "R_init": int(5_800_000 * f * 0.03),
#         #
#         "Intervention_vaccination_effect_delays" : [[10, 21]],
#         "Intervention_vaccination_efficacies" : [[0.95, 0.65]],
#         #
#         "do_interventions": True,
#         "continuous_interventions_to_apply":  [[1, 2, 3, 4, 5]],
#         "intervention_removal_delay_in_clicks": [0],
#         "make_restrictions_at_kommune_level": [False],
#     },
# ]

all_simulation_parameters = [
    {
        "N_tot": int(5_800_000 * f),
        "rho": 0.1,
        #"epsilon_rho": 1,
        "contact_matrices_name": [["2021_fase1_sce1"], ["2021_fase1_sce2"]],                  # The target activity in the society
        #
        "Intervention_contact_matrices_name": [["ned2021jan"]],         # Nedlukningen i januar
        #
        "restriction_thresholds": [[ 0, (datetime(2021, 2, 8) - start_date).days]],
        #
        "threshold_interventions_to_apply": [[3]],          # 3: Matrix intervention
        #
        "start_date_offset" : (start_date - datetime(2020, 12, 28)).days,    # Simulation start date - vaccination start date
        "day_max": (end_date - start_date).days,
        #
        "beta": noise(0.0095, 0.0005),
        #"beta": [0.009, 0.0095, 0.01, 0.0105, 0.011],
        "beta_UK_multiplier": noise(1.5, 0.2),
        #"beta_UK_multiplier": [1.1, 1.3, 1.5, 1.7],
        "lambda_I": 4 / 2.52,
        "lambda_E": 4 / 2.5,
        #
        "N_init": noise(30000 * f, 2000 * f),
        "N_init_UK_frac": noise(0.02, 0.01),
        #"N_init_UK_frac": [0.00, 0.01, 0.02, 0.03],
        #
        "R_init": int(5_800_000 * f * 0.03),
        #
        "Intervention_vaccination_effect_delays" : [[10, 21]],
        "Intervention_vaccination_efficacies" : [[0.95, 0.65]],
        #
        "do_interventions": True,
        "continuous_interventions_to_apply":  [[1, 2, 3, 4, 5]],
        "intervention_removal_delay_in_clicks": [0],
        "make_restrictions_at_kommune_level": [False],
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
                save_csv=False,
            )

        N_files_total += N_files

    print(f"\n{N_files_total:,} files were generated, total duration {utils.format_time(t.elapsed)}")
    print("Finished simulating!")
