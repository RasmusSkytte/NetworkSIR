import numpy as np
import pandas as pd
from datetime import datetime

from tqdm import tqdm

from src.utils import utils
from src.simulation import simulation
from src import file_loaders


if utils.is_local_computer() :
    from src import rc_params
    import matplotlib.pyplot as plt

from contexttimer import Timer


params, start_date = utils.load_params("cfg/simulation_parameters_debugging.yaml")

if utils.is_local_computer():
    f = 0.1
    N_runs = 1
    num_cores_max = 1
else :
    f = 0.1
    N_runs = 1
    num_cores_max = 30

# Scale the population
params["N_tot"]  = int(params["N_tot"]  * f)
params["N_init"] = int(params["N_init"] * f)
params["R_init"] = int(params["R_init"] * f)


N_files_total = 0
if __name__ == "__main__":
    with Timer() as t:

        N_files_total +=  simulation.run_simulations(params, N_runs=N_runs, num_cores_max=num_cores_max)

    print(f"\n{N_files_total:,} files were generated, total duration {utils.format_time(t.elapsed)}")
    print("Finished simulating!")