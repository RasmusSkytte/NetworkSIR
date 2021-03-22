from contexttimer import Timer

from tqdm import tqdm

import datetime

from src.utils      import utils
from src.simulation import simulation
from src.utils      import file_loaders

from src.analysis.helpers  import *
from src.analysis.plotters import *


# This runs simulations with specified percentage effects of the seasonality model
params, start_date = utils.load_params("cfg/analyzers/initializer.yaml")

if utils.is_local_computer():
    f = 0.01
    n_steps = 1
    num_cores_max = 1
    N_runs = 1
else :
    f = 0.5
    n_steps = 3
    num_cores_max = 5
    N_runs = 1


if num_cores_max == 1 :
    verbose = True
else :
    verbose = False


# Scale the population
params["N_tot"]  = int(params["N_tot"]  * f)
params["R_init"] = int(params["R_init"] * f)


N_files_total = 0
if __name__ == "__main__":
    with Timer() as t:

        N_files_total +=  simulation.run_simulations(params, N_runs=N_runs, num_cores_max=num_cores_max, verbose=verbose)

    print(f"\n{N_files_total:,} files were generated, total duration {utils.format_time(t.elapsed)}")
    print("Finished simulating!")

