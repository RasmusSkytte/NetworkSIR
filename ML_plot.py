import numpy as np
import pandas as pd

from datetime import datetime

from scipy.stats import norm

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
    f = 0.01
    N = 5
    n_sigma = 0
    num_cores_max = 1
else :
    f = 0.1
    N = 1
    n_sigma = 2
    num_cores_max = 30

# Scale the population
params["N_tot"]  = int(params["N_tot"]  * f)
params["N_init"] = int(params["N_init"] * f)
params["R_init"] = int(params["R_init"] * f)


# Store the variables we loop over
beta = params["beta"]
beta_sigma = 0.00025

beta_UK_multiplier = params["beta_UK_multiplier"]
beta_UK_multiplier_sigma = 0.1

N_init = params["N_init"]
N_init_sigma = 1000 * f

N_init_UK_frac = params["N_init_UK_frac"]
N_init_UK_frac_sigma = 0.005


# Define the noise function
noise = lambda m, s, d : m if s == 0 else np.round(m + np.linspace(-s, s, 2*s + 1) * d, 5)

z = np.arange(n_sigma, -1, -1)
p = norm.pdf(z)
p /= p.max()

N_runs = np.floor(p * N).astype(int)

# Run the ML plot loop
N_files_total = 0
for s, n in zip(z, N_runs) :

    params["beta"]               = noise(beta,               s, beta_sigma)
    params["beta_UK_multiplier"] = noise(beta_UK_multiplier, s, beta_UK_multiplier_sigma)
    params["N_init"]             = noise(N_init,             s, N_init_sigma)
    params["N_init_UK_frac"]     = noise(N_init_UK_frac,     s, N_init_UK_frac_sigma)

    if __name__ == "__main__":
        with Timer() as t:

            N_files_total +=  simulation.run_simulations(params, N_runs=n, num_cores_max=num_cores_max)

print(f"\n{N_files_total:,} files were generated, total duration {utils.format_time(t.elapsed)}")
print("Finished simulating!")