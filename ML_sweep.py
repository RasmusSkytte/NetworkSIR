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


params, start_date = utils.load_params("cfg/simulation_parameters_2021_fase1.yaml")

if utils.is_local_computer():
    f = 0.1
    #noise = lambda m, d : m
    n_sigma = 1
    num_cores_max = 3
else :
    f = 0.1
    n_sigma = 2
    num_cores_max = 30

noise = lambda m, d : np.round(m + np.linspace(-(n_sigma * d), (n_sigma * d), 2*n_sigma + 1), 5)

# Sweep around parameter set
params["beta"]               = noise(params["beta"], 0.00025)
params["beta_UK_multiplier"] = noise(params["beta_UK_multiplier"], 0.1)
params["N_init"]             = noise(params["N_init"] * f, 1000 * f)
params["N_init_UK_frac"]     = noise(params["N_init_UK_frac"], 0.005)

# Scale the population
params["N_tot"]  = int(params["N_tot"]  * f)
params["N_init"] = int(params["N_init"] * f)
params["R_init"] = int(params["R_init"] * f)


N_files_total = 0
if __name__ == "__main__":
    with Timer() as t:

        N_files_total +=  simulation.run_simulations(params, N_runs=25, num_cores_max=num_cores_max)

    print(f"\n{N_files_total:,} files were generated, total duration {utils.format_time(t.elapsed)}")
    print("Finished simulating!")



from src.analysis.helpers import *

# Load the covid index data
df_index = pd.read_feather("Data/covid_index.feather")

# Get the beta value (Here, scaling parameter for the index cases)
beta       = df_index["beta"][0]
beta_simga = df_index["beta_sd"][0]

# Find the index for the starting date
ind = np.where(df_index["date"] == datetime(2021, 1, 1).date())[0][0]

# Only fit to data after this date
logK       = df_index["logI"][ind:]     # Renaming the index I to index K to avoid confusion with I state in SIR model
logK_sigma = df_index["logI_sd"][ind:]

# Determine the covid_index_offset
covid_index_offset = (datetime(2021, 1, 1).date() - start_date).days

s = np.array([  76,  148,  275,  460,  510,  617, 101])
n = np.array([3654, 4020, 3901, 3579, 2570, 2003, 225])
p = s / n
p_var = p * (1 - p) / n

#fraction        = np.array([0.04,  0.074,  0.13,  0.2])
#fraction_sigma  = np.array([0.006, 0.0075, 0.015, 0.016])
fraction = p
fraction_sigma = 2 * np.sqrt(p)
fraction_offset = 1


for subset in [{"contact_matrices_name" : "2021_fase1_sce1"}, {"contact_matrices_name" : "2021_fase1_sce2"}] :

    print(subset)

    # Load the ABM simulations
    abm_files = file_loaders.ABM_simulations(base_dir="Output/ABM", subset=subset, verbose=True)

    lls     = []

    for cfg in tqdm(
        abm_files.iter_cfgs(),
        desc="Calculating log-likelihoods",
        total=len(abm_files.cfgs)) :

        ll = []

        for filename in abm_files.cfg_to_filenames(cfg) :

            # Load
            I_tot_scaled, f = load_from_file(filename)

            # Evaluate
            tmp_ll =  0.5 * compute_loglikelihood(I_tot_scaled, (logK, logK_sigma, covid_index_offset), transformation_function = lambda x : np.log(x) - beta * np.log(80_000))
            tmp_ll += 0.5 * compute_loglikelihood(f, (fraction, fraction_sigma, fraction_offset))

            ll.append(tmp_ll)

        # Store loglikelihoods
        lls.append(np.mean(ll))

    lls = np.array(lls)
    cfgs = [cfg for cfg in abm_files.iter_cfgs()]
    cfg_best = cfgs[np.nanargmax(lls)]
    ll_best = lls[np.nanargmax(lls)]
    print("--- Best parameters ---")
    print(f"Weighted loglikelihood : {ll_best:.3f}")
    print(f"beta : {cfg_best.beta:.5f}")
    print(f"beta_UK_multiplier : {cfg_best.beta_UK_multiplier:.3f}")
    print(f"N_init : {cfg_best.N_init:.0f}")
    print(f"N_init_UK_frac : {cfg_best.N_init_UK_frac:.3f}")



    betas     = np.array([cfg.beta               for cfg in cfgs])
    rel_betas = np.array([cfg.beta_UK_multiplier for cfg in cfgs])

    N_init         = np.array([cfg.N_init         for cfg in cfgs])
    N_init_UK_frac = np.array([cfg.N_init_UK_frac for cfg in cfgs])

    best = lambda arr : np.array([np.mean(lls[arr == v]) for v in np.unique(arr)])
    err  = lambda arr : np.array([np.std( lls[arr == v]) for v in np.unique(arr)])

    if False:# utils.is_local_computer() :
        rc_params.set_rc_params()

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        axes = axes.flatten()

        axes[0].errorbar(np.unique(betas), best(betas), yerr=err(betas), fmt="o")
        axes[0].scatter(cfg_best.beta, ll_best, fmt="x")
        axes[0].set_xlabel('beta')

        axes[1].errorbar(np.unique(N_init), best(N_init), yerr=err(N_init), fmt="o")
        axes[1].scatter(cfg_best.N_init, ll_best, fmt="x")
        axes[1].set_xlabel('N_init')

        axes[2].errorbar(np.unique(rel_betas), best(rel_betas), yerr=err(rel_betas), fmt="o")
        axes[2].scatter(cfg_best.beta_UK_multiplier, ll_best, fmt="x")
        axes[2].set_xlabel('rel. beta')

        axes[3].errorbar(np.unique(N_init_UK_frac), best(N_init_UK_frac), yerr=err(N_init_UK_frac), fmt="o")
        axes[3].scatter(cfg_best.N_init_UK_frac, ll_best, fmt="x")
        axes[3].set_xlabel('N_init_UK_frac')

        plt.savefig('Figures/LogLikelihood_parameters.png')


    def terminal_printer(name, arr, val) :
        u_arr = np.unique(arr)
        I = np.argmax(u_arr == val)

        out_string = "["
        for i in range(len(u_arr)) :
            if i == I :
                out_string += " *" + str(u_arr[i]) + "*"
            else :
                out_string += "  " + str(u_arr[i]) + " "
        out_string += " ]"
        print(name + "\t" + out_string)

    print("--- Maximum likelihood value locations ---")
    terminal_printer("beta* :      ", betas,          cfg_best.beta)
    terminal_printer("rel_beta* :  ", rel_betas,      cfg_best.beta_UK_multiplier)
    terminal_printer("N_init* :    ", N_init,         cfg_best.N_init)
    terminal_printer("N_UK_frac* : ", N_init_UK_frac, cfg_best.N_init_UK_frac)
