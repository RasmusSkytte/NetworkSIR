import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm

from tqdm import tqdm

from importlib import reload
from src.utils import utils
from src import file_loaders
from src import rc_params

if utils.is_local_computer() :
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates



def aggregate_array(arr, chunk_size=10) :

    # Get the average number per day
    days = int(len(arr) / chunk_size)
    for k in range(days) :
        arr[k] = np.mean(arr[k*chunk_size:(k+1)*chunk_size])
    return arr[:days]

def load_from_file(filename) :

    cfg = utils.read_cfg_from_hdf5_file(filename)

    # Load the csv summery file
    df = file_loaders.pandas_load_file(filename)

    # Extract the values
    I_tot    = df["I"].to_numpy() 
    I_uk     = df["I^V_1"].to_numpy()

    # Get daily averages
    I_tot = aggregate_array(I_tot)
    I_uk = aggregate_array(I_uk)

    # Scale the number of infected
    I_tot_scaled = I_tot / 2.7 * (5_800_000 / cfg.network.N_tot)

    # Get the fraction of UK variants
    with np.errstate(divide='ignore', invalid='ignore'):
        f = I_uk / I_tot
        f[np.isnan(f)] = -1

    return(I_tot_scaled, f)


def compute_likelihood(I_tot_scaled, f, index, fraction) :
    
    # Compute the likelihood
    ll =  compute_loglikelihood_covid_index(I_tot_scaled, index)
    ll += compute_loglikelihood_fraction_uk(f, fraction)

    return ll


def compute_loglikelihood_covid_index(I, index):

    # Unpack values
    covid_index, covid_index_sigma, covid_index_offset = index

    if len(I) >= len(covid_index) + covid_index_offset :

        # Get the range corresponding to the tests
        I_model = I[covid_index_offset:covid_index_offset+len(covid_index)]

        # Model input is number of infected. We assume 80.000 daily tests in the model
        logK_model = np.log(I_model) - beta * np.log(80_000)

        # Calculate (log) proability for every point
        log_prop = norm.logpdf(logK_model, loc=covid_index, scale=covid_index_sigma)

        # Determine the log likelihood
        return np.sum(log_prop)

    else :
        return np.nan

def compute_loglikelihood_fraction_uk(f, fraction):

    # Unpack values
    fraction, fraction_sigma, fraction_offset = fraction

    # Get weekly values
    f = aggregate_array(f, chunk_size=7)

    if len(f) >= len(fraction) + fraction_offset :

        # Get the range corresponding to the tests
        fraction_model = f[fraction_offset:fraction_offset+len(fraction)]

        # Calculate (log) proability for every point
        log_prop = norm.logpdf(fraction_model, loc=fraction, scale=fraction_sigma)

        # Determine the log likelihood
        return np.sum(log_prop)

    else :
        return np.nan

rc_params.set_rc_params()

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
covid_index_offset = (datetime(2021, 1, 1) - datetime(2020, 12, 8)).days

fraction        = np.array([0.04,  0.074,  0.13,  0.2])
fraction_sigma  = np.array([0.006, 0.0075, 0.015, 0.016])
fraction_offset = 2


# Load the ABM simulations
abm_files = file_loaders.ABM_simulations(base_dir="Output/ABM", subset=None, verbose=True)

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
        tmp_ll = compute_likelihood(I_tot_scaled, f, 
                                    (logK, logK_sigma, covid_index_offset), 
                                    (fraction, fraction_sigma, fraction_offset))

        ll.append(tmp_ll)

    # Store loglikelihoods
    lls.append(np.mean(ll))

lls = np.array(lls)
cfgs = [cfg for cfg in abm_files.iter_cfgs()]
cfg_best = cfgs[np.nanargmax(lls)]
ll_best = lls[np.nanargmax(lls)]
print("--- Best parameters ---")
print(f"loglikelihood : {ll_best:.3f}")
print(f"beta : {cfg_best.beta:.5f}")
print(f"beta_UK_multiplier : {cfg_best.beta_UK_multiplier:.3f}")
print(f"N_init : {cfg_best.N_init:.0f}")
print(f"N_init_UK_frac : {cfg_best.N_init_UK_frac:.3f}")



betas = np.array([cfg.beta for cfg in cfgs])
rel_betas = np.array([cfg.beta_UK_multiplier for cfg in cfgs])

N_init = np.array([cfg.N_init for cfg in cfgs])
N_init_UK_frac = np.array([cfg.N_init_UK_frac for cfg in cfgs])

best = lambda arr : np.array([np.mean(lls[arr == v]) for v in np.unique(arr)])
err = lambda arr : np.array([np.std(lls[arr == v]) for v in np.unique(arr)])

if not utils.is_local_computer() :
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

    plt.savefig('Figures/test.png')

else :
    def terminal_printer(name, arr, val) :
    
        I = np.argmax(np.unique(arr) == val)
        out_string = "["
        for i in range(len(arr)) :
            if i == I :
                out_string += " +"
            else :
                out_string += " -"
        out_string += " ]"
        print(name + "\t" + out_string)

    print("--- Maximum likelihood value locations ---")
    terminal_printer("beta* :    ", betas,          cfg_best.beta)
    terminal_printer("rel_beta* :", rel_betas,      cfg_best.beta_UK_multiplier)
    terminal_printer("N_init* :  ", N_init,         cfg_best.N_init)
    terminal_printer("N_UK* :    ", N_init_UK_frac, cfg_best.N_init_UK_frac)
    