
import numpy as np
from scipy.stats import norm

from src.utils import utils
from src import file_loaders


def aggregate_array(arr, chunk_size=10) :

    chunks = int(len(arr) / chunk_size)
    for k in range(chunks) :
        arr[k] = np.mean(arr[k*chunk_size:(k+1)*chunk_size])
    return arr[:chunks]

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
    covid_index, covid_index_sigma, covid_index_offset, beta = index
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