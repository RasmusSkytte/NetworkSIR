
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

    # Get weekly values
    I_tot_week = aggregate_array(I_tot, chunk_size=7)
    I_uk_week  = aggregate_array(I_uk,  chunk_size=7)

    # Get the fraction of UK variants
    with np.errstate(divide='ignore', invalid='ignore'):
        f = I_uk_week / I_tot_week
        f[np.isnan(f)] = -1

    return(I_tot_scaled, f)


def compute_loglikelihood(arr, data, transformation_function = lambda x : x) :

    # Unpack values
    data_values, data_sigma, data_offset = data
    if len(arr) >= len(data_values) + data_offset :

        # Get the range corresponding to the tests
        arr_model = arr[data_offset:data_offset+len(data_values)]

        # Calculate (log) proability for every point
        log_prop = norm.logpdf(transformation_function(arr_model), loc=data_values, scale=data_sigma)

        # Determine scaled the log likelihood
        return np.sum(log_prop) / len(log_prop)

    else :
        return np.nan
