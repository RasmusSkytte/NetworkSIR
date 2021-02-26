
import numpy as np
import pandas as pd

import datetime

from scipy.stats import norm

from src.utils import utils
from src.utils import file_loaders


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


def load_covid_index(start_date) :

    # Load the covid index data
    df_index = pd.read_feather("Data/covid_index_2021.feather")

    # Get the beta value (Here, scaling parameter for the index cases)
    beta       = df_index["beta"][0]
    beta_sigma = df_index["beta_sd"][0]

    # Find the index for the starting date
    ind = np.where(df_index["date"] == datetime.datetime(2021, 1, 1).date())[0][0]

    # Only fit to data after this date
    logK       = df_index["logI"][ind:]     # Renaming the index I to index K to avoid confusion with I state in SIR model
    logK_sigma = df_index["logI_sd"][ind:] / 3
    t          = df_index["date"][ind:]

    # Determine the covid_index_offset
    covid_index_offset = (datetime.datetime(2021, 1, 1).date() - start_date).days

    return (logK, logK_sigma, beta, covid_index_offset, t)


def load_b117_fraction() :

    #       uge     53     1     2     3     4     5     6
    s = np.array([  76,  148,  275,  460,  510,  617,  101])
    n = np.array([3654, 4020, 3901, 3579, 2570, 2003,  225])
    p = s / n
    p_var = p * (1 - p) / n

    fraction = p
    fraction_sigma = 2 * np.sqrt(p_var)
    fraction_offset = 1

    t = pd.date_range(start = datetime.datetime(2020, 12, 28), periods = len(fraction), freq = "W-SUN")

    return (fraction, fraction_sigma, fraction_offset, t)