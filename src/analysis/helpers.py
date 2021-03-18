
import numpy as np
import pandas as pd

import datetime

from scipy.stats import norm

from src.utils import utils
from src.utils import file_loaders

ref_tests = 100_000

def aggregate_array(arr, chunk_size=10) :

    tmp = arr.copy()

    shp = np.shape(tmp)

    if len(shp) == 1 :
        tmp = tmp.reshape(shp[0], 1)
        shp = np.shape(tmp)

    chunks = int(shp[0] / chunk_size)

    out_arr = np.zeros((chunks, shp[1]))
    for k in range(chunks) :
        out_arr[k, :] = np.mean(tmp[k*chunk_size:(k+1)*chunk_size, :], axis=0)

    if len(np.shape(arr)) == 1 :
        return np.squeeze(out_arr)
    else :
        return out_arr


def load_from_file(filename, start_date) :

    cfg = utils.read_cfg_from_hdf5_file(filename)

    # Load the csv summery file
    df = file_loaders.pandas_load_file(filename)




    # Find all columns with "T_"
    test_cols = [col for col in df.columns if 'T_l' in col]

    # Determine the output dimension
    N_labels     = len(np.unique(np.array([int(col.split('_')[2]) for col in test_cols])))
    N_variants   = len(np.unique(np.array([int(col.split('_')[4]) for col in test_cols])))
    N_age_groups = len(np.unique(np.array([int(col.split('_')[6]) for col in test_cols])))

    # Load into a multidimensional array
    stratified_infections = np.zeros((len(df), N_labels, N_variants, N_age_groups))

    for col in test_cols :
        l, v, a = (int(col.split('_')[2]), int(col.split('_')[4]), int(col.split('_')[6]))

        stratified_infections[:, l, v, a] = df[col]

    # Scale the tests
    stratified_infections *= (5_800_000 / cfg.network.N_tot) / 2.7




    # Find all columns with "V_"
    vaccine_cols = [col for col in df.columns if 'V_' in col]

    # Load into a multidimensional array
    stratified_vaccinations = np.zeros((len(df), N_age_groups))

    for col in vaccine_cols :
        a = int(col.split('_')[2])

        stratified_vaccinations[:, a] = df[col]



    # Convert to observables
    T_total      = np.sum(stratified_infections, axis=(1, 2, 3))

    T_variants   = np.sum(stratified_infections, axis=(1, 3))
    T_uk         = T_variants[:, 1]

    T_age_groups = np.sum(stratified_infections, axis=(1, 2))

    T_regions    = np.sum(stratified_infections, axis=(2, 3))

    V_age_groups = stratified_vaccinations


    # Get daily values
    T_total      = T_total
    T_variants   = T_variants
    T_uk         = T_uk
    T_age_groups = T_age_groups
    T_regions    = T_regions

    # Get weekly values
    # Remove days if not starting on a monday
    if start_date.weekday() > 0 :
        I = 7-start_date.weekday()
    else :
        I = 0

    T_total_week = aggregate_array(T_total[I:], chunk_size=7)
    T_uk_week    = aggregate_array(T_uk[I:],    chunk_size=7)

    # Get the fraction of UK variants
    with np.errstate(divide='ignore', invalid='ignore'):
        f = T_uk_week / T_total_week
        f[np.isnan(f)] = -1

    return T_total, f, T_age_groups, T_variants, T_regions, V_age_groups


def parse_time_ranges(start_date, end_date) :

    t_day = pd.date_range(start=start_date, end=end_date, freq="D")
    t_day = t_day[:-1]

    weeks =  [w for w in pd.unique(t_day.isocalendar().week) if np.sum(t_day.isocalendar().week == w) == 7]
    t_week = pd.to_datetime([date for date, dayofweek, week in zip(t_day, t_day.dayofweek, t_day.isocalendar().week) if (dayofweek == 6 and week in weeks)])

    return t_day, t_week


def compute_loglikelihood(input_data, validation_data, transformation_function = lambda x : x) :

    # Unpack values
    input_values, t_input = input_data
    data_values, data_sigma, t_data = validation_data

    intersection = t_input.intersection(t_data)

    arr_model = [val for val, t in zip(input_values, t_input) if t in intersection]

    arr_data  = [val for val, t in zip(data_values,  t_data)  if t in intersection]
    arr_sigma = [val for val, t in zip(data_sigma,   t_data)  if t in intersection]

    # Calculate (log) proability for every point
    log_prop = norm.logpdf(transformation_function(arr_model), loc=arr_data, scale=arr_sigma)

    # Determine scaled the log likelihood
    return np.sum(log_prop) / len(log_prop)


def load_covid_index() :

    # Load the covid index data
    df_index = pd.read_feather(file_loaders.load_yaml('cfg/files.yaml')['CovidIndex'])

    # Get the beta value (Here, scaling parameter for the index cases)
    beta       = df_index["beta"][0]
    beta_sigma = df_index["beta_sd"][0]


    # Only fit to data after this date
    logK       = df_index["logI"]     # Renaming the index I to index K to avoid confusion with I state in SIR model
    logK_sigma = df_index["logI_sd"]
    t          = df_index["date"]

    return (logK, logK_sigma, beta, pd.to_datetime(t))


def load_b117_fraction() :

    #       uge     53     1     2     3     4     5     6     7     8
    s = np.array([  80,  154,  284,  470,  518,  662,  922, 1570, 1472])
    n = np.array([3915, 4154, 4035, 3685, 2659, 2233, 1956, 2388, 1939])
    p = s / n
    p_var = p * (1 - p) / n

    fraction = p
    fraction_sigma = 2 * np.sqrt(p_var)

    t = pd.date_range(start='2020-12-28', periods=len(fraction), freq="W-SUN")

    return (fraction, fraction_sigma, t)


def load_infected_per_category(beta, category='AgeGr') :

    raw_data = pd.read_csv(file_loaders.load_yaml("cfg/files.yaml")["RegionData"], sep="\t")

    tests_per_age_group = pd.pivot_table(raw_data, values=['test'], index=['PrDate'], columns=[category],  aggfunc=np.sum).to_numpy().astype(float)
    tests_per_day = np.sum(tests_per_age_group, axis = 1)

    # Adjust to ref_tests level
    tests_per_age_group_adjusted = tests_per_age_group * ref_tests / np.repeat(tests_per_day.reshape(-1, 1), tests_per_age_group.shape[1], axis=1)

    data = pd.pivot_table(raw_data, values=['pos'], index=['PrDate'], columns=[category],  aggfunc=np.sum)
    positive_per_age_group = data.to_numpy().astype(float)
    positive_per_age_group *= (tests_per_age_group_adjusted / tests_per_age_group)**beta

    return pd.to_datetime(data.index), positive_per_age_group