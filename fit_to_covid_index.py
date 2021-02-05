import numpy as np
import pandas as pd
import datetime

from scipy.stats import norm

df_index = pd.read_feather("Data/covid_index.feather")

# Get the beta value (Here, scaling parameter for the index cases)
beta       = df_index["beta"][0]
beta_simga = df_index["beta_sd"][0]

# Find the index for the starting date
ind = np.where(df_index["date"] == datetime.date(2021, 1, 1))[0][0]

# Only fit to data after this date
logK       = df_index["logI"][ind:]     # Renaming the index I to index K to avoid confusion with I state in SIR model
logK_sigma = df_index["logI_sd"][ind:]




# Determine the model data
I_model = ....

# Model input is number of infected. We assume 80.000 daily tests in the model
logK_model = np.log(I_model) -  beta * np.log(80_000)

# Calculate (log) proability for every point
log_prop = norm.logpdf(logK_model, loc=logK, scale=logK_sigma)

# Determine the  log likelihood
loglikelihood = np.sum(log_prop)
