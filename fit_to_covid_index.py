import numpy as np
import pandas as pd
import datetime

from scipy.stats import norm

df_index = pd.read_feather("Data/covid_index.feather")

# Find the index for the starting date
ind = np.where(df_index["date"] == datetime.date(2021, 1, 1))[0][0]

# Only fit to data after this date
logI       = df_index["logI"][ind:]
logI_sigma = df_index["logI_sd"][ind:]




# Determine the model data
logI_model = logI + 0.1*np.random.uniform(np.shape(logI))



# Calculate (log) proability for every point
log_prop = norm.logpdf(logI_model, loc=logI, scale=logI_sigma)

# Determine the  log likelihood
loglikelihood = np.sum(log_prop)


