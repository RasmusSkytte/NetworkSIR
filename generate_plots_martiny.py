import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import True_
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# from iminuit import Minuit
from collections import defaultdict
import joblib
from importlib import reload
from src.utils import utils
from src import plot
from src import file_loaders
from src import rc_params
from src import fits
from src import database

rc_params.set_rc_params(dpi=100)
num_cores_max = 30

make_1D_scan = True
force_rerun = True
verbose = True
make_fits = False


#%%

reload(plot)
reload(file_loaders)

abm_files = file_loaders.ABM_simulations(verbose=True)
N_files = len(abm_files)

#%%

reload(plot)

network_files = file_loaders.ABM_simulations(base_dir="Data/network", filetype="hdf5")
plot.plot_corona_type(
    network_files,
    force_rerun=force_rerun,
    xlim=(15, 100),
    N_max_runs=3,
    reposition_x_axis=True,
    normalize=False,
)

