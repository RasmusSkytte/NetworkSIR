import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import scipy as sp
import pickle
# from iminuit import Minuit
from collections import defaultdict
import joblib
from importlib import reload
from src.utils import utils
from src import plot
from src import file_loaders
from src import rc_params
from src import fits

from matplotlib.backends.backend_pdf import PdfPages
try:
    from src.utils import utils

    # from src import simulation_utils
    from src import file_loaders
    from src import SIR
except ImportError:
    import utils

    # import simulation_utils
    import file_loaders
    import SIR

rc_params.set_rc_params()

#%%

def analyse_single_ABM_simulation(cfg, abm_files, network_files, fi_list, pc_list):
    filenames = abm_files.cfg_to_filenames(cfg)
    network_filenames = network_files.cfg_to_filenames(cfg)

    N_tot = cfg.N_tot
    i = 0
    for  filename, network_filename in zip(filenames, network_filenames):
        df = file_loaders.pandas_load_file(filename)
        day_found_infected, R_true, freedom_impact, pandemic_control, my_state = file_loaders.load_Network_file(network_filename)
        t = df["time"].values
        pandemic_control2 = pandemic_control_calc(df["I"])
        fi_list.append(np.mean(freedom_impact[1:]))
        pc_list.append(np.mean(pandemic_control2))
    return fi_list, pc_list

def pandemic_control_calc(N_infected):
    lambda_I = 0.5
    N_tot = 580000
    I_crit = 2000.0*N_tot/5_800_000/lambda_I*4
    tal = 500
    b = np.log(tal)/I_crit
    #return (1.0/(1+np.exp(-b*(N_infected-I_crit))))
    return (1.0/(1.0+np.exp(-b*(N_infected-I_crit)))-(1/(1+tal)))*((tal+1)/tal)

reload(plot)
reload(file_loaders)
already_analysed = True
abm_files = file_loaders.ABM_simulations(verbose=True)
network_files = file_loaders.ABM_simulations(base_dir="Data/network", filetype="hdf5")
pdf_name = Path(f"Figures/scatter_plots.pdf")
utils.make_sure_folder_exist(pdf_name)
with PdfPages(pdf_name) as pdf:
    if not already_analysed:
        fi_list = []
        pc_list = []
        cfg_list = []
        # for ABM_parameter in tqdm(abm_files.keys, desc="Plotting individual ABM parameters"):
        for cfg in tqdm(
            abm_files.iter_cfgs(),
            desc="Plotting individual ABM parameters",
            total=len(abm_files.cfgs),
        ):

          
           fi_list, pc_list = analyse_single_ABM_simulation(cfg, abm_files, network_files, fi_list, pc_list)
           cfg_list.append(cfg)
    else:
        data = pickle.load(open('loadsofdata.pickle', 'rb') )
        fi_list,pc_list, cfg_list = data
    fig, axes = plt.subplots(ncols=1, figsize=(16, 12))
    fig.subplots_adjust(top=0.8)
    
    for i in range(len(fi_list)):
        marker = "."
        if cfg_list[i].N_tot > 1_000_000:
            if cfg_list[i].intervention_removal_delay_in_clicks==0:
                color = 'g'

            elif cfg_list[i].intervention_removal_delay_in_clicks==20:
                color = 'b'

            elif cfg_list[i].intervention_removal_delay_in_clicks==40:
                color = 'r'
            if cfg_list[i].beta<0.0051:
                size = 150
            else:
                size = 300
            if cfg_list[i].make_restrictions_at_kommune_level:
                if cfg_list[i].tracking_delay == 0:
                    marker = "v"
                elif cfg_list[i].tracking_delay == 10:    
                    marker = ">"
                elif cfg_list[i].tracking_delay == 30:    
                    marker = "^"
            else:
                if cfg_list[i].tracking_delay == 0:
                    marker = "1"
                elif cfg_list[i].tracking_delay == 10:    
                    marker = "2"
                elif cfg_list[i].tracking_delay == 30:    
                    marker = "3"


            axes.scatter(fi_list[i],pc_list[i],s=size, c = color, marker=marker)
    pdf.savefig(fig, dpi=100)
    plt.close("all")

    fig, axes = plt.subplots(ncols=1, figsize=(16, 12))
    fig.subplots_adjust(top=0.8)
    
    for i in range(len(fi_list)):
        marker = "."
        if cfg_list[i].make_restrictions_at_kommune_level:
            if cfg_list[i].intervention_removal_delay_in_clicks==0:
                color = 'g'

            elif cfg_list[i].intervention_removal_delay_in_clicks==20:
                color = 'b'

            elif cfg_list[i].intervention_removal_delay_in_clicks==40:
                color = 'r'
            if cfg_list[i].beta<0.0051:
                size = 150
            else:
                size = 300
            if cfg_list[i].tracking_delay == 0:
                marker = "v"
            elif cfg_list[i].tracking_delay == 10:    
                marker = ">"
            elif cfg_list[i].tracking_delay == 30:    
                marker = "^"
            axes.scatter(fi_list[i],pc_list[i],s=size, c = color, marker=marker)
            if marker == ".":
                print(cfg_list[i].make_restrictions_at_kommune_level, cfg_list[i].intervention_removal_delay_in_clicks, cfg_list[i].tracking_delay)

    pdf.savefig(fig, dpi=100)
    plt.close("all")

    fig, axes = plt.subplots(ncols=1, figsize=(16, 12))
    fig.subplots_adjust(top=0.8)
    
    for i in range(len(fi_list)):
        marker = "."
        if not cfg_list[i].make_restrictions_at_kommune_level:
            if cfg_list[i].intervention_removal_delay_in_clicks==0:
                color = 'g'

            elif cfg_list[i].intervention_removal_delay_in_clicks==20:
                color = 'b'

            elif cfg_list[i].intervention_removal_delay_in_clicks==40:
                color = 'r'
            if cfg_list[i].beta<0.0051:
                size = 150
            else:
                size = 300
            if cfg_list[i].tracking_delay == 0:
                marker = "1"
            elif cfg_list[i].tracking_delay == 10:    
                marker = "2"
            elif cfg_list[i].tracking_delay == 30:    
                marker = "3"
            axes.scatter(fi_list[i],pc_list[i],s=size, c = color, marker=marker)
            if marker == ".":
                print(cfg_list[i].make_restrictions_at_kommune_level, cfg_list[i].intervention_removal_delay_in_clicks, cfg_list[i].tracking_delay)

    pdf.savefig(fig, dpi=100)
    plt.close("all")

    fig, axes = plt.subplots(ncols=1, figsize=(16, 12))
    fig.subplots_adjust(top=0.8)
    
    for i in range(len(fi_list)):
        marker = "."
        if cfg_list[i].intervention_removal_delay_in_clicks == 20 and cfg_list[i].tracking_delay == 10:
            if cfg_list[i].beta<0.0051:
                color = "r"
            else:
                color = "b"

            
            axes.scatter(fi_list[i],pc_list[i], c = color, marker=marker)
           

    pdf.savefig(fig, dpi=100)
    plt.close("all")
   
    fig, axes = plt.subplots(ncols=1, figsize=(16, 12))
    fig.subplots_adjust(top=0.8)
    
    for i in range(len(fi_list)):
        marker = "."
        if cfg_list[i].intervention_removal_delay_in_clicks==20:
            if cfg_list[i].beta<0.0051:
                color = "r"
            else:
                color = "b"
            if cfg_list[i].tracking_delay == 0:
                marker = "v"
            elif cfg_list[i].tracking_delay == 10:    
                marker = ">"
            elif cfg_list[i].tracking_delay == 30:    
                marker = "^"    
            
            axes.scatter(fi_list[i],pc_list[i], c = color, marker=marker)
            
    pdf.savefig(fig, dpi=100)
    plt.close("all")

