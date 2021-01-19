import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import scipy as sp

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

def fit_exponential(x, y):	
	y = np.array(y)	
	popt, pcov = sp.optimize.curve_fit(exponential, x, y, p0=(y[0], 1, 1), bounds=([y[0]*0.5,0,0], [y[0]*1.5,2,100]))
	return popt, pcov

def exponential(x, a, b, c, T = 8):
   		return a * b ** (x / T) + c

def simple_ratio_with_symmetric_smoothing(x, n_smooth, T=8):
	x_smoothed = [np.sum(x[i-n_smooth:i+n_smooth])/(2*n_smooth+1) for i in range(n_smooth,len(x)-n_smooth)]
	return [x_smoothed[i+T]/x_smoothed[i] for i in range(len(x_smoothed)-T)]

def fit_on_small_symmetric_range(x, window, T=8):
	x_smoothed = [np.sum(x[i-window:i+window])/(2*window+1) for i in range(window,len(x)-window)]
	results = []
	for i in range(window,len(x)-window):
		x_fit = np.array(x_smoothed[i-window:i+window])
		try:
			popt,_ = fit_exponential(np.arange(len(x_fit)), x_fit)
			results.append(popt[1])
		except:
			results.append(0)
	return results

def troels_fit(x, y):
	ey  = np.sqrt(y)

def pandemic_control_calc(N_infected):
    lambda_I = 0.5
    N_tot = 580000
    I_crit = 2000.0*N_tot/5_800_000/lambda_I*4
    tal = 500
    b = np.log(tal)/I_crit
    #return (1.0/(1+np.exp(-b*(N_infected-I_crit))))
    return (1.0/(1.0+np.exp(-b*(N_infected-I_crit)))-(1/(1+tal)))*((tal+1)/tal)

def analyse_single_ABM_simulation(cfg, abm_files, network_files, fi_list, pc_list, name_list ):
    filenames = abm_files.cfg_to_filenames(cfg)
    network_filenames = network_files.cfg_to_filenames(cfg)

    N_tot = cfg.N_tot
    fig, axes = plt.subplots(ncols=2, figsize=(16, 7))
    fig.subplots_adjust(top=0.8)
    i = 0
    for  filename, network_filename in zip(filenames, network_filenames):
        df = file_loaders.pandas_load_file(filename)
        day_found_infected, R_true, freedom_impact, pandemic_control, my_state = file_loaders.load_Network_file(network_filename)
        t = df["time"].values
        pandemic_control2 = pandemic_control_calc(df["I"])
        label = r"ABM" if i == 0 else None
        #print("n_inf", np.sum([1 for day in day_found_infected if day >=0]), "mean", np.mean(day_found_infected))
        #axes[0].hist(day_found_infected[day_found_infected>=0], bins = range(100))
        axes[0].plot(R_true[1:], lw=4, c="k", label=label)
        axes[0].plot(freedom_impact[1:], lw=4, c="b", label=label)
        axes[0].plot(pandemic_control[1:], lw=4, c="r", label=label)
        axes[1].plot(t, df["I"],lw=4, c="k", label=label)
        fi_list.append(np.mean(freedom_impact[1:]))
        pc_list.append(np.mean(pandemic_control2))
        print("filename", "R_mean", np.mean(R_true[1:]), "freedom_impact", np.mean(freedom_impact[1:]),"pandemic_control", np.mean(pandemic_control[1:]),"pandemic_control2",np.mean(pandemic_control2))

        if i in range(9,15):
                name = str(cfg.threshold_info[1]) + str(cfg.threshold_info[2])
        else:
            name = str(cfg.tracking_delay)# + " " + str(cfg.tracking_rates) 
        name_list.append(name)      
        # popt, _ = fit_exponential(t, df["I1"]/2)
        # axes[0].plot(t, exponential(t, *popt), label="Fitted Curve") #same as line above \/
        # RS = [popt[1]]
        # l = int(len(t)/4)
        # popt, _ = fit_exponential(t[l:], df["I1"][l:]/2)
        # RS.append(popt[1])
        # axes[0].plot(t, exponential(t, *popt), label="shorter Fitted Curve") #same as line above \/
        # title = "contact number" + str(popt[1])
        # axes[0].set_title(title)

        # axes[1].plot(t[30:-30-80],simple_ratio_with_symmetric_smoothing(df["I1"]/2,30,80),label="simple 3 day smoothing, real")
        # axes[1].plot(range(1,93),simple_ratio_with_symmetric_smoothing(np.bincount(day_found_infected[day_found_infected>=0]),1,8),label="simple 1 day smoothing, tested")
        # axes[1].plot(range(3,91),simple_ratio_with_symmetric_smoothing(np.bincount(day_found_infected[day_found_infected>=0]),3,8),label="simple 3 day smoothing, tested")
        # axes[1].plot(range(3,99),fit_on_small_symmetric_range(np.bincount(day_found_infected[day_found_infected>=0]),3,8),label="fit 3 day smoothing, tested")
        # axes[1].plot(range(5,89),simple_ratio_with_symmetric_smoothing(np.bincount(day_found_infected[day_found_infected>=0]),5,8),label="simple 5 day smoothing, tested")
        # axes[1].plot(range(5,97), fit_on_small_symmetric_range(np.bincount(day_found_infected[day_found_infected>=0]),5,8),label="fit 5 day smoothing, tested")
        # axes[1].plot([1,100],[RS[0],RS[0]])
        # axes[1].plot([1,100],[RS[1],RS[1]])
        i += 1
        #axes[1].legend()
    

    return fig, axes, fi_list, pc_list,name_list


rc_params.set_rc_params()

#%%

reload(plot)
reload(file_loaders)

abm_files = file_loaders.ABM_simulations(verbose=True)
network_files = file_loaders.ABM_simulations(base_dir="Data/network", filetype="hdf5")
pdf_name = Path(f"Figures/data_anal.pdf")
utils.make_sure_folder_exist(pdf_name)
with PdfPages(pdf_name) as pdf:
        fi_list = []
        pc_list = []
        name_list = []
        # for ABM_parameter in tqdm(abm_files.keys, desc="Plotting individual ABM parameters"):
        for cfg in tqdm(
            abm_files.iter_cfgs(),
            desc="Plotting individual ABM parameters",
            total=len(abm_files.cfgs),
        ):

            # break


            fig, _, fi_list, pc_list, name_list = analyse_single_ABM_simulation(cfg, abm_files, network_files, fi_list, pc_list, name_list)


            pdf.savefig(fig, dpi=100)
            plt.close("all")

        fig, axes = plt.subplots(ncols=1, figsize=(16, 7))
        fig.subplots_adjust(top=0.8)
        
        for i in range(9,31):      
            if i in range(15):
                color = 'b' 
            elif i in range(15,19):
                color = 'g'
            elif i in range(19,23):
                color = 'r'
            elif i in range(23,27):
                color = 'm'
            elif i in range(27,31):
                color = 'k'
            axes.scatter(fi_list[i],pc_list[i], c =color, label=name_list[i])
            axes.text(fi_list[i]*1.01,pc_list[i]*1.01,name_list[i],fontsize=12)
        #axes.legend()
        pdf.savefig(fig, dpi=100)
        plt.close("all")

        fig, axes = plt.subplots(ncols=1, figsize=(16, 7))
        fig.subplots_adjust(top=0.8)
        
        for i in range(31,67):      
            if i in range(31,40):
                color = 'g'
            elif i in range(40,49):
                color = 'r'
            elif i in range(49,58):
                color = 'm'
            elif i in range(58,66):
                color = 'k'
            if i!=57 and i!=66:    
                axes.scatter(fi_list[i],pc_list[i], c = color)
                axes.text(fi_list[i]*1.01,pc_list[i]*1.01, str(i - 31),fontsize=12)
        
        #axes.legend()
        pdf.savefig(fig, dpi=100)
        plt.close("all")
        fig, axes = plt.subplots(ncols=1, figsize=(16, 7))
        fig.subplots_adjust(top=0.8)
        
        for j in range(67,103): 
            i = j - 36    
            if i in range(31,40):
                color = 'g'
            elif i in range(40,49):
                color = 'r'
            elif i in range(49,58):
                color = 'm'
            elif i in range(58,66):
                color = 'k'
            if i!=57 and i!=66:       
                axes.scatter(fi_list[j],pc_list[j], c = color)
                axes.text(fi_list[j]*1.01,pc_list[j]*1.01, str(i - 31),fontsize=12)
        
        #axes.legend()
        pdf.savefig(fig, dpi=100)
        plt.close("all")

        fig, axes = plt.subplots(ncols=1, figsize=(16, 7))
        fig.subplots_adjust(top=0.8)
        
        for j in range(103,len(fi_list)):      
            i = j - 72
            if i in range(31,40):
                color = 'g'
            elif i in range(40,49):
                color = 'r'
            elif i in range(49,58):
                color = 'm'
            elif i in range(58,66):
                color = 'k'
            if i!=57 and i!=66:  
                axes.scatter(fi_list[j],pc_list[j], c = color)
                axes.text(fi_list[j]*1.01,pc_list[j]*1.01, str(i - 31),fontsize=12)
        
        #axes.legend()
        pdf.savefig(fig, dpi=100)
        plt.close("all")


        #one fig
        fig, axes = plt.subplots(ncols=1, figsize=(16, 7))
        fig.subplots_adjust(top=0.8)

        for i in range(31,67):      
            
            color = 'g'
            
            if i!=57 and i!=66:    
                axes.scatter(fi_list[i],pc_list[i], c = color)
                #axes.text(fi_list[i]*1.01,pc_list[i]*1.01, str(i - 31),fontsize=12)
        
       
        
        for j in range(67,103): 
            i = j - 36    
            color = 'b'
            if i!=57 and i!=66:       
                axes.scatter(fi_list[j],pc_list[j], c = color)
                #axes.text(fi_list[j]*1.01,pc_list[j]*1.01, str(i - 31),fontsize=12)
        
        
        for j in range(103,139):      
            i = j - 72
            
            color = 'm'
            
            if i!=57 and i!=66:  
                axes.scatter(fi_list[j],pc_list[j], c = color)
                #axes.text(fi_list[j]*1.01,pc_list[j]*1.01, str(i - 31),fontsize=12)

        for j in range(139, len(fi_list)):      
            i = j - 108            
            color = 'r'
            
            axes.scatter(fi_list[j],pc_list[j], c = color)
                #axes.text(fi_list[j]*1.01,pc_list[j]*1.01, str(i - 31),fontsize=12)
        
        #axes.legend()
        pdf.savefig(fig, dpi=100)
        plt.close("all")




  
