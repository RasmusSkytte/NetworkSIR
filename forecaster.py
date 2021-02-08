
# Trin til at forecaste

# Trin 1, kør modellen med et i et stort parameter rum,


# Trin 2, sammenlign med de forventede test tal, og vælg de mest sandsynelige parametre

# Trin 3, kør modellen med et mindre parameter rum centreret omkring de mest sandsynelige parametre

# Trin 4, farvekod graferne efter deres likelihood


import numpy as np
import pandas as pd
import datetime

from src.utils import utils
from src.simulation import simulation
from contexttimer import Timer

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from tqdm import tqdm
from pathlib import Path
from importlib import reload
from src.utils import utils
from src import plot
from src import file_loaders
from src import rc_params





N_tot_max = False


num_cores_max = 1
N_runs = 1


dry_run     = False
force_rerun = True
verbose     = True





wide_parameter_sweep = {
        "N_tot": 200_000,
        "weighted_random_initial_infections": True,
        "lambda_I": 4 / 2.52,
        "lambda_E": 4 / 2.5,
        "rho": 0.0,
        "epsilon_rho": 1,
        "intervention_removal_delay_in_clicks": [21],
        "make_restrictions_at_kommune_level": [False],
        "burn_in": 0,
        "do_interventions": True,
        "interventions_to_apply": [[3, 4, 5, 6, 7]],
        "N_init": [45],
        "N_init_UK": [15],
        "beta": [0.0125],
        "beta_UK_multiplier": [1.5],
        "tracking_delay": [10],
        "day_max": 20,
        "days_of_vacci_start": 0 # number of days after vaccinations calender start. 31 = 1-st of feb.
    }





N_files_total = 0

with Timer() as t:

    if dry_run:
        print("\n\nRunning a dry run, nothing will actually be simulated.!!!\n\n")

    if force_rerun:
        print("Notice: forced rerun is set to True")

    # break
    verbose = True
    N_files = simulation.run_simulations(
        wide_parameter_sweep,
        N_runs=N_runs,
        num_cores_max=num_cores_max,
        N_tot_max=N_tot_max,
        verbose=verbose,
        force_rerun=force_rerun,
        dry_run=dry_run,
        save_csv=True,
        save_initial_network=False,
    )

    N_files_total += N_files

print(f"\n{N_files_total:,} files were generated, total duration {utils.format_time(t.elapsed)}")
print("Finished simulating!")






############## Import test data
df_index = pd.read_feather("Data/covid_index.feather")

# Find the index for the starting date
ind = np.where(df_index["date"] == datetime.date(2021, 1, 1))[0][0]






def pandemic_control_calc(N_infected):
    lambda_I = 0.5
    N_tot = 580000
    I_crit = 2000.0*N_tot/5_800_000/lambda_I*4
    tal = 500
    b = np.log(tal)/I_crit
    #return (1.0/(1+np.exp(-b*(N_infected-I_crit))))
    return (1.0/(1.0+np.exp(-b*(N_infected-I_crit)))-(1/(1+tal)))*((tal+1)/tal)




def analyse_single_ABM_simulation(cfg, abm_files, network_files, fi_list, pc_list, name_list,vaccinations_per_age_group, vaccination_schedule ):
    filenames = abm_files.cfg_to_filenames(cfg)
    network_filenames = network_files.cfg_to_filenames(cfg)

    N_tot = cfg.N_tot
    fig, axes = plt.subplots(ncols=2, figsize=(16, 7))
    fig.subplots_adjust(top=0.8)
    i = 0
    for  filename, network_filename in zip(filenames, network_filenames):
        df = file_loaders.pandas_load_file(filename)
        day_found_infected, R_true, freedom_impact, R_true_brit, my_state = file_loaders.load_Network_file(network_filename)
        t = df["time"].values
        pandemic_control2 = pandemic_control_calc(df["I"])
        label = r"ABM" if i == 0 else None
        #print("n_inf", np.sum([1 for day in day_found_infected if day >=0]), "mean", np.mean(day_found_infected))
        #axes[0].hist(day_found_infected[day_found_infected>=0], bins = range(100))
        #axes[0].plot(R_true[1:], lw=4, c="k", label=label)
        #axes[0].plot(R_true_brit[1:], lw=4, c="r", label=label)
        #axes[0].plot(freedom_impact[1:], lw=4, c="b", label=label)
        #axes[0].plot(pandemic_control[1:], lw=4, c="r", label=label)
        axes[1].plot(t, np.array(df["I"])/N_tot*5_800_000,lw=4, c="k", label=label)
        ids = np.array([int(ts) - vaccination_schedule[0] + 21 for ts in t])
        ids[ids < 0] == 0
        ids[ids > 140] == 0
        #vac_array = [np.sum(vaccinations_per_age_group[i])*N_tot/5_800_000 if i < 130 else _ for i in ids ]
        #if len(ids) > len(vac_array):
        #    axes[1].plot(ids[:len(vac_array)], vac_array)
        #else:
        #    axes[1].plot(ids, vac_array)
        #fi_list.append(np.mean(freedom_impact[1:]))
        #pc_list.append(np.mean(pandemic_control2))
        print("filename", "R_mean", np.mean(R_true[1:]), "freedom_impact", np.mean(freedom_impact[1:]),"R_true_brit", np.mean(R_true_brit[1:]),"pandemic_control2",np.mean(pandemic_control2))

        if i in range(9,15):
                name = str(cfg.N_init) + str(c)
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
        n_pos = [724, 886, 760, 773, 879, 754, 625, 652, 592, 668, 456, 431, 377, 488]
        dates = np.arange(19,33)
        axes[1].scatter(dates, n_pos)
        axes[1].set_xlim(19, 100)
        #axes[0].set_xlim(19, 100)
        axes[1].set_ylim(0, 2500)
        axes[1].set_ylabel("N infected")
        axes[1].set_xlabel("days into 2021")
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
vaccinations_per_age_group, _, vaccination_schedule = utils.load_vaccination_schedule()
vaccinations_per_age_group=vaccinations_per_age_group.astype(np.int64)
vaccination_schedule = np.arange(len(vaccination_schedule),dtype=np.int64) + 10
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


            fig, _, fi_list, pc_list, name_list = analyse_single_ABM_simulation(cfg, abm_files, network_files, fi_list, pc_list, name_list, vaccinations_per_age_group, vaccination_schedule)


            pdf.savefig(fig, dpi=100)
            plt.close("all")
