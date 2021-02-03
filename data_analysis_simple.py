import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from importlib import reload
from src.utils import utils
from src import plot
from src import file_loaders
from src import rc_params

from matplotlib.backends.backend_pdf import PdfPages

from src.utils import utils
from src import file_loaders


def analyse_single_ABM_simulation(cfg, abm_files, fi_list, pc_list, name_list):
    filenames = abm_files.cfg_to_filenames(cfg)

    N_tot = cfg.N_tot
    fig, axes = plt.subplots(ncols=2, figsize=(16, 7))
    fig.subplots_adjust(top=0.8)

    i = 0
    for  filename in filenames:
        df = file_loaders.pandas_load_file(filename)

        print(df)

        t = df["time"].values

        label = r"ABM" if i == 0 else None


        axes[1].plot(t, np.array(df["I"])/N_tot*5_800_000,lw=4, c="k", label=label)



        axes[1].set_xlim(19, 100)
        #axes[0].set_xlim(19, 100)
        axes[1].set_ylim(0, 2500)
        axes[1].set_ylabel("N infected")
        axes[1].set_xlabel("days into 2021")

        i += 1

    return fig, axes, fi_list, pc_list, name_list


rc_params.set_rc_params()

reload(plot)
reload(file_loaders)

abm_files = file_loaders.ABM_simulations(verbose=True)

pdf_name = Path(f"Figures/data_analysis_simple.pdf")
utils.make_sure_folder_exist(pdf_name)

with PdfPages(pdf_name) as pdf:
        fi_list = []
        pc_list = []
        name_list = []

        for cfg in tqdm(
            abm_files.iter_cfgs(),
            desc="Plotting individual ABM parameters",
            total=len(abm_files.cfgs),
        ):

            fig, _, fi_list, pc_list, name_list = analyse_single_ABM_simulation(cfg, abm_files, fi_list, pc_list, name_list)

            pdf.savefig(fig, dpi=100)
            plt.close("all")
