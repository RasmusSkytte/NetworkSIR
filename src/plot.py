import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pandas.errors import EmptyDataError
from src import rc_params
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import pandas as pd
from matplotlib.ticker import EngFormatter
from collections import defaultdict
import warnings

try:
    from src import utils
    from src import simulation_utils
    from src import file_loaders
    from src import SIR
except ImportError:
    import utils
    import simulation_utils
    import file_loaders
    import SIR

rc_params.set_rc_params()


def compute_df_deterministic(cfg, variable, T_max=100):
    # checks that the curve has flattened out
    while True:
        df_deterministic = SIR.integrate(cfg, T_max, dt=0.01, ts=1)
        delta_1_day = df_deterministic[variable].iloc[-2] - df_deterministic[variable].iloc[-1]
        if variable == "R":
            delta_1_day *= -1
        delta_rel = delta_1_day / cfg.N_tot
        if 0 <= delta_rel < 1e-5:
            break
        T_max *= 1.1
    return df_deterministic


def plot_ABM_simulations(abm_files, force_rerun=False):

    # pdf_name = "test.pdf"
    pdf_name = Path(f"Figures/ABM_simulations.pdf")
    utils.make_sure_folder_exist(pdf_name)

    if pdf_name.exists() and not force_rerun:
        print(f"{pdf_name} already exists")
        return None

    with PdfPages(pdf_name) as pdf, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure was using constrained_layout==True")

        d_ylabel = {"I": "Infected", "R": "Recovered"}
        d_label_loc = {"I": "upper right", "R": "lower right"}

        for ABM_parameter in tqdm(abm_files.keys):
            # break

            cfg = utils.string_to_dict(ABM_parameter)

            fig, axes = plt.subplots(ncols=2, figsize=(18, 7), constrained_layout=True)
            fig.subplots_adjust(top=0.8)

            T_max = 0
            lw = 0.1 * 10 / np.sqrt(len(abm_files[ABM_parameter]))

            stochastic_noise_I = []
            stochastic_noise_R = []

            # file, i = abm_files[ABM_parameter][0], 0
            for i, file in enumerate(abm_files[ABM_parameter]):
                df = file_loaders.pandas_load_file(file)
                t = df["time"].values
                label = "ABM" if i == 0 else None

                axes[0].plot(t, df["I"], lw=lw, c="k", label=label)
                axes[1].plot(t, df["R"], lw=lw, c="k", label=label)

                if t.max() > T_max:
                    T_max = t.max()

                stochastic_noise_I.append(df["I"].max())
                stochastic_noise_R.append(df["R"].iloc[-1])

            for variable, ax in zip(["I", "R"], axes):

                df_deterministic = compute_df_deterministic(cfg, variable, T_max=T_max)

                ax.plot(
                    df_deterministic["time"], df_deterministic[variable], lw=2.5, color="red", label="SEIR",
                )
                leg = ax.legend(loc=d_label_loc[variable])
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(2.0)

                ax.set(xlabel="Time", ylim=(0, None), ylabel=d_ylabel[variable])
                # ax.set_xlabel('Time', ha='right')
                ax.xaxis.set_label_coords(0.91, -0.14)

                ax.yaxis.set_major_formatter(EngFormatter())

                ax.set_rasterized(True)
                ax.set_rasterization_zorder(0)

            names = [r"I_\mathrm{max}^\mathrm{ABM}", r"R_\infty^\mathrm{ABM}"]
            for name, x, ax in zip(names, [stochastic_noise_I, stochastic_noise_R], axes):

                mu, std = np.mean(x), utils.SDOM(x)

                n_digits = int(np.log10(utils.round_to_uncertainty(mu, std)[0])) + 1
                rel_uncertainty = std / mu
                s_mu = utils.human_format_scientific(mu, digits=n_digits)

                s = r"$ " + f"{name} = ({s_mu[0]}" + r"\pm " + f"{rel_uncertainty*100:.2}" + r"\% )" + r"\cdot " + f"{s_mu[1]}" + r"$"
                ax.text(
                    -0.1, -0.2, s, horizontalalignment="left", transform=ax.transAxes, fontsize=24,
                )

            title = utils.dict_to_title(cfg, len(abm_files[ABM_parameter]))
            fig.suptitle(title, fontsize=24)
            plt.subplots_adjust(wspace=0.3)

            pdf.savefig(fig, dpi=100)
            plt.close("all")


# %%


def compute_ABM_SEIR_proportions(filenames):
    "Compute the fraction (z) between ABM and SEIR for I_max and R_inf "

    I_max_ABM = []
    R_inf_ABM = []
    for filename in filenames:
        try:
            df = file_loaders.pandas_load_file(filename)
        except EmptyDataError:
            print(f"Empty file error at {filename}")
            continue
        I_max_ABM.append(df["I"].max())
        R_inf_ABM.append(df["R"].iloc[-1])
    I_max_ABM = np.array(I_max_ABM)
    R_inf_ABM = np.array(R_inf_ABM)

    T_max = max(df["time"].max() * 1.2, 300)
    cfg = utils.string_to_dict(filename)
    df_SIR = SIR.integrate(cfg, T_max, dt=0.01, ts=0.1)

    # break out if the SIR model dies out
    if df_SIR["I"].max() < cfg.N_init:
        N = len(I_max_ABM)
        return np.full(N, np.nan), np.full(N, np.nan), cfg

    z_rel_I = I_max_ABM / df_SIR["I"].max()
    z_rel_R = R_inf_ABM / df_SIR["R"].iloc[-1]

    return z_rel_I, z_rel_R, cfg


def get_1D_scan_results(scan_parameter, non_default_parameters):
    "Compute the fraction between ABM and SEIR for all simulations related to the scan_parameter"

    simulation_parameters_1D_scan = simulation_utils.get_simulation_parameters_1D_scan(scan_parameter, non_default_parameters)
    N_simulation_parameters = len(simulation_parameters_1D_scan)
    if N_simulation_parameters == 0:
        return None

    base_dir = Path("Data") / "ABM"

    x = np.zeros(N_simulation_parameters)
    y_I = np.zeros(N_simulation_parameters)
    y_R = np.zeros(N_simulation_parameters)
    sy_I = np.zeros(N_simulation_parameters)
    sy_R = np.zeros(N_simulation_parameters)
    n = np.zeros(N_simulation_parameters)

    # ABM_parameter = simulation_parameters_1D_scan[0]
    for i, ABM_parameter in enumerate(tqdm(simulation_parameters_1D_scan, desc=scan_parameter)):
        filenames = [str(filename) for filename in base_dir.rglob("*.csv") if f"{ABM_parameter}/" in str(filename)]

        z_rel_I, z_rel_R, cfg = compute_ABM_SEIR_proportions(filenames)

        x[i] = cfg[scan_parameter]
        y_I[i] = np.mean(z_rel_I)
        y_R[i] = np.mean(z_rel_R)
        sy_I[i] = utils.SDOM(z_rel_I)
        sy_R[i] = utils.SDOM(z_rel_R)
        n[i] = len(z_rel_I)

    return x, y_I, y_R, sy_I, sy_R, n, cfg


def extract_limits(ylim):
    """ deals with both limits of the form (0, 1) and [(0, 1), (0.5, 1.5)] """
    if isinstance(ylim, (tuple, list)):
        if isinstance(ylim[0], (float, int)):
            ylim0 = ylim1 = ylim
        elif isinstance(ylim[0], (tuple, list)):
            ylim0, ylim1 = ylim
    else:
        ylim0 = ylim1 = (None, None)

    return ylim0, ylim1


def _plot_1D_scan_res(res, scan_parameter, ylim, do_log):

    x, y_I, y_R, sy_I, sy_R, n, cfg = res

    d_par_pretty = utils.get_d_translate()
    title = utils.dict_to_title(cfg, exclude=[scan_parameter, "ID"])
    xlabel = r"$" + d_par_pretty[scan_parameter] + r"$"

    ylim0, ylim1 = extract_limits(ylim)

    # n>1 datapoints
    mask = n > 1

    factor = 0.8
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16 * factor, 9 * factor))  #
    fig.suptitle(title, fontsize=28 * factor)

    ax0.errorbar(
        x[mask], y_I[mask], sy_I[mask], fmt=".", color="black", ecolor="black", elinewidth=1, capsize=10,
    )
    ax0.errorbar(
        x[~mask], y_I[~mask], sy_I[~mask], fmt=".", color="grey", ecolor="grey", elinewidth=1, capsize=10,
    )
    ax0.set(xlabel=xlabel, ylim=ylim0)

    ax1.errorbar(
        x[mask], y_R[mask], sy_R[mask], fmt=".", color="black", ecolor="black", elinewidth=1, capsize=10,
    )
    ax1.errorbar(
        x[~mask], y_R[~mask], sy_R[~mask], fmt=".", color="grey", ecolor="grey", elinewidth=1, capsize=10,
    )
    ax1.set(xlabel=xlabel, ylim=ylim1)

    if do_log:
        ax0.set_xscale("log")
        ax1.set_xscale("log")

    fig.tight_layout()
    fig.subplots_adjust(top=0.8, wspace=0.45)

    return fig, (ax0, ax1)


from pandas.errors import EmptyDataError


def plot_1D_scan(scan_parameter, do_log=False, ylim=None, non_default_parameters=None):

    if not non_default_parameters:
        non_default_parameters = {}

    res = get_1D_scan_results(scan_parameter, non_default_parameters)
    if not res:
        return None

    fig, (ax0, ax1) = _plot_1D_scan_res(res, scan_parameter, ylim, do_log)

    ax0.set(ylabel=r"$I_\mathrm{max}^\mathrm{ABM} \, / \,\, I_\mathrm{max}^\mathrm{SEIR}$")
    ax1.set(ylabel=r"$R_\infty^\mathrm{ABM} \, / \,\, R_\infty^\mathrm{SEIR}$")

    figname_pdf = f"Figures/1D_scan/1D_scan_{scan_parameter}"
    for key, val in non_default_parameters.items():
        figname_pdf += f"_{key}_{val}"
    figname_pdf += f".pdf"

    Path(figname_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figname_pdf, dpi=100)  # bbox_inches='tight', pad_inches=0.3
    plt.close("all")


#%%


def plot_fits(all_fits, force_rerun=False, verbose=False, do_log=False):

    pdf_name = f"Figures/Fits.pdf"
    Path(pdf_name).parent.mkdir(parents=True, exist_ok=True)

    if Path(pdf_name).exists() and not force_rerun:
        print(f"{pdf_name} already exists")
        return None

    with PdfPages(pdf_name) as pdf, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure was using constrained_layout==True")

        leg_loc = {"I": "upper right", "R": "lower right"}
        d_ylabel = {"I": "Infected", "R": "Recovered"}

        for ABM_parameter, fit_objects in tqdm(all_fits.items()):
            # break

            # skip if no fits
            if len(fit_objects) == 0:
                print(f"Skipping {ABM_parameter}")
                continue

            cfg = utils.string_to_dict(ABM_parameter)

            fig, axes = plt.subplots(ncols=2, figsize=(18, 7), constrained_layout=True)
            fig.subplots_adjust(top=0.8)

            for i, fit_object in enumerate(fit_objects.values()):
                # break

                df = file_loaders.pandas_load_file(fit_object.filename)
                t = df["time"].values
                T_max = max(t) * 1.1
                df_fit = fit_object.calc_df_fit(ts=0.1, T_max=T_max)

                lw = 0.8
                for I_or_R, ax in zip(["I", "R"], axes):

                    label = "ABM" if i == 0 else None
                    ax.plot(t, df[I_or_R], "k-", lw=lw, label=label)

                    label_min = "Fit Range" if i == 0 else None
                    ax.axvline(fit_object.t.min(), ymin=0, ymax=0.25, lw=lw, alpha=0.8, label=label_min)
                    ax.axvline(fit_object.t.max(), ymin=0, ymax=0.25, lw=lw, alpha=0.8)

                    label = "Fits" if i == 0 else None
                    ax.plot(
                        df_fit["time"], df_fit[I_or_R], lw=lw, color="green", label=label,
                    )

            all_I_max_MC = []
            all_R_inf_MC = []

            for i, fit_object in enumerate(fit_objects.values()):

                all_I_max_MC.extend(fit_object.I_max_MC)
                all_R_inf_MC.extend(fit_object.R_inf_MC)

            I_median, I_errors = utils.get_central_confidence_intervals(all_I_max_MC)
            s = utils.format_asymmetric_uncertanties(I_median, I_errors, "I")
            axes[0].text(
                -0.15, -0.25, s, horizontalalignment="left", transform=axes[0].transAxes, fontsize=24,
            )

            R_median, R_errors = utils.get_central_confidence_intervals(all_R_inf_MC)
            s = utils.format_asymmetric_uncertanties(R_median, R_errors, "R")
            axes[1].text(
                -0.15, -0.25, s, horizontalalignment="left", transform=axes[1].transAxes, fontsize=24,
            )

            fit_values_deterministic = {
                "lambda_E": cfg.lambda_E,
                "lambda_I": cfg.lambda_I,
                "beta": cfg.beta,
                "tau": 0,
            }

            df_SIR = fit_object.calc_df_fit(fit_values=fit_values_deterministic, ts=0.1, T_max=T_max)

            for I_or_R, ax in zip(["I", "R"], axes):

                ax.plot(
                    df_SIR["time"], df_SIR[I_or_R], lw=lw * 5, color="red", label="SEIR", zorder=0,
                )

                if do_log:
                    ax.set_yscale("log", nonposy="clip")

                ax.set(xlim=(0, None), ylim=(0, None))

                ax.set(xlabel="Time", ylabel=d_ylabel[I_or_R])
                ax.set_rasterized(True)
                ax.set_rasterization_zorder(0)
                ax.yaxis.set_major_formatter(EngFormatter())

                leg = ax.legend(loc=leg_loc[I_or_R])
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(2.0)
                    legobj.set_alpha(1.0)

            title = utils.dict_to_title(cfg, len(fit_objects))
            fig.suptitle(title, fontsize=24)
            plt.subplots_adjust(wspace=0.3)

            pdf.savefig(fig, dpi=100)
            plt.close("all")


#%%


def compute_fit_ABM_proportions(fit_objects):
    "Compute the fraction (z) between the fits and the ABM simulations for I_max and R_inf "

    N = len(fit_objects)

    I_max_fit = np.zeros(N)
    R_inf_fit = np.zeros(N)
    I_max_ABM = np.zeros(N)
    R_inf_ABM = np.zeros(N)

    for i, fit_object in enumerate(fit_objects.values()):
        # break

        df = file_loaders.pandas_load_file(fit_object.filename)
        I_max_ABM[i] = df["I"].max()
        R_inf_ABM[i] = df["R"].iloc[-1]

        t = df["time"].values
        T_max = max(t) * 1.1
        df_fit = fit_object.calc_df_fit(ts=0.1, T_max=T_max)
        I_max_fit[i] = df_fit["I"].max()
        R_inf_fit[i] = df_fit["R"].iloc[-1]

    z_rel_I = I_max_fit / I_max_ABM
    z_rel_R = R_inf_fit / R_inf_ABM

    return z_rel_I, z_rel_R


def get_1D_scan_fit_results(all_fits, scan_parameter, non_default_parameters):
    "Compute the fraction between ABM and SEIR for all simulations related to the scan_parameter"

    simulation_parameters_1D_scan = simulation_utils.get_simulation_parameters_1D_scan(scan_parameter, non_default_parameters)

    selected_fits = {key: val for key, val in all_fits.items() if key in simulation_parameters_1D_scan}

    N_simulation_parameters = len(selected_fits)
    if N_simulation_parameters == 0:
        return None

    N = len(selected_fits)

    x = np.zeros(N)
    y_I = np.zeros(N)
    y_R = np.zeros(N)
    sy_I = np.zeros(N)
    sy_R = np.zeros(N)
    n = np.zeros(N)

    it = tqdm(enumerate(selected_fits.items()), desc=scan_parameter, total=N)
    for i, (ABM_parameter, fit_objects) in it:
        # break

        cfg = utils.string_to_dict(ABM_parameter)
        z_rel_I, z_rel_R = compute_fit_ABM_proportions(fit_objects)

        x[i] = cfg[scan_parameter]
        y_I[i] = np.mean(z_rel_I)
        y_R[i] = np.mean(z_rel_R)
        sy_I[i] = utils.SDOM(z_rel_I)
        sy_R[i] = utils.SDOM(z_rel_R)
        n[i] = len(z_rel_I)

    return x, y_I, y_R, sy_I, sy_R, n, cfg


from pandas.errors import EmptyDataError


def plot_1D_scan_fit_results(all_fits, scan_parameter, do_log=False, ylim=None, non_default_parameters=None):

    if not non_default_parameters:
        non_default_parameters = {}

    res = get_1D_scan_fit_results(all_fits, scan_parameter, non_default_parameters)
    if not res:
        return None

    fig, (ax0, ax1) = _plot_1D_scan_res(res, scan_parameter, ylim, do_log)

    ax0.set(ylabel=r"$I_\mathrm{max}^\mathrm{fit} \, / \,\, I_\mathrm{max}^\mathrm{ABM}$")
    ax1.set(ylabel=r"$R_\infty^\mathrm{fit} \, / \,\, R_\infty^\mathrm{ABM}$")

    figname_pdf = f"Figures/1D_scan_fits/1D_scan_fit_{scan_parameter}"
    for key, val in non_default_parameters.items():
        figname_pdf += f"_{key}_{val}"
    figname_pdf += f".pdf"

    Path(figname_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figname_pdf, dpi=100)  # bbox_inches='tight', pad_inches=0.3
    plt.close("all")


#%%

import h5py
from matplotlib.ticker import EngFormatter


def _load_my_state_and_my_number_of_contacts(filename):
    with h5py.File(filename, "r") as f:
        my_state = f["my_state"][()]
        my_number_of_contacts = f["my_number_of_contacts"][()]
    return my_state, my_number_of_contacts


def _plot_number_of_contacts(filename):

    my_state, my_number_of_contacts = _load_my_state_and_my_number_of_contacts(filename)

    mask_S = my_state[-1] == -1
    mask_R = my_state[-1] == 8
    # mask_EI = np.isin(my_state[-1], np.arange(8))

    fig, ax = plt.subplots()
    ax.hist(my_number_of_contacts, 100, range=(0, 200), histtype="step", label="All")
    ax.hist(my_number_of_contacts[mask_S], 100, range=(0, 200), histtype="step", label="S")
    ax.hist(my_number_of_contacts[mask_R], 100, range=(0, 200), histtype="step", label="R")
    # ax.hist(my_number_of_contacts[mask_EI], 100, range=(0, 200), histtype="step", label="EI")

    ax.legend()
    title = utils.string_to_title(filename)
    ax.yaxis.set_major_formatter(EngFormatter())
    ax.set(xlabel="# of contacts", ylabel="Counts", title=title)
    return fig, ax


def plot_number_of_contacts(network_files, force_rerun=False):

    if len(network_files) == 0:
        return None

    pdf_name = f"Figures/Number_of_contacts.pdf"
    Path(pdf_name).parent.mkdir(parents=True, exist_ok=True)

    if Path(pdf_name).exists() and not force_rerun:
        print(f"{pdf_name} already exists")
        return None

    with PdfPages(pdf_name) as pdf, warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure was using constrained_layout==True")
        for network_filename in tqdm(network_files, desc="Number of contacts"):
            cfg = utils.string_to_dict(str(network_filename))
            if cfg.ID != 0:
                continue
            else:
                fig, ax = _plot_number_of_contacts(network_filename)
                pdf.savefig(fig, dpi=100)
                plt.close("all")
