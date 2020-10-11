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
    from src.utils import utils

    # from src import simulation_utils
    from src import file_loaders
    from src import SIR
except ImportError:
    import utils

    # import simulation_utils
    import file_loaders
    import SIR

from numba import njit
from functools import lru_cache

rc_params.set_rc_params()

# https://carto.com/carto-colors/
d_colors = {
    "blue": "#1D6996",
    "red": "#E41A1C",
    "green": "#0F8554",
    "orange": "#E17C05",
    "purple": "#94346E",
    "light_blue": "#38A6A5",
    "light_green": "#73AF48",
    "yellow": "#EDAD08",
    "grey": "#666666",
}


def compute_df_deterministic(cfg, variable, T_max=100):
    # checks that the curve has flattened out
    while True:
        df_deterministic = SIR.integrate(cfg, T_max, dt=0.01, ts=0.1)
        delta_1_day = df_deterministic[variable].iloc[-2] - df_deterministic[variable].iloc[-1]
        if variable == "R":
            delta_1_day *= -1
        delta_rel = delta_1_day / cfg.N_tot
        if 0 <= delta_rel < 1e-5:
            break
        T_max *= 1.1
    return df_deterministic


# class LatexEngFormatter(EngFormatter):
#     def __init__(self, unit="", places=None, sep=" "):
#         self.has_printed = False
#         super().__init__(unit=unit, places=places, sep=sep, usetex=False, useMathText=False)

#     def __call__(self, x, pos=None):
#         s = super().__call__(x, pos)
#         s = s.split(self.sep)
#         s[-1] = r"\, \mathrm{" + s[-1] + r"}"
#         s = r"$" + self.sep.join(s) + r"$"
#         return s


def plot_single_ABM_simulation(
    cfg, abm_files, add_top_text=True, xlim=(0, None), legend_fontsize=30
):

    filenames = abm_files.cfg_to_filenames(cfg)

    if not isinstance(cfg, utils.DotDict):
        cfg = utils.DotDict(cfg)

    d_ylabel = {"I": "Infected", "R": "Recovered"}
    d_label_loc = {"I": "upper right", "R": "lower right"}

    fig, axes = plt.subplots(ncols=2, figsize=(16, 7))
    fig.subplots_adjust(top=0.8)

    T_max = 0
    lw = 0.3 * 10 / np.sqrt(len(filenames))

    stochastic_noise_I = []
    stochastic_noise_R = []

    # file, i = abm_files[ABM_parameter][0], 0
    for i, filename in enumerate(filenames):
        df = file_loaders.pandas_load_file(filename)
        t = df["time"].values
        label = r"ABM" if i == 0 else None

        axes[0].plot(t, df["I"], lw=lw, c="k", label=label)
        axes[1].plot(t, df["R"], lw=lw, c="k", label=label)

        if t.max() > T_max:
            T_max = t.max()

        stochastic_noise_I.append(df["I"].max())
        stochastic_noise_R.append(df["R"].iloc[-1])

    for variable, ax in zip(["I", "R"], axes):

        df_deterministic = compute_df_deterministic(cfg, variable, T_max=T_max)

        ax.plot(
            df_deterministic["time"],
            df_deterministic[variable],
            lw=lw * 4,
            color=d_colors["red"],
            label="SEIR",
        )
        leg = ax.legend(loc=d_label_loc[variable], fontsize=legend_fontsize)
        for legobj in leg.legendHandles:
            legobj.set_linewidth(lw * 4)

        ax.set(
            xlabel="Time [days]",
            ylim=(0, None),
            ylabel=d_ylabel[variable],
            xlim=xlim,
        )
        # ax.set_xlabel('Time', ha='right')
        # ax.xaxis.set_label_coords(0.91, -0.14)
        ax.yaxis.set_major_formatter(EngFormatter())

    if add_top_text:
        names = [r"I_\mathrm{max}^\mathrm{ABM}", r"R_\infty^\mathrm{ABM}"]
        for name, x, ax in zip(names, [stochastic_noise_I, stochastic_noise_R], axes):
            s = utils.format_relative_uncertainties(x, name)
            ax.text(
                0.5,
                1.05,
                s,
                horizontalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
            )

    title = utils.dict_to_title(cfg, len(filenames))
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(wspace=0.4)

    return fig, ax


def plot_ABM_simulations(abm_files, force_rerun=False):

    # pdf_name = "test.pdf"
    pdf_name = Path(f"Figures/ABM_simulations.pdf")
    utils.make_sure_folder_exist(pdf_name)

    if pdf_name.exists() and not force_rerun:
        print(f"{pdf_name} already exists\n", flush=True)
        return None

    # def iter_folders(self):
    #     for cfg in self.cfgs:
    #         filenames = self.d[cfg.hash]
    #         yield cfg, filenames

    with PdfPages(pdf_name) as pdf:

        # for ABM_parameter in tqdm(abm_files.keys, desc="Plotting individual ABM parameters"):
        for cfg in tqdm(
            abm_files.iter_cfgs(),
            desc="Plotting individual ABM parameters",
            total=len(abm_files.cfgs),
        ):

            # break

            fig, ax = plot_single_ABM_simulation(cfg, abm_files)

            pdf.savefig(fig, dpi=100)
            plt.close("all")


# %%


def compute_ABM_SEIR_proportions(cfg, filenames):
    "Compute the fraction (z) between ABM and SEIR for I_max and R_inf "

    I_max_ABM = []
    R_inf_ABM = []
    for filename in filenames:
        try:
            df = file_loaders.pandas_load_file(filename)
        except EmptyDataError as e:
            print(f"Empty file error at {filename}")
            raise e
            # continue
        I_max_ABM.append(df["I"].max())
        R_inf_ABM.append(df["R"].iloc[-1])
    I_max_ABM = np.array(I_max_ABM)
    R_inf_ABM = np.array(R_inf_ABM)

    T_max = max(df["time"].max() * 1.2, 300)
    df_SIR = SIR.integrate(cfg, T_max, dt=0.01, ts=0.1)

    # break out if the SIR model dies out
    if df_SIR["I"].max() < cfg.N_init:
        N = len(I_max_ABM)
        return np.full(N, np.nan), np.full(N, np.nan)

    z_rel_I = I_max_ABM / df_SIR["I"].max()
    z_rel_R = R_inf_ABM / df_SIR["R"].iloc[-1]

    return z_rel_I, z_rel_R


# def get_1D_scan_cfgs_all_filenames(scan_parameter, non_default_parameters):
#     return cfgs, all_filenames


def get_1D_scan_results(scan_parameter, non_default_parameters):
    "Compute the fraction between ABM and SEIR for all simulations related to the scan_parameter"

    cfgs, all_filenames = utils.get_1D_scan_cfgs_all_filenames(
        scan_parameter, non_default_parameters
    )
    N_cfgs = len(cfgs)
    if N_cfgs <= 1:
        return None

    x = np.zeros(N_cfgs)
    y_I = np.zeros(N_cfgs)
    y_R = np.zeros(N_cfgs)
    sy_I = np.zeros(N_cfgs)
    sy_R = np.zeros(N_cfgs)
    n = np.zeros(N_cfgs)

    # ABM_parameter = simulation_parameters_1D_scan[0]
    it = zip(cfgs, all_filenames)
    for i, (cfg, filenames) in enumerate(tqdm(it, desc=scan_parameter, total=N_cfgs)):

        z_rel_I, z_rel_R = compute_ABM_SEIR_proportions(cfg, filenames)

        x[i] = cfg[scan_parameter]
        y_I[i] = np.mean(z_rel_I)
        sy_I[i] = utils.SDOM(z_rel_I)
        y_R[i] = np.mean(z_rel_R)
        sy_R[i] = utils.SDOM(z_rel_R)

        n[i] = len(z_rel_I)

    if np.isfinite(y_I).sum() <= 1 and np.isfinite(y_R).sum() <= 1:
        return None

    return x, y_I, y_R, sy_I, sy_R, n, cfg


def extract_limits(lim):
    """ deals with both limits of the form (0, 1) and [(0, 1), (0.5, 1.5)] """
    if isinstance(lim, (tuple, list, np.ndarray)):
        if isinstance(lim[0], (float, int)):
            lim0 = lim1 = lim
        elif isinstance(lim[0], (tuple, list, np.ndarray)):
            lim0, lim1 = lim
    else:
        lim0 = lim1 = (None, None)

    return lim0, lim1


def _plot_1D_scan_res(res, scan_parameter, ylim=None, do_log=False, add_title=True, **kwargs):

    # kwargs = {}

    x, y_I, y_R, sy_I, sy_R, n, cfg = res

    d_par_pretty = utils.get_parameter_to_latex()
    title = utils.dict_to_title(cfg, exclude=[scan_parameter, "ID"])

    label_pretty = d_par_pretty[scan_parameter]
    xlabel = r"$" + label_pretty + r"$"
    if scan_parameter == "rho":
        xlabel += r"\, \Huge [km$^{-1}$]"

    ylim0, ylim1 = extract_limits(ylim)

    # n>1 datapoints
    mask = n > 1
    mask_n1 = n == 1
    if scan_parameter == "event_size_max":
        mask_event_size_max = x != 0
        mask = mask & mask_event_size_max
        mask_n1 = mask_n1 & mask_event_size_max

    factor = 0.7

    if "axes" not in kwargs:
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16 * factor, 9 * factor))  #
        if add_title:
            fig.suptitle(title, fontsize=16 * factor)
    else:
        ax0, ax1 = kwargs["axes"]

    errorbar_kwargs = dict(
        fmt=kwargs.get("fmt", "."),
        elinewidth=1,
        capsize=10,
    )

    ax0.errorbar(
        x[mask],
        y_I[mask],
        sy_I[mask],
        **errorbar_kwargs,
        color=kwargs.get("color", "black"),
        ecolor=kwargs.get("color", "black"),
        # label=kwargs.get("label", None),
    )
    ax0.errorbar(
        x[mask_n1],
        y_I[mask_n1],
        sy_I[mask_n1],
        **errorbar_kwargs,
        color="grey",
        ecolor="grey",
    )
    ax0.set(ylim=ylim0)

    ax1.errorbar(
        x[mask],
        y_R[mask],
        sy_R[mask],
        **errorbar_kwargs,
        color=kwargs.get("color", "black"),
        ecolor=kwargs.get("color", "black"),
        label=kwargs.get("label", None),
    )
    ax1.errorbar(
        x[mask_n1],
        y_R[mask_n1],
        sy_R[mask_n1],
        **errorbar_kwargs,
        color="grey",
        ecolor="grey",
    )
    ax1.set(ylim=ylim1)

    if scan_parameter == "event_size_max":
        mask_limit = ~mask_event_size_max
        if any(mask_limit):
            ax0.axhline(y_I[mask_limit], ls="--", color=kwargs.get("color", "black"))
            ax1.axhline(y_R[mask_limit], ls="--", color=kwargs.get("color", "black"))

    ax0.set_xlabel(xlabel, labelpad=kwargs.get("labelpad", -5))
    ax1.set_xlabel(xlabel, labelpad=kwargs.get("labelpad", -5))

    if "label" in kwargs:
        ax1.legend()

    if do_log:
        ax0.set_xscale("log")
        ax1.set_xscale("log")

    if "axes" not in kwargs:
        fig.tight_layout()
        fig.subplots_adjust(top=0.8, wspace=kwargs.get("wspace", 0.55))
        return fig, (ax0, ax1)


from pandas.errors import EmptyDataError


def plot_1D_scan(
    scan_parameter,
    do_log=False,
    ylim=None,
    non_default_parameters=None,
    figname_pdf=None,
    **kwargs,
):

    if non_default_parameters is None:
        non_default_parameters = {}

    res = get_1D_scan_results(scan_parameter, non_default_parameters)
    if res is None:
        return None

    # kwargs = {}
    fig, (ax0, ax1) = _plot_1D_scan_res(res, scan_parameter, ylim, do_log, **kwargs)

    ax0.set(ylabel=r"$I_\mathrm{max}^\mathrm{ABM} \, / \,\, I_\mathrm{max}^\mathrm{SEIR}$")
    ax1.set(ylabel=r"$R_\infty^\mathrm{ABM} \, / \,\, R_\infty^\mathrm{SEIR}$")

    if figname_pdf is None:
        figname_pdf = f"Figures/1D_scan/1D_scan_{scan_parameter}"
        for key, val in non_default_parameters.items():
            figname_pdf += f"_{key}_{val}"
        figname_pdf += f".pdf"

    Path(figname_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figname_pdf, dpi=100)  # bbox_inches='tight', pad_inches=0.3
    plt.close("all")


#%%


def rug_plot(xs, ax, ymax=0.1, **kwargs):
    for x in xs:
        ax.axvline(x, ymax=ymax, **kwargs)


def plot_single_fit(
    cfg,
    fit_objects,
    add_top_text=True,
    xlim=(0, None),
    ylim=(0, None),
    legend_loc=None,
    legend_fontsize=28,
):

    relative_names = [
        r"\frac{I_\mathrm{max}^\mathrm{fit}} {I_\mathrm{max}^\mathrm{ABM}}",
        r"\frac{R_\infty^\mathrm{fit}} {R_\infty^\mathrm{fit}}",
    ]

    # fit_objects = all_fits[ABM_parameter]

    d_ylabel = {"I": r"Infected", "R": r"Recovered"}

    if legend_loc is None:
        legend_loc = {"I": "upper right", "R": "lower right"}

    # cfg = utils.string_to_dict(ABM_parameter)

    fig, axes = plt.subplots(ncols=2, figsize=(16, 7))
    fig.subplots_adjust(top=0.8)

    for i, fit_object in enumerate(fit_objects.values()):
        # break

        df = file_loaders.pandas_load_file(fit_object.filename)
        t = df["time"].values
        T_max = max(t) * 1.1
        df_fit = fit_object.calc_df_fit(ts=0.1, T_max=T_max)

        lw = 0.9
        for I_or_R, ax in zip(["I", "R"], axes):

            label = "ABM" if i == 0 else None
            ax.plot(t, df[I_or_R], "k-", lw=lw, label=label)

            label = "Fits" if i == 0 else None
            ax.plot(
                df_fit["time"],
                df_fit[I_or_R],
                lw=lw,
                color=d_colors["green"],
                label=label,
            )

            if i == 0:
                axvline_kwargs = dict(lw=lw, color=d_colors["blue"], alpha=0.4)
                tmp = df.query("@fit_object.t.min() <= time <= @fit_object.t.max()")
                ax.fill_between(tmp["time"], tmp[I_or_R], **axvline_kwargs)

                vertical_lines = tmp["time"].iloc[0], tmp["time"].iloc[-1]
                line_kwargs = dict(ymax=0.6, color=d_colors["blue"], lw=2 * lw)
                ax.axvline(vertical_lines[0], **line_kwargs)
                ax.axvline(vertical_lines[1], **line_kwargs)

                ax.text(
                    vertical_lines[0] * 0.65,
                    0.3 * ax.get_ylim()[1],
                    "Fit Range",
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=34,
                    rotation=90,
                    color=d_colors["blue"],
                )

            # x_rug = [fit_object.t.min(), fit_object.t.max()]
            # rug_plot(x_rug, ax, ymin=-0.01, ymax=0.03, color="k", lw=lw)

    if add_top_text:

        # calculate monte carlo simulated fit results
        # all_I_max_MC = []
        # all_R_inf_MC = []
        all_I_max_fit = []
        all_R_inf_fit = []
        for i, fit_object in enumerate(fit_objects.values()):
            # all_I_max_MC.extend(fit_object.I_max_MC)
            # all_R_inf_MC.extend(fit_object.R_inf_MC)
            all_I_max_fit.append(fit_object.I_max_fit)
            all_R_inf_fit.append(fit_object.R_inf_MC)
        d_fits = {"I": all_I_max_fit, "R": all_R_inf_fit}

        names_fit = {}
        names_fit["I"] = r"I_\mathrm{max}^\mathrm{fit}"
        names_fit["R"] = r"R_\infty^\mathrm{fit}"

        for I_or_R, ax in zip(["I", "R"], axes):

            s = utils.format_relative_uncertainties(d_fits[I_or_R], names_fit[I_or_R])

            ax.text(
                0.1,
                1.05,
                s,
                horizontalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
            )

        # calculate fraction between fit and ABM simulation
        z = defaultdict(list)
        for fit_object in fit_objects.values():
            z["I"].append(fit_object.I_max_fit / fit_object.I_max_ABM)
            z["R"].append(fit_object.R_inf_fit / fit_object.R_inf_ABM)

        for I_or_R, name, ax in zip(["I", "R"], relative_names, axes):
            mu, std = np.mean(z[I_or_R]), utils.SDOM(z[I_or_R])
            n_digits = int(np.log10(utils.round_to_uncertainty(mu, std)[0])) + 1
            s_mu = utils.human_format_scientific(mu, digits=n_digits)
            s = r"$ " + f"{name} = {s_mu[0]}" + r"\pm " + f"{std:.{n_digits}f}" + r"$"
            ax.text(0.55, 1.05, s, transform=ax.transAxes, fontsize=12)

    fit_values_deterministic = {
        "lambda_E": cfg.lambda_E,
        "lambda_I": cfg.lambda_I,
        "beta": cfg.beta,
        "tau": 0,
    }

    df_SIR = fit_object.calc_df_fit(fit_values=fit_values_deterministic, ts=0.1, T_max=T_max)

    for I_or_R, ax in zip(["I", "R"], axes):

        ax.plot(
            df_SIR["time"],
            df_SIR[I_or_R],
            lw=lw * 4,
            color=d_colors["red"],
            label="SEIR",
            zorder=0,
        )

        ax.set(xlim=xlim, ylim=ylim)
        ax.set(xlabel="Time [days]", ylabel=d_ylabel[I_or_R])
        # if add_top_text:
        #     ax.xaxis.set_label_coords(0.91, -0.14)
        ax.yaxis.set_major_formatter(EngFormatter())

    leg = axes[0].legend(loc=legend_loc["I"], fontsize=legend_fontsize)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
        legobj.set_alpha(1.0)

    title = utils.dict_to_title(cfg, len(fit_objects))
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(wspace=0.4)

    return fig, ax


def plot_fits(all_fits, force_rerun=False, verbose=False):

    pdf_name = f"Figures/Fits.pdf"
    Path(pdf_name).parent.mkdir(parents=True, exist_ok=True)

    if Path(pdf_name).exists() and not force_rerun:
        print(f"{pdf_name} already exists")
        return None

    with PdfPages(pdf_name) as pdf:

        for hash_, fit_objects in tqdm(all_fits.items(), desc="Plotting all fits"):
            # break

            # skip if no fits
            if len(fit_objects) == 0:
                if verbose:
                    print(f"Skipping {hash_}")
                continue

            cfg = utils.hash_to_cfg(hash_)
            fig, ax = plot_single_fit(cfg, fit_objects)

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

    cfgs, _ = utils.get_1D_scan_cfgs_all_filenames(scan_parameter, non_default_parameters)

    cfg_hashes = set([cfg.hash for cfg in cfgs])
    selected_fits = {hash_: val for hash_, val in all_fits.items() if hash_ in cfg_hashes}

    N_cfgs = len(selected_fits)
    if N_cfgs <= 1:
        return None

    N = len(selected_fits)

    x = np.zeros(N)
    y_I = np.zeros(N)
    y_R = np.zeros(N)
    sy_I = np.zeros(N)
    sy_R = np.zeros(N)
    n = np.zeros(N)

    it = tqdm(enumerate(selected_fits.items()), desc=scan_parameter, total=N)
    for i, (hash_, fit_objects) in it:
        # break

        cfg = utils.hash_to_cfg(hash_)
        # cfg = utils.string_to_dict(ABM_parameter)
        z_rel_I, z_rel_R = compute_fit_ABM_proportions(fit_objects)

        x[i] = cfg[scan_parameter]

        if len(z_rel_I) > 0:
            y_I[i] = np.mean(z_rel_I)
            sy_I[i] = utils.SDOM(z_rel_I)
        else:
            y_I[i] = sy_I[i] = np.nan

        if len(z_rel_R) > 0:
            y_R[i] = np.mean(z_rel_R)
            sy_R[i] = utils.SDOM(z_rel_R)
        else:
            y_R[i] = sy_R[i] = np.nan

        n[i] = len(z_rel_I)

    return x[n > 0], y_I[n > 0], y_R[n > 0], sy_I[n > 0], sy_R[n > 0], n[n > 0], cfg


from pandas.errors import EmptyDataError


def plot_1D_scan_fit_results(
    all_fits,
    scan_parameter,
    do_log=False,
    ylim=None,
    non_default_parameters=None,
    figname_pdf=None,
):

    if not non_default_parameters:
        non_default_parameters = {}

    res = get_1D_scan_fit_results(all_fits, scan_parameter, non_default_parameters)
    if not res:
        return None

    fig, (ax0, ax1) = _plot_1D_scan_res(res, scan_parameter, ylim, do_log)

    ax0.set(ylabel=r"$I_\mathrm{max}^\mathrm{fit} \, / \,\, I_\mathrm{max}^\mathrm{ABM}$")
    ax1.set(ylabel=r"$R_\infty^\mathrm{fit} \, / \,\, R_\infty^\mathrm{ABM}$")

    if figname_pdf is None:
        figname_pdf = f"Figures/1D_scan_fits/1D_scan_fit_{scan_parameter}"
        for key, val in non_default_parameters.items():
            figname_pdf += f"_{key}_{val}"
        figname_pdf += f".pdf"

    Path(figname_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figname_pdf, dpi=100)  # bbox_inches='tight', pad_inches=0.3
    plt.close("all")


#%%

import h5py


def _load_my_state_and_my_number_of_contacts(filename):
    with h5py.File(filename, "r") as f:
        my_state = f["my_state"][()]
        my_number_of_contacts = f["my_number_of_contacts"][()]
    return my_state, my_number_of_contacts


def plot_single_number_of_contacts(
    filename,
    make_fraction_subplot=True,
    figsize=None,
    xlim=None,
    title=None,
    add_legend=True,
    loc="best",
    xlabel=None,
    ylabel=None,
    fontsize=30,
    labelsize=None,
    add_average_arrows=False,
    title_pad=20,
):

    if xlabel is None:
        xlabel = "Number of contacts"

    if ylabel is None:
        ylabel = "Counts"

    my_state, my_number_of_contacts = _load_my_state_and_my_number_of_contacts(filename)

    mask_S = my_state[-1] == -1
    mask_R = my_state[-1] == 8

    if xlim is None:
        x_min = np.percentile(my_number_of_contacts, 0.01)
        x_max = np.percentile(my_number_of_contacts, 99)
    else:
        x_min, x_max = xlim
    x_range = (x_min, x_max)
    N_bins = int(x_max - x_min)

    kwargs = {"bins": N_bins, "range": x_range, "histtype": "step"}

    if make_fraction_subplot:
        fig, (ax1, ax2) = plt.subplots(
            figsize=figsize, nrows=2, sharex=True, gridspec_kw={"height_ratios": [2.5, 1]}
        )
    else:
        fig, ax1 = plt.subplots(figsize=figsize)

    H_all = ax1.hist(my_number_of_contacts, label="All", color=d_colors["blue"], **kwargs)
    H_S = ax1.hist(
        my_number_of_contacts[mask_S], label="Susceptable", color=d_colors["red"], **kwargs
    )
    H_R = ax1.hist(
        my_number_of_contacts[mask_R], label="Recovered", color=d_colors["green"], **kwargs
    )

    x = 0.5 * (H_all[1][:-1] + H_all[1][1:])
    frac_S = H_S[0] / H_all[0]
    s_frac_S = np.sqrt(frac_S * (1 - frac_S) / H_all[0])
    frac_R = H_R[0] / H_all[0]
    s_frac_R = np.sqrt(frac_R * (1 - frac_R) / H_all[0])

    if make_fraction_subplot:
        kwargs_errorbar = dict(fmt=".", elinewidth=1.5, capsize=4, capthick=1.5)
        ax2.errorbar(x, frac_S, s_frac_S, color=d_colors["red"], **kwargs_errorbar)
        ax2.errorbar(x, frac_R, s_frac_R, color=d_colors["green"], **kwargs_errorbar)

    if add_legend:
        ax1.legend(loc=loc)
    ax1.yaxis.set_major_formatter(EngFormatter())
    ax1.set(xlim=x_range)
    ax1.set_ylabel(ylabel, fontsize=fontsize)
    ax1.set_title(title, pad=title_pad, fontsize=fontsize)

    if make_fraction_subplot:
        ax2.set(ylim=(0, 1), ylabel=r"Fraction")
        ax2.set(xlabel=xlabel)
    else:
        ax1.set_xlabel(xlabel, fontsize=fontsize)

    if labelsize:
        ax1.tick_params(axis="both", labelsize=labelsize)

    if add_average_arrows:
        ymax = ax1.get_ylim()[1]
        mean_all = np.mean(my_number_of_contacts)
        mean_S = np.mean(my_number_of_contacts[mask_S])
        mean_R = np.mean(my_number_of_contacts[mask_R])

        arrowprops = dict(ec="white", width=6, headwidth=20, headlength=15)
        ax1.annotate(
            "",
            xy=(mean_all, ymax * 0.01),
            xytext=(mean_all, ymax * 0.2),
            arrowprops=dict(**arrowprops, fc=d_colors["blue"]),
        )
        ax1.annotate(
            "",
            xy=(mean_S, ymax * 0.01),
            xytext=(mean_S, ymax * 0.2),
            arrowprops=dict(**arrowprops, fc=d_colors["red"]),
        )
        ax1.annotate(
            "",
            xy=(mean_R, ymax * 0.01),
            xytext=(mean_R, ymax * 0.2),
            arrowprops=dict(**arrowprops, fc=d_colors["green"]),
        )

    return fig, ax1


def plot_number_of_contacts(network_files, force_rerun=False):

    if len(network_files) == 0:
        return None

    pdf_name = f"Figures/Number_of_contacts.pdf"
    Path(pdf_name).parent.mkdir(parents=True, exist_ok=True)

    if Path(pdf_name).exists() and not force_rerun:
        print(f"{pdf_name} already exists")
        return None

    with PdfPages(pdf_name) as pdf:
        for network_filename in tqdm(
            network_files.iter_all_files(),
            desc="Number of contacts",
            total=len(network_files),
        ):
            # cfg = utils.string_to_dict(str(network_filename))
            if "_ID__0" in network_filename:
                fig, ax = plot_single_number_of_contacts(network_filename)
                pdf.savefig(fig, dpi=100)
                plt.close("all")


#%%


# def _load_my_state_and_coordinates(filename):
#     with h5py.File(filename, "r") as f:
#         my_state = f["my_state"][()]
#         coordinates = f["coordinates"][()]
#     return my_state, coordinates


# @njit
# def hist2d_numba(data_2D, bins, ranges):
#     H = np.zeros((bins[0], bins[1]), dtype=np.uint64)
#     delta = 1 / ((ranges[:, 1] - ranges[:, 0]) / bins)
#     for t in range(data_2D.shape[0]):
#         i = (data_2D[t, 0] - ranges[0, 0]) * delta[0]
#         j = (data_2D[t, 1] - ranges[1, 0]) * delta[1]
#         if 0 <= i < bins[0] and 0 <= j < bins[1]:
#             H[int(i), int(j)] += 1
#     return H


# @njit
# def get_ranges(x):
#     return np.array(([x[:, 0].min(), x[:, 0].max()], [x[:, 1].min(), x[:, 1].max()]))


# def histogram2d(data_2D, bins=None, ranges=None):
#     if bins is None:
#         print("No binning provided, using (100, 100) as default")
#         bins = np.array((100, 100))
#     if isinstance(bins, int):
#         bins = np.array([bins, bins])
#     elif isinstance(bins, list) or isinstance(bins, tuple):
#         bins = np.array(bins, dtype=int)
#     if ranges is None:
#         ranges = get_ranges(data_2D)
#         ranges[:, 0] *= 0.99
#         ranges[:, 1] *= 1.01
#     return hist2d_numba(data_2D, bins=bins, ranges=ranges)


# def _get_N_bins_xy(coordinates):

#     lon_min = coordinates[:, 0].min()
#     lon_max = coordinates[:, 0].max()
#     lon_mid = np.mean([lon_min, lon_max])

#     lat_min = coordinates[:, 1].min()
#     lat_max = coordinates[:, 1].max()
#     lat_mid = np.mean([lat_min, lat_max])

#     N_bins_x = int(utils.haversine(lon_min, lat_mid, lon_max, lat_mid)) + 1
#     N_bins_y = int(utils.haversine(lon_mid, lat_min, lon_mid, lat_max)) + 1

#     return N_bins_x, N_bins_y


# if False:

#     filename = "Data/network/N_tot__5800000__N_init__100__rho__0.01__epsilon_rho__0.04__mu__40.0__sigma_mu__0.0__beta__0.01__sigma_beta__0.0__lambda_E__1.0__lambda_I__1.0__algo__2__ID__0.hdf5"

#     day = -1
#     grid_km = 2

#     my_state, coordinates = _load_my_state_and_coordinates(filename)

#     N_bins_x, N_bins_y = _get_N_bins_xy(coordinates)
#     ranges = get_ranges(coordinates)

#     my_state_day = my_state[day]

#     kwargs = {"bins": (N_bins_x / grid_km, N_bins_y / grid_km), "ranges": ranges}
#     counts_1d_all = histogram2d(coordinates, **kwargs)
#     counts_1d_R = histogram2d(coordinates[my_state_day == 8], **kwargs)

#     H = counts_1d_R / counts_1d_all
#     H_1d = H[counts_1d_all > 0]

#     from matplotlib.colors import LogNorm

#     fig, ax = plt.subplots()
#     norm = LogNorm(vmin=0.01, vmax=1)
#     im = ax.imshow(H.T, interpolation="none", origin="lower", cmap="viridis")  # norm=norm
#     fig.colorbar(im, ax=ax, extend="max")

#     fig2, ax2 = plt.subplots()
#     ax2.hist(H[counts_1d_all > 0], 100)
#     # ax2.set_yscale("log")

#     fig3, ax3 = plt.subplots()
#     ax3.hist(counts_1d_all.flatten(), 100)
#     # ax3.set_yscale("log")


#     # %%
