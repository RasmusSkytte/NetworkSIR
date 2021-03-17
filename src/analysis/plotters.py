import numpy as np

import scipy
import datetime

import h5py

import matplotlib        as mpl
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates

from src.utils import utils
from src.utils import file_loaders

def plot_simulation_cases_and_variant_fraction(total_tests, f, t_day, t_week, axes) :

    # Create the plots
    tmp_handles_0 = axes[0].plot(t_day,  total_tests, lw=4, c='k')[0]
    tmp_handles_1 = axes[1].plot(t_week, f,           lw=4, c='k')[0]

    return [tmp_handles_0, tmp_handles_1]

def plot_simulation_category(tests_by_category, t, axes) :

    tmp_handles = []
    # Create the plots
    for i in range(np.size(tests_by_category, 1)) :
        tmp_handle = axes[i].plot(t, tests_by_category[:, i], lw=4, c=plt.cm.tab10(i))[0]
        tmp_handles.append(tmp_handle)

    return tmp_handles

def plot_simulation_growth_rates(tests_by_variant, t, axes) :

    # Add the total tests also
    tests_by_variant = np.concatenate((np.sum(tests_by_variant, axis=1).reshape(-1, 1), tests_by_variant), axis=1)

    t += datetime.timedelta(days=0.5)
    tmp_handles = []

    for i in range(tests_by_variant.shape[1]) :

        y = tests_by_variant[:, i]

        if np.all(y == 0) :
            continue

        window_size = 7 # days
        t_w = np.arange(window_size)
        R_w = []

        t_max = len(y)-window_size
        if np.any(y == 0) :
            t_max = min(t_max, np.where(y > 0)[0][-1])

        for j in range(t_max) :
            y_w = y[j:(j+window_size)]
            res, _ = scipy.optimize.curve_fit(lambda t, a, r: a * np.exp(r * t), t_w, y_w, p0=(np.max(y_w), 0))
            R_w.append(1 + 4.7 * res[1])

        t_w = t[window_size:(window_size+t_max)]
        tmp_handles.append(axes[i].plot(t_w, R_w, lw=4, c='k')[0])

    return tmp_handles




def plot_contact_distribution(filename):

    my = _load_my_from_file(filename)

    cfg = file_loaders.filename_to_cfg(filename)
    N_tot = cfg.network.N_tot
    factor = 1 / N_tot

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

    H_all = ax1.hist(
        my_number_of_contacts,
        weights=factor * np.ones_like(my_number_of_contacts),
        label="All",
        color=d_colors["blue"],
        **kwargs,
    )
    H_S = ax1.hist(
        my_number_of_contacts[mask_S],
        weights=factor * np.ones_like(my_number_of_contacts[mask_S]),
        label="Susceptable",
        color=d_colors["red"],
        **kwargs,
    )
    H_R = ax1.hist(
        my_number_of_contacts[mask_R],
        weights=factor * np.ones_like(my_number_of_contacts[mask_R]),
        label="Recovered",
        color=d_colors["green"],
        **kwargs,
    )

    x = 0.5 * (H_all[1][:-1] + H_all[1][1:])
    frac_S = H_S[0] / H_all[0]
    s_frac_S = np.sqrt(frac_S * (1 - frac_S) / (H_all[0] / factor))
    frac_R = H_R[0] / H_all[0]
    s_frac_R = np.sqrt(frac_R * (1 - frac_R) / (H_all[0] / factor))

    if make_fraction_subplot:
        kwargs_errorbar = dict(fmt=".", elinewidth=1.5, capsize=4, capthick=1.5)
        ax2.errorbar(x, frac_S, s_frac_S, color=d_colors["red"], **kwargs_errorbar)
        ax2.errorbar(x, frac_R, s_frac_R, color=d_colors["green"], **kwargs_errorbar)

    if add_legend:
        ax1.legend(loc=loc)
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1))
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



def set_date_xaxis(ax, start_date, end_date) :

    months     = mdates.MonthLocator()
    months_fmt = mdates.DateFormatter('%b')

    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(months_fmt)
    ax.set_xlim([start_date, end_date])


def set_rc_params(dpi=300):
    plt.rcParams['figure.figsize'] = (16, 10)
    plt.rcParams['figure.dpi'] = dpi

    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.size'] = 8

    plt.rcParams['axes.titlepad'] = 20
    plt.rcParams['axes.titlesize'] = 24

    plt.rcParams['axes.labelsize'] = 20

    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    mpl.rc('axes', edgecolor='k', linewidth=2)

set_rc_params()


def _load_data_from_network_file(filename, variables, cfg=None) :

    if cfg is None :
        cfg = utils.read_cfg_from_hdf5_file(filename)


    with h5py.File(filename, "r") as f:

        if not isinstance(variables, list) :
            return f[variables][()]

        else :
            out = []

            for variable in variables :
                out.append(f[variable][()])

            return out