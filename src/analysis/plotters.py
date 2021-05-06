import numpy as np

import scipy
import datetime

import h5py

import matplotlib        as mpl
import matplotlib.pyplot as plt
import matplotlib.dates  as mdates

from src.utils import utils

def plot_simulation_cases(total_tests, t_day, axes, color='k', label='') :

    # Create the plots
    return axes.plot(t_day[:len(total_tests)], total_tests, lw=4, c=color, label=label)[0]


def plot_variant_fraction(f, t_week, axes, color='k', label='') :

    # Create the plots
    return axes.plot(t_week[:len(f)], f, lw=4, c=color, label=label)[0]


def plot_simulation_cases_and_variant_fraction(total_tests, f, t_day, t_week, axes, color='k') :

    # Create the plots
    handle_0 = plot_simulation_cases(total_tests, t_day, axes[0], color=color)
    handle_1 = plot_variant_fraction(f, t_week, axes[1], color=color)

    return [handle_0, handle_1]

def plot_simulation_category(tests_by_category, t, axes, linestyle = '-') :

    tmp_handles = []
    # Create the plots
    for i in range(np.size(tests_by_category, 1)) :
        tmp_handle = axes[i].plot(t[:np.size(tests_by_category, 0)], tests_by_category[:, i], lw=4, linestyle=linestyle, c=plt.cm.tab10(i % 10))[0]
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


def set_date_xaxis(ax, start_date, end_date, interval=1) :

    months     = mdates.MonthLocator(interval=interval)
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