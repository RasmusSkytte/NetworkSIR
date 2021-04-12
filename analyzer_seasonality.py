from contexttimer import Timer

from tqdm import tqdm

import datetime

from src.utils      import utils
from src.simulation import simulation
from src.utils      import file_loaders

from src.analysis.helpers  import *
from src.analysis.plotters import *


# This runs simulations with specified percentage effects of the seasonality model
params, start_date = utils.load_params("cfg/analyzers/seasonality.yaml")

if utils.is_local_computer():
    f = 0.2
    n_steps = 1
    num_cores_max = 3
    N_runs = 7
else :
    f = 0.1
    n_steps = 1
    num_cores_max = 15
    N_runs = 3


if num_cores_max == 1 :
    verbose = True
else :
    verbose = False


# Scale the population
params["N_tot"]  = int(params["N_tot"]  * f)
params["N_init"] = int(params["N_init"] * f)
params["R_init"] = int(params["R_init"] * f)


N_files_total = 0
if __name__ == "__main__":
    with Timer() as t:

        N_files_total +=  simulation.run_simulations(params, N_runs=N_runs, num_cores_max=num_cores_max, verbose=verbose)

    print(f"\n{N_files_total:,} files were generated, total duration {utils.format_time(t.elapsed)}")
    print("Finished simulating!")


# Load the simulations
subset = {'seasonal_list_name' : 'reference', 'Intervention_vaccination_schedule_name' : 'None'}
data = file_loaders.ABM_simulations(subset=subset)

if len(data.filenames) == 0 :
    raise ValueError(f'No files loaded with subset: {subset}')


# Get a cfg out
cfg = data.cfgs[0]
end_date   = start_date + datetime.timedelta(days=cfg.day_max)

t_day, t_week = parse_time_ranges(start_date, end_date)

logK, logK_sigma, beta, t_index      = load_covid_index()
fraction, fraction_sigma, t_fraction = load_b117_fraction()

# Prepare output file
fig_name = 'Figures/seasonality.png'
file_loaders.make_sure_folder_exist(fig_name)


# Prepare figure
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 12))
axes = axes.flatten()

print('Plotting the individual ABM simulations. Please wait', flush=True)
for cfg in tqdm(data.iter_cfgs(), total=len(data.cfgs)) :

    for filename in data.cfg_to_filenames(cfg) :

        # Load
        total_tests, f, _, _, _, _ = load_from_file(filename, start_date)

        # Create the plots
        I = np.argmax(cfg.seasonal_strength == np.array(params['seasonal_strength']))
        plot_simulation_cases_and_variant_fraction(total_tests, f, t_day, t_week, axes, color=plt.cm.tab10(I))




# Plot the covid index
m  = np.exp(logK) * (ref_tests ** beta)
ub = np.exp(logK + logK_sigma) * (ref_tests ** beta) - m
lb = m - np.exp(logK - logK_sigma) * (ref_tests ** beta)
s  = np.stack((lb.to_numpy(), ub.to_numpy()))

axes[0].errorbar(t_index, m, yerr=s, fmt='o', lw=2)



# Plot the WGS B.1.1.7 fraction
axes[1].errorbar(t_fraction, fraction, yerr=fraction_sigma, fmt='s', lw=2)


# Get restriction_thresholds from a cfg
restriction_thresholds = data.cfgs[0].restriction_thresholds

axes[0].set_ylim(0, 20000)
axes[0].set_ylabel('Daglige positive')


axes[1].set_ylim(0, 1)
axes[1].set_ylabel('frac. B.1.1.7')


fig.canvas.draw()

ylims = [ax.get_ylim() for ax in axes]

# Get the transition dates
restiction_days = restriction_thresholds[1::2]

for day in restiction_days :
    restiction_date = start_date + datetime.timedelta(days=day)
    for ax, lim in zip(axes, ylims) :
        ax.plot([restiction_date, restiction_date], lim, '--', color='k', linewidth=2)


set_date_xaxis(axes[1], start_date, end_date)



for ax, lim in zip(axes, ylims) :
    ax.set_ylim(lim[0], lim[1])



plt.savefig(fig_name)
