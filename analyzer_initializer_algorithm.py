from contexttimer import Timer

from tqdm import tqdm

import datetime

from src.utils      import utils
from src.simulation import simulation
from src.utils      import file_loaders

from src.analysis.helpers  import *
from src.analysis.plotters import *


# This runs simulations with specified percentage effects of the seasonality model
params, start_date = utils.load_params("cfg/analyzers/initializer.yaml")

if utils.is_local_computer():
    f = 0.01
    n_steps = 1
    num_cores_max = 1
    N_runs = 5
else :
    f = 0.5
    n_steps = 3
    num_cores_max = 5
    N_runs = 1


if num_cores_max == 1 :
    verbose = True
else :
    verbose = False


# Scale the population
params["N_tot"]  = int(params["N_tot"]  * f)
params["R_init"] = int(params["R_init"] * f)


N_files_total = 0
if __name__ == "__main__":
    with Timer() as t:

        N_files_total +=  simulation.run_simulations(params, N_runs=N_runs, num_cores_max=num_cores_max, verbose=verbose)

    print(f"\n{N_files_total:,} files were generated, total duration {utils.format_time(t.elapsed)}")
    print("Finished simulating!")


# Load the simulations
subset = {'N_init' : 1, 'Intervention_vaccination_schedule_name' : 'None'}
data = file_loaders.ABM_simulations(subset=subset)

if len(data.filenames) == 0 :
    raise ValueError(f'No files loaded with subset: {subset}')


# Prepare output file
fig_name = 'Figures/outbreak_network.png'
file_loaders.make_sure_folder_exist(fig_name)


# Prepare figure
fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 12))
axes = axes.flatten()

print('Plotting the individual ABM simulations. Please wait', flush=True)
for network_filename in tqdm(data.iter_network_files(), total=len(data.networks)) :

    # Get the distribution in the restricted network
    state, connections, connection_type, connection_status = file_loaders.load_data_from_network_file(network_filename, ['my_state', 'my_connections', 'my_connection_type', 'my_connection_status'])

    # Infected agents
    infected_agents = [agent for agent, s in enumerate(state[-1]) if s >= 0]
    #infected_agents = [s for s in list(state) if s >= 0]

    print(infected_agents)
