import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt

from contexttimer import Timer

from tqdm import tqdm

from src.utils      import utils
from src.simulation import simulation
from src.utils      import file_loaders

from src.analysis.helpers  import *
from src.analysis.plotters import *


# This runs simulations with specified percentage effects of the seasonality model
params, start_date = utils.load_params("cfg/analyzers/initializer.yaml")

if utils.is_local_computer():
    f = 0.01
    num_cores_max = 1
    N_runs = 1
else :
    f = 0.5
    num_cores_max = 5
    N_runs = 1


if num_cores_max == 1 :
    verbose = True
else :
    verbose = False


# Scale the population
params["N_tot"]  = int(params["N_tot"]  * f)
params["R_init"] = int(params["R_init"] * f)

# Use "event_size_max"  as trick to generate several runs on the same network
params['event_size_max'] = np.arange(9)


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


print('Plotting the individual ABM simulations. Please wait', flush=True)
for n, network_filename in tqdm(enumerate(data.iter_network_files()), total=len(data.networks)) :

    # Prepare figure
    fig = plt.figure(figsize=(12, 12))

    # Get the distribution in the restricted network
    state, connections, connection_type, connection_status = file_loaders.load_data_from_network_file(network_filename, ['my_state', 'my_connections', 'my_connection_type', 'my_connection_status'])

    # Infected agents
    infected_agents = [agent for agent, s in enumerate(state[-1]) if s >= 0]

    # .. and their neighbours
    neighbours = [agent for infected_agent in infected_agents for agent in connections[infected_agent]]

    # Combined
    agents = np.unique(infected_agents + neighbours)

    id_to_index = pd.Series(np.arange(len(agents)), index=agents)
    index_to_id = pd.Series(agents, index=np.arange(len(agents)))

    # Determine the number of agents to plot
    n_agents = len(agents)

    # Plot the network
    G = nx.Graph()

    for k, agent in enumerate(infected_agents) :
        for l, neighbour in enumerate(connections[agent]) :
            i = id_to_index[agent]
            j = id_to_index[neighbour]

            if k < 10 :

                if connection_status[agent][l] == 1 :
                    color = 'g'
                else :
                    color = 'k'

                if connection_type[agent][l] == 0 :
                    weight = 3
                else :
                    weight = 1

                G.add_edge(i, j, color=color, weight=weight)

    color_map = []
    for node in G:
        if index_to_id[node] in infected_agents:
            color_map.append('red')
        else:
            color_map.append('blue')

    pos = nx.spring_layout(G, k=0.05, iterations=20)

    colors  = [    G[u][v]['color']  for u,v in G.edges]
    weights = [2 * G[u][v]['weight'] for u,v in G.edges]

    nx.draw(G, pos, node_color=color_map, edge_color=colors, width=weights, node_size=20)

    plt.title(f'N = {len(infected_agents)}')

    plt.savefig(fig_name.replace('.png', f'_{n}.png'))
