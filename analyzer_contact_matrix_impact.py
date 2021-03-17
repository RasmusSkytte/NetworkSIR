import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import h5py

from src.utils      import utils
from src.utils      import file_loaders
from src.analysis   import plotters
from src.simulation import simulation

from tqdm import tqdm


# This script generates a network based on the defualt network parameters and those specified in this file
# After network generation, the script computes the contact distribution for each label
# Then, matrix restrictions are applied to each label and the contact distributions are recomputed



# 1) Define network to generate
f = 0.01
verbose = False


Intervention_contact_matrices_name = [['basis'], ['ned2021jan']]


# 2) Generate networks
simulation.run_simulations({'N_tot' : int(5_800_000 * f), 'day_max' : 0, 'initial_infection_distribution' : 'random', 'Intervention_contact_matrices_name' : Intervention_contact_matrices_name}, num_cores_max=1, verbose=verbose)

# 3) Plot the contact distribution
data = file_loaders.ABM_simulations(subset={'day_max' : 0})


def contact_counter(connection_type, connection_status, types=[0, 1, 2]) :

    contact_counts = np.zeros(len(connection_type))

    for i, (agent_contacts, agent_contact_status) in enumerate(zip(connection_type, connection_status)) :
        contact_counts[i] = np.sum(np.logical_and(np.isin(agent_contacts, types), np.array(agent_contact_status) == True))

    return contact_counts


plotters.set_rc_params()

for network_filename, cfg in tqdm(
    zip(data.iter_network_files(), data.iter_cfgs()),
    desc='Analyzing number of contacts',
    total=len(data)) :

    # Prepare figure
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    axes = axes.flatten()

    # Get distribution in the initial network
    hash_initial_network = utils.cfg_to_hash(cfg.network, exclude_ID=False)
    initial_network_filename = f'Initialized_networks/{hash_initial_network}.hdf5'

    with h5py.File(initial_network_filename, 'r') as f :
        cfg.pop('hash')
        my_hdf5ready = file_loaders.load_jitclass_to_dict(f['my'])
        my = file_loaders.load_My_from_dict(my_hdf5ready, cfg.deepcopy())

    # Get the distribution in the restricted network
    connection_type, connection_status = plotters._load_data_from_network_file(network_filename, ['my_connection_type', 'my_connection_status'], cfg=cfg)

    filters = [[0,1,2], [0], [1], [2]]
    titles  = ['All', 'Home', 'Work', 'Other']

    for i, (types, title) in enumerate(zip(filters, titles)) :

        number_of_contacts_reference  = contact_counter(my.connection_type, my.connection_status, types=types)
        number_of_contacts_restricted = contact_counter(   connection_type,    connection_status, types=types)

        x_min = np.min(number_of_contacts_reference) - 0.5
        x_max = np.max(number_of_contacts_reference) + 0.5

        x_range = (x_min, x_max)
        N_bins = int(x_max - x_min)

        kwargs = {"bins": N_bins, "range": x_range, "histtype": "step"}

        # Plot the distribution of contacts
        axes[i].hist(number_of_contacts_reference,  weights=np.ones_like(number_of_contacts_reference)  / cfg.network.N_tot, color='k', **kwargs)
        axes[i].hist(number_of_contacts_restricted, weights=np.ones_like(number_of_contacts_restricted) / cfg.network.N_tot, color='b', **kwargs)

        # Adjust axes
        axes[i].yaxis.set_major_formatter(PercentFormatter(xmax=1))
        axes[i].set(xlim=x_range)
        axes[i].set_title(title)

        if i % 2 == 0 :
            axes[i].set_ylabel('Counts')

    plt.tight_layout()
    fig.savefig(f'Figures/contacts_{cfg.Intervention_contact_matrices_name[0]}.png')