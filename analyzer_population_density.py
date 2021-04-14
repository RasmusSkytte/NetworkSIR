
import pandas as pd
import numpy as np

from contexttimer import Timer

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import h5py

from tqdm import tqdm

from src.utils      import utils
from src.simulation import simulation
from src.utils      import file_loaders
from src.analysis   import plotters

from src.analysis.helpers  import *




# This runs simulations with specified percentage effects of the seasonality model
params, start_date = utils.load_params("cfg/analyzers/population_density.yaml")

if utils.is_local_computer():
    f = 0.1
    n_steps = 1
    num_cores_max = 1
    N_runs = 1
else :
    f = 1
    n_steps = 3
    num_cores_max = 5
    N_runs = 1


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

        N_files_total += simulation.run_simulations(params, N_runs=N_runs, num_cores_max=num_cores_max, verbose=verbose)

    print(f"\n{N_files_total:,} files were generated, total duration {utils.format_time(t.elapsed)}")
    print("Finished simulating!")

if not utils.is_local_computer():
    return


# Load the simulations
subset = {'labels' : 'kommune'}
data = file_loaders.ABM_simulations(subset=subset)

if len(data.networks) == 0 :
    raise ValueError(f'No files loaded with subset: {subset}')


# Prepare output file
fig_name = 'Figures/population_density.png'
file_loaders.make_sure_folder_exist(fig_name)


# Prepare figure
fig = plt.figure(figsize=(12, 12))
axes = plt.gca()

print('Plotting the individual ABM simulations. Please wait', flush=True)
for network in tqdm(data.iter_network_files(), total=len(data.networks)) :

    # Load from network file
    cfg = utils.read_cfg_from_hdf5_file(network)
    state, label, conection_status = plotters._load_data_from_network_file(network, ['my_state', 'my_label', 'my_connection_status'])
    conection_status
    # Count infected
    I = np.argmax(np.sum(state >= 0, axis=1) > 5_800_000 * f * 0.065)
    d = pd.DataFrame(data = zip(state[I], label), columns=['state', 'label'])
    d['infected'] = d['state'].apply(lambda x : 1 if x >= 0 else 0)

    # Load the kommune_dict from the initialized network to get municipality info.
    filename = f'Initialized_networks/{utils.cfg_to_hash(cfg.network, exclude_ID=False)}.hdf5'

    kommune_dict = {'id_to_name' : pd.read_hdf(filename, 'id_to_name'),
                    'name_to_id' : pd.read_hdf(filename, 'name_to_id')}

    d['kommune'] = kommune_dict['id_to_name'][d['label']].values

    # Load more network information
    with h5py.File(filename, 'r') as hf :
        my = file_loaders.load_jitclass_to_dict(hf['my'])


    # Compute the area for each municipality
    # DST: ARE207, 2021
    kommune_to_area = pd.Series({'København' : 90.10, 'Frederiksberg' : 8.70, 'Dragør' : 18.30, 'Tårnby' : 66.10, 'Albertslund' : 23.40, 'Ballerup' : 33.90, 'Brøndby' : 21.00, 'Gentofte' : 25.60, 'Gladsaxe' : 24.90, 'Glostrup' : 13.30, 'Herlev' : 12.10, 'Hvidovre' : 22.90, 'Høje-Taastrup' : 78.20, 'Ishøj' : 26.50, 'Lyngby-Taarbæk' : 38.80, 'Rødovre' : 12.20, 'Vallensbæk' : 9.50, 'Allerød' : 67.40, 'Egedal' : 125.80, 'Fredensborg' : 112.10, 'Frederikssund' : 248.50, 'Furesø' : 56.80, 'Gribskov' : 279.50, 'Halsnæs' : 122.00, 'Helsingør' : 118.90, 'Hillerød' : 213.50, 'Hørsholm' : 31.30, 'Rudersdal' : 73.30, 'Bornholm' : 588.50, 'Christiansø' : 0.00, 'Greve' : 60.40, 'Køge' : 257.50, 'Lejre' : 238.90, 'Roskilde' : 211.80, 'Solrød' : 40.10, 'Faxe' : 404.90, 'Guldborgsund' : 900.70, 'Holbæk' : 577.30, 'Kalundborg' : 575.40, 'Lolland' : 886.60, 'Næstved' : 676.80, 'Odsherred' : 354.20, 'Ringsted' : 294.70, 'Slagelse' : 568.30, 'Sorø' : 308.50, 'Stevns' : 250.10, 'Vordingborg' : 620.20, 'Assens' : 511.60, 'Faaborg-Midtfyn' : 633.60, 'Kerteminde' : 206.20, 'Langeland' : 290.70, 'Middelfart' : 298.90, 'Nordfyns' : 452.30, 'Nyborg' : 276.80, 'Odense' : 305.60, 'Svendborg' : 415.40, 'Ærø' : 90.10, 'Billund' : 540.20, 'Esbjerg' : 795.30, 'Fanø' : 57.60, 'Fredericia' : 133.60, 'Haderslev' : 816.80, 'Kolding' : 604.40, 'Sønderborg' : 496.50, 'Tønder' : 1283.90, 'Varde' : 1240.10, 'Vejen' : 813.70, 'Vejle' : 1058.60, 'Aabenraa' : 940.70, 'Favrskov' : 540.30, 'Hedensted' : 551.00, 'Horsens' : 519.40, 'Norddjurs' : 721.00, 'Odder' : 223.70, 'Randers' : 747.80, 'Samsø' : 113.60, 'Silkeborg' : 850.40, 'Skanderborg' : 416.90, 'Syddjurs' : 689.80, 'Aarhus' : 468.10, 'Herning' : 1321.10, 'Holstebro' : 793.00, 'Ikast-Brande' : 733.50, 'Lemvig' : 509.70, 'Ringkøbing-Skjern' : 1473.40, 'Skive' : 683.50, 'Struer' : 246.20, 'Viborg' : 1408.90, 'Brønderslev' : 633.10, 'Frederikshavn' : 652.30, 'Hjørring' : 926.90, 'Jammerbugt' : 864.10, 'Læsø' : 122.00, 'Mariagerfjord' : 718.30, 'Morsø' : 366.50, 'Rebild' : 621.30, 'Thisted' : 1072.10, 'Vesthimmerlands' : 769.80, 'Aalborg' : 1137.40})

    d['area'] = kommune_to_area [d['kommune']].values
    d['total']= np.ones_like(d['state'])

    # Group infected by area of municipality
    data = pd.pivot_table(d, values=['infected', 'area', 'total'], index='kommune', aggfunc={'infected':'mean', 'area':'mean', 'total' : 'count'})

    I = np.argmax(cfg.network.rho == np.array(params['rho']))
    plt.scatter(np.log(data['total'] / (f * data['area'] )), data['infected'], color=plt.cm.tab10(I), label=f'rho = {cfg.network.rho}')


    # How does rho translate to distance?
    #print(conection_status[0])
    #print(conection_status[0] == True)

    connections = utils.NestedArray.from_dict(my['connections']).to_nested_numba_lists()

    coordinates = my['coordinates']

    #distances = []
    #for agent in range(len(connections)) :

    agent_distances = lambda agent : [utils.haversine_scipy(coordinates[agent], coordinates[contact]) for contact in np.array(connections[agent])]

    distances = []
    for agent in np.arange(len(connections)) :
        distances.extend(agent_distances(agent))

    print('--------------')
    print(cfg.network.rho)
    print(np.mean(distances))
    print(np.std(distances))




plt.legend()
plt.ylabel('Percent infected')
plt.xlabel('log(density)')

axes.yaxis.set_major_formatter(PercentFormatter(xmax=1))

plt.savefig(fig_name)
