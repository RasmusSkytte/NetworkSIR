import numpy as np
import pandas as pd

from datetime import datetime

from scipy.stats import norm

from tqdm import tqdm

from src.utils import utils
from src.simulation import simulation
from src import file_loaders

from tinydb import Query

from functools import partial
from p_tqdm import p_umap, p_uimap

if utils.is_local_computer() :
    from src import rc_params
    import matplotlib.pyplot as plt

from contexttimer import Timer


if utils.is_local_computer():
    f = 0.1
    N = 10
    n_steps = 0 # 2 per sigma
    num_cores_max = 3
else :
    f = 1
    N = 5
    n_steps = 2 # 2 per sigma
    num_cores_max = 15


cfgs_all = []

filenames = ["cfg/simulation_parameters_2021_fase1.yaml", 
             "cfg/simulation_parameters_2021_fase2.yaml",
             "cfg/simulation_parameters_2021_fase2_sce7.yaml",
             "cfg/simulation_parameters_2021_fase2_sce8.yaml"]

for filename in filenames :
    params, start_date = utils.load_params(filename)

    # Scale the population
    params["N_tot"]  = int(params["N_tot"]  * f)
    params["N_init"] = int(params["N_init"] * f)
    params["R_init"] = int(params["R_init"] * f)


    # Store the variables we loop over
    beta = params["beta"]
    beta_sigma = 0.00025

    beta_UK_multiplier = params["beta_UK_multiplier"]
    beta_UK_multiplier_sigma = 0.1

    N_init = params["N_init"]
    N_init_sigma = 1000 * f

    N_init_UK_frac = params["N_init_UK_frac"]
    N_init_UK_frac_sigma = 0.005


    # Define the noise function
    noise = lambda m, s, d : m if s == 0 else np.round(m + np.linspace(-s, s, 4*s + 1) * d, 5)

    z = np.arange(n_steps, -1, -1)
    p = norm.pdf(z / 2) ** 4
    p /= p.max()

    N_runs = np.floor(p * N).astype(int)

    # Run the ML plot loop
    for s, n in zip(z, N_runs) :

        if n == 0 :
            continue

        if   s == 0 :
            cfgs_all.extend(utils.generate_cfgs(params, N_runs=n))
        
        elif s == 1 :
            
            params["beta"]               = noise(beta,               s, beta_sigma)
            cfgs_all.extend(utils.generate_cfgs(params, N_runs=n))
            params["beta"]               = beta
            
            params["beta_UK_multiplier"] = noise(beta_UK_multiplier, s, beta_UK_multiplier_sigma)
            cfgs_all.extend(utils.generate_cfgs(params, N_runs=n))
            params["beta_UK_multiplier"] = beta_UK_multiplier
            
            params["N_init"]             = noise(N_init,             s, N_init_sigma)
            cfgs_all.extend(utils.generate_cfgs(params, N_runs=n))
            params["N_init"]             = N_init
            
            params["N_init_UK_frac"]     = noise(N_init_UK_frac,     s, N_init_UK_frac_sigma)
            cfgs_all.extend(utils.generate_cfgs(params, N_runs=n))
            params["N_init_UK_frac"]     = N_init_UK_frac
        
        elif s == 2 :

            params["beta"]               = noise(beta,               s, beta_sigma)
            cfgs_all.extend(utils.generate_cfgs(params, N_runs=n))
            params["beta"]               = beta
            
            params["beta_UK_multiplier"] = noise(beta_UK_multiplier, s, beta_UK_multiplier_sigma)
            cfgs_all.extend(utils.generate_cfgs(params, N_runs=n))
            params["beta_UK_multiplier"] = beta_UK_multiplier
            
            params["N_init"]             = noise(N_init,             s, N_init_sigma)
            cfgs_all.extend(utils.generate_cfgs(params, N_runs=n))
            params["N_init"]             = N_init
            
            params["N_init_UK_frac"]     = noise(N_init_UK_frac,     s, N_init_UK_frac_sigma)
            cfgs_all.extend(utils.generate_cfgs(params, N_runs=n))
            params["N_init_UK_frac"]     = N_init_UK_frac


            params["beta"]               = noise(beta,               s-1, beta_sigma)
            params["beta_UK_multiplier"] = noise(beta_UK_multiplier, s-1, beta_UK_multiplier_sigma)
            cfgs_all.extend(utils.generate_cfgs(params, N_runs=n))
            params["beta"]               = beta
            params["beta_UK_multiplier"] = beta_UK_multiplier

            params["beta"]               = noise(beta,               s-1, beta_sigma)
            params["N_init"]             = noise(N_init,             s-1, N_init_sigma)
            cfgs_all.extend(utils.generate_cfgs(params, N_runs=n))
            params["beta"]               = beta
            params["N_init"]             = N_init

            params["beta"]               = noise(beta,               s-1, beta_sigma)
            params["N_init_UK_frac"]     = noise(N_init_UK_frac,     s-1, N_init_UK_frac_sigma)
            cfgs_all.extend(utils.generate_cfgs(params, N_runs=n))
            params["beta"]               = beta
            params["N_init_UK_frac"]     = N_init_UK_frac
        
            
            params["beta_UK_multiplier"] = noise(beta_UK_multiplier, s-1, beta_UK_multiplier_sigma)
            params["N_init"]             = noise(N_init,             s-1, N_init_sigma)
            cfgs_all.extend(utils.generate_cfgs(params, N_runs=n))
            params["beta_UK_multiplier"] = beta_UK_multiplier
            params["N_init"]             = N_init

            params["beta_UK_multiplier"] = noise(beta_UK_multiplier, s-1, beta_UK_multiplier_sigma)
            params["N_init_UK_frac"]     = noise(N_init_UK_frac,     s-1, N_init_UK_frac_sigma)
            cfgs_all.extend(utils.generate_cfgs(params, N_runs=n))
            params["beta_UK_multiplier"] = beta_UK_multiplier
            params["N_init_UK_frac"]     = N_init_UK_frac

            params["N_init"]             = noise(N_init,             s-1, N_init_sigma)
            params["N_init_UK_frac"]     = noise(N_init_UK_frac,     s-1, N_init_UK_frac_sigma)
            cfgs_all.extend(utils.generate_cfgs(params, N_runs=n))
            params["beta"]               = beta
            params["N_init_UK_frac"]     = N_init_UK_frac

        else :
            raise ValueError("Too many sigma, not yet implemented") # TODO: Implement

hashes = [utils.cfg_to_hash(cfg, exclude_ID=False) for cfg in cfgs_all]
_, ind = np.unique(hashes, return_index=True)

for i in reversed(range(np.max(ind))) :
    if i not in ind :
        cfgs_all.pop(i)

N_tot_max = np.max([cfg.network.N_tot for cfg in cfgs_all])

if __name__ == "__main__":
    with Timer() as t:

        db_cfg = utils.get_db_cfg()
        q = Query()

        db_counts  = np.array([db_cfg.count((q.hash == cfg.hash) & (q.network.ID == cfg.network.ID)) for cfg in cfgs_all])

        assert np.max(db_counts) <= 1

        cfgs = [cfg for (cfg, count) in zip(cfgs_all, db_counts) if count == 0]

        N_files = len(cfgs)

        num_cores = utils.get_num_cores_N_tot(N_tot_max, num_cores_max)


        # kwargs = {}
        if num_cores == 1 :
            for cfg in tqdm(cfgs) :
                cfg_out = simulation.run_single_simulation(cfg, save_initial_network=True, verbose=False)
                simulation.update_database(db_cfg, q, cfg_out)

        else :
            # First generate the networks
            f_single_network = partial(simulation.run_single_simulation, only_initialize_network=True, save_initial_network=True, verbose=False)

            # Get the network hashes
            network_hashes = set([utils.cfg_to_hash(cfg.network, exclude_ID=False) for cfg in cfgs])

            # Get list of unique cfgs
            cfgs_network = []
            for cfg in cfgs :
                network_hash = utils.cfg_to_hash(cfg.network, exclude_ID=False)

                if network_hash in network_hashes :
                    cfgs_network.append(cfg)
                    network_hashes.remove(network_hash)

            # Generate the networks
            print("Generating networks. Please wait")
            p_umap(f_single_network, cfgs_network, num_cpus=num_cores)

            # Then run the simulations on the network
            print("Running simulations. Please wait")
            f_single_simulation = partial(simulation.run_single_simulation, verbose=False)
            for cfg in p_uimap(f_single_simulation, cfgs, num_cpus=num_cores) :
                simulation.update_database(db_cfg, q, cfg)


print(f"\n{N_files:,} files were generated, total duration {utils.format_time(t.elapsed)}")
print("Finished simulating!")