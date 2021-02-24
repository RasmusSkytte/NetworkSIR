# from re import X TODO : Delete line
from sys import version
import numpy as np
# import pandas as pd TODO : Delete line
# import matplotlib.pyplot as plt TODO : Delete line
from pathlib import Path
# import multiprocessing as mp TODO : Delete line
import h5py
# from resource import getrusage, RUSAGE_SELF TODO : Delete line
import warnings
# from importlib import reload TODO : Delete line
import os
from IPython.display import display
from contexttimer import Timer

# import yaml TODO : Delete line

# conda install -c numba/label/dev numba
# import numba as nb TODO : Delete line
# from numba import njit, prange, objmode, typeof TODO : Delete line
from numba.typed import List, Dict # TODO : Delete "Dict"
# import uuid TODO : Delete line
import datetime
from numba.core.errors import (
    NumbaTypeSafetyWarning,
    NumbaExperimentalFeatureWarning,
    NumbaPendingDeprecationWarning, # TODO : Delete line
)


from tinydb import TinyDB, Query
from tqdm import tqdm
from functools import partial
from p_tqdm import p_umap, p_uimap

# import awkward as awkward0  # conda install awkward0, conda install -c conda-forge pyarrow    TODO : Delete line
# import awkward1 as ak  # pip install awkward1 TODO : Delete line

debugging = False
while True :
    path = Path("").cwd()
    if "src" in str(path) :
        os.chdir(path.parent)
        debugging = True
    else :
        break

# print(Path("").cwd())
from src.utils import utils
from src.simulation import nb_simulation
from src.simulation import nb_load_jitclass
from src import file_loaders


hdf5_kwargs = dict(track_order=True)
np.set_printoptions(linewidth=200)


class Simulation :

    def __init__(self, cfg, verbose=False) :

        self.verbose = verbose

        self.cfg = cfg.deepcopy()
        self.cfg.pop("hash")

        self.N_tot = cfg.network.N_tot

        self.hash = cfg.hash

        self.my = nb_simulation.initialize_My(self.cfg)
        utils.set_numba_random_seed(utils.hash_to_seed(self.hash))

        if self.cfg.version == 1 :
            if self.cfg.do_interventions :
                raise AssertionError("interventions not yet implemented for version 1")

        if self.verbose :
            print("Importing work and other matrices")

    def _initialize_network(self) :
        """ Initializing the network for the simulation
        """

        # generate coordinates based on population density
        self.df_coordinates = utils.load_df_coordinates(self.N_tot, self.cfg.network.ID)
        coordinates_raw = utils.df_coordinates_to_coordinates(self.df_coordinates)

        if self.verbose :
            print(f"\nINITIALIZE VERSION {self.cfg.version} NETWORK")

        #Version 1 do not have [house, work, other], from version 2 this is implemented.
        if self.cfg.version >= 2 :
            #(
            #    people_in_household,
            #    age_distribution_per_people_in_household,
            #) = utils.load_household_data()
            household_size_dist_per_kommune, age_distribution_per_person_in_house_per_kommune, kommune_id = utils.load_household_data_kommune_specific()
            N_ages = len(age_distribution_per_person_in_house_per_kommune[0,0])
            kommune_ids = []
            for val in self.df_coordinates["kommune"].values :
                kommune_ids.append(kommune_id.get_loc(val))
            kommune_ids = np.array(kommune_ids)

            self.N_ages = N_ages
            if self.verbose :
                print("Connect Household") #was household and families are used interchangebly. Most places it is changed to house(hold) since it just is people living at the same adress.

            (
                mu_counter,
                counter_ages,
                agents_in_age_group,
            ) = nb_simulation.place_and_connect_families_kommune_specific(
                self.my,
                household_size_dist_per_kommune,
                age_distribution_per_person_in_house_per_kommune,
                coordinates_raw,
                kommune_ids,
                self.N_ages,
                verbose=self.verbose)

            if self.verbose :
                print("Connecting work and others, currently slow, please wait")

            nb_simulation.connect_work_and_others(
                self.my,
                N_ages,
                mu_counter,
                np.array(self.cfg.network.work_matrix),
                np.array(self.cfg.network.other_matrix),
                agents_in_age_group,
                verbose=self.verbose)

        else :

            if self.verbose :
                print("SETUP WEIGHTS AND COORDINATES")
            nb_simulation.v1_initialize_my(self.my, coordinates_raw)

            if self.verbose :
                print("CONNECT NODES")
            nb_simulation.v1_connect_nodes(self.my)

            agents_in_age_group = List()
            agents_in_age_group.append(np.arange(self.cfg.network.N_tot, dtype=np.uint32))

        self.agents_in_age_group = agents_in_age_group

        return None

    def _save_initialized_network(self, filename) :
        if self.verbose :
            print(f"Saving initialized network to {filename}", flush=True)
        utils.make_sure_folder_exist(filename)
        my_hdf5ready = nb_load_jitclass.jitclass_to_hdf5_ready_dict(self.my)

        with h5py.File(filename, "w", **hdf5_kwargs) as f :
            group_my = f.create_group("my")
            nb_load_jitclass.save_jitclass_hdf5ready(group_my, my_hdf5ready)
            utils.NestedArray(self.agents_in_age_group).add_to_hdf5_file(f, "agents_in_age_group")
            f.create_dataset("N_ages", data=self.N_ages)
            self._add_cfg_to_hdf5_file(f)

    def _load_initialized_network(self, filename) :
        if self.verbose :
            print(f"Loading previously initialized network, please wait", flush=True)
        with h5py.File(filename, "r") as f :
            self.agents_in_age_group = utils.NestedArray.from_hdf5(
                f, "agents_in_age_group"
            ).to_nested_numba_lists()
            self.N_ages = f["N_ages"][()]

            my_hdf5ready = nb_load_jitclass.load_jitclass_to_dict(f["my"])
            self.my = nb_load_jitclass.load_My_from_dict(my_hdf5ready, self.cfg)
        self.df_coordinates = utils.load_df_coordinates(self.N_tot, self.cfg.network.ID)

        # Update connection weights
        for agent in range(self.cfg.network.N_tot) :
            nb_simulation.set_infection_weight(self.my, agent)

    def initialize_network(self, force_rerun=False, save_initial_network=False, only_initialize_network=False, force_load_initial_network=False) :
        filename = "Initialized_networks/"
        filename += f"{utils.cfg_to_hash(self.cfg.network, exclude_ID=False)}.hdf5"

        if force_load_initial_network :
            initialize_network = False
            if self.verbose :
                print("Force loading initialized network")

        elif force_rerun :
            initialize_network = True
            if self.verbose :
                print("Initializing network since it was forced to")

        elif not utils.file_exists(filename) :
            initialize_network = True
            if self.verbose :
                print("Initializing network since the hdf5-file does not exist")

        else :
            initialize_network = False

        # Initalizing network and (optionally) saving it
        if initialize_network :
            utils.set_numba_random_seed(self.cfg.network.ID)

            self._initialize_network()
            if save_initial_network :
                self._save_initialized_network(filename)

        # Loading initialized network
        elif not only_initialize_network :
            self._load_initialized_network(filename)

    def initialize_states(self) :
        utils.set_numba_random_seed(utils.hash_to_seed(self.hash))

        if self.verbose :
            print("\nINITIAL INFECTIONS")

        np.random.seed(utils.hash_to_seed(self.hash))

        self.nts = 0.1  # Time step (0.1 - ten times a day)
        self.N_states = 9  # number of states
        self.N_infectious_states = 4  # This means the 5'th state
        self.initial_ages_exposed = np.arange(self.N_ages)  # means that all ages are exposed

        self.state_total_counts     = np.zeros(self.N_states, dtype=np.uint32)
        self.variant_counts         = np.zeros(2, dtype=np.uint32)  # TODO: Generalize this to work for more variants
        self.infected_per_age_group = np.zeros(self.N_ages, dtype=np.uint32)

        self.agents_in_state = utils.initialize_nested_lists(self.N_states, dtype=np.uint32)

        self.g = nb_simulation.Gillespie(self.my, self.N_states)

        self.SIR_transition_rates = utils.initialize_SIR_transition_rates(self.N_states, self.N_infectious_states, self.cfg)


        if self.cfg.initialize_at_kommune_level :
            infected_per_kommune, immunized_per_kommune, kommune_names, my_kommune = file_loaders.load_kommune_data(self.df_coordinates, self.cfg.initial_infection_distribution)

            if self.cfg.R_init > 0 :
                raise ValueError("R_init not implemented when using kommune configuration")

            nb_simulation.initialize_states_from_kommune_data(
                self.my,
                self.g,
                self.SIR_transition_rates,
                self.state_total_counts,
                self.variant_counts,
                self.infected_per_age_group,
                self.agents_in_state,
                self.agents_in_age_group,
                self.initial_ages_exposed,                # self.N_infectious_states,
                self.N_states,
                infected_per_kommune,
                immunized_per_kommune,
                kommune_names,
                my_kommune,
                verbose=self.verbose)

        else :
            nb_simulation.initialize_states(
                self.my,
                self.g,
                self.SIR_transition_rates,
                self.state_total_counts,
                self.variant_counts,
                self.infected_per_age_group,
                self.agents_in_state,
                self.agents_in_age_group,
                self.initial_ages_exposed,
                self.N_states,
                verbose=self.verbose)

    def run_simulation(self, verbose_interventions=None) :
        utils.set_numba_random_seed(utils.hash_to_seed(self.hash))

        if self.verbose :
            print("\nRUN SIMULATION")

        if self.cfg.make_restrictions_at_kommune_level :
            labels = self.df_coordinates["idx"].values
        else :
            labels = self.df_coordinates["idx"].values * 0

        if verbose_interventions is None :
            verbose_interventions = self.verbose

        # Load the projected vaccination schedule
        # TODO: This should properably be done at cfg generation for consistent hashes
        vaccinations_per_age_group, vaccination_schedule = utils.load_vaccination_schedule(self.cfg)

        # Load the restriction contact matrices
        # TODO: This should properably be done at cfg generation for consistent hashes
        work_matrix_restrict = []
        other_matrix_restrict = []

        for scenario in self.cfg.Intervention_contact_matrices_name :
            tmp_work_matrix_restrict, tmp_other_matrix_restrict, _, _ = utils.load_contact_matrices(scenario=scenario)

            work_matrix_restrict.append(tmp_work_matrix_restrict)
            other_matrix_restrict.append(tmp_other_matrix_restrict)


        self.intervention = nb_simulation.Intervention(
            self.my.cfg,
            self.my.cfg_network,
            labels = labels,
            vaccinations_per_age_group = np.array(vaccinations_per_age_group),
            vaccination_schedule = np.array(vaccination_schedule),
            work_matrix_restrict = np.array(work_matrix_restrict),
            other_matrix_restrict = np.array(other_matrix_restrict),
            verbose=verbose_interventions)

        res = nb_simulation.run_simulation(
            self.my,
            self.g,
            self.intervention,
            self.SIR_transition_rates,
            self.state_total_counts,
            self.variant_counts,
            self.infected_per_age_group,
            self.agents_in_state,
            self.N_states,
            self.N_infectious_states,
            self.nts,
            self.verbose)


        out_time, out_state_counts, out_variant_counts, out_infected_per_age_group, out_my_state, intervention = res

        self.out_time = out_time
        self.my_state = np.array(out_my_state)
        self.df = utils.counts_to_df(out_time, out_state_counts, out_variant_counts, out_infected_per_age_group)

        self.intervention = intervention

        return self.df

    def _get_filename(self, name="ABM", filetype="hdf5") :
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        filename = f"Output/{name}/{self.hash}/{name}_{date}_{self.hash}_ID__{self.cfg.network.ID}.{filetype}"
        return filename

    def _save_cfg(self) :
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        filename_cfg = f"Output/cfgs/cfg_{date}_{self.hash}.yaml"
        self.cfg.dump_to_file(filename_cfg, exclude="network.ID")
        return None

    def _add_cfg_to_hdf5_file(self, f, cfg=None) :
        if cfg is None :
            cfg = self.cfg

        utils.add_cfg_to_hdf5_file(f, cfg)

    def _save_dataframe(self, save_csv=False, save_hdf5=True) :

        # Save CSV
        if save_csv :
            filename_csv = self._get_filename(name="ABM", filetype="csv")
            utils.make_sure_folder_exist(filename_csv)
            self.df.to_csv(filename_csv, index=False)

        if save_hdf5 :
            filename_hdf5 = self._get_filename(name="ABM", filetype="hdf5")
            utils.make_sure_folder_exist(filename_hdf5)
            with h5py.File(filename_hdf5, "w", **hdf5_kwargs) as f :  #
                f.create_dataset("df", data=utils.dataframe_to_hdf5_format(self.df))
                self._add_cfg_to_hdf5_file(f)

        return None

    def _save_simulation_results(self, save_only_ID_0=False, time_elapsed=None) :

        if save_only_ID_0 and self.cfg.network.ID != 0 :
            return None

        filename_hdf5 = self._get_filename(name="network", filetype="hdf5")
        utils.make_sure_folder_exist(filename_hdf5)

        with h5py.File(filename_hdf5, "w", **hdf5_kwargs) as f :  #
            f.create_dataset("my_state", data=self.my_state)
            f.create_dataset("my_corona_type", data=self.my.corona_type)
            f.create_dataset("my_number_of_contacts", data=self.my.number_of_contacts)
            f.create_dataset("day_found_infected", data=self.intervention.day_found_infected)
            f.create_dataset("coordinates", data=self.my.coordinates)
            # import ast; ast.literal_eval(str(cfg))
            f.create_dataset("cfg_str", data=str(self.cfg))
            f.create_dataset("R_true", data=self.intervention.R_true_list)
            f.create_dataset("freedom_impact", data=self.intervention.freedom_impact_list)
            f.create_dataset("R_true_brit", data=self.intervention.R_true_list_brit)
            f.create_dataset("df", data=utils.dataframe_to_hdf5_format(self.df))
            # f.create_dataset(
            #     "df_coordinates",
            #     data=utils.dataframe_to_hdf5_format(self.df_coordinates, cols_to_str="kommune"),
            # )

            if time_elapsed :
                f.create_dataset("time_elapsed", data=time_elapsed)

            self._add_cfg_to_hdf5_file(f)

        return None

    def save(self, save_csv=False, save_hdf5=True, save_only_ID_0=False, time_elapsed=None) :
        self._save_cfg()
        self._save_dataframe(save_csv=save_csv, save_hdf5=save_hdf5)
        self._save_simulation_results(save_only_ID_0=save_only_ID_0, time_elapsed=time_elapsed)


#%%


def run_single_simulation(
    cfg,
    verbose=False,
    force_rerun=False,
    only_initialize_network=False,
    save_initial_network=False,
    save_csv=False,
) :
    with Timer() as t, warnings.catch_warnings() :
        if not verbose :
            # ignore warning about run_algo
            warnings.simplefilter("ignore", NumbaExperimentalFeatureWarning)
            warnings.simplefilter("ignore", NumbaTypeSafetyWarning)
            # warnings.simplefilter("ignore", NumbaPendingDeprecationWarning)

        simulation = Simulation(cfg, verbose)

        simulation.initialize_network(
            force_rerun=force_rerun, save_initial_network=save_initial_network, only_initialize_network=only_initialize_network
        )

        if only_initialize_network :
            return None

        simulation.initialize_states()

        simulation.run_simulation()

        simulation.save(time_elapsed=t.elapsed, save_hdf5=True, save_csv=save_csv)

    return cfg


def update_database(db_cfg, q, cfg) :

    if not db_cfg.contains((q.hash == cfg.hash) & (q.network.ID == cfg.network.ID)) :
        db_cfg.insert(cfg)


def run_simulations(
        simulation_parameters,
        N_runs=2,
        num_cores_max=None,
        N_tot_max=False,
        verbose=False,
        force_rerun=False,
        dry_run=False,
        **kwargs) :

    if isinstance(simulation_parameters, dict) :
        simulation_parameters = utils.format_simulation_paramters(simulation_parameters)
        cfgs_all = utils.generate_cfgs(simulation_parameters, N_runs, N_tot_max, verbose=verbose)

        N_tot_max = utils.d_num_cores_N_tot[utils.extract_N_tot_max(simulation_parameters)]

    elif isinstance(simulation_parameters[0], utils.DotDict) :
        cfgs_all = simulation_parameters

        N_tot_max = np.max([cfg.network.N_tot for cfg in cfgs_all])

    else :
        raise ValueError(f"simulation_parameters not of the correct type")

    if len(cfgs_all) == 0 :
        N_files = 0
        return N_files

    db_cfg = utils.get_db_cfg()
    q = Query()

    db_counts  = np.array([db_cfg.count((q.hash == cfg.hash) & (q.network.ID == cfg.network.ID)) for cfg in cfgs_all])

    assert np.max(db_counts) <= 1

    # keep only cfgs that are not in the database already
    if force_rerun :
        cfgs = cfgs_all
    else :
        cfgs = [cfg for (cfg, count) in zip(cfgs_all, db_counts) if count == 0]

    N_files = len(cfgs)

    num_cores = utils.get_num_cores_N_tot(N_tot_max, num_cores_max)

    if isinstance(simulation_parameters, dict) :
        s_simulation_parameters = str(simulation_parameters)
    elif isinstance(simulation_parameters, list) :
        s_simulation_parameters = f"{len(simulation_parameters)} runs"
    else :
        raise AssertionError("simulation_parameters neither list nor dict")

    print( f"\n\n" f"Generating {N_files :3d} network-based simulations",
           f"with {num_cores} cores",
           f"based on {s_simulation_parameters}.",
           "Please wait. \n",
           flush=True)

    if dry_run or N_files == 0 :
        return N_files

    # kwargs = {}
    if num_cores == 1 :
        for cfg in tqdm(cfgs) :
            cfg_out = run_single_simulation(cfg, save_initial_network=True, verbose=verbose, **kwargs)
            update_database(db_cfg, q, cfg_out)

    else :
        # First generate the networks
        f_single_network = partial(run_single_simulation, only_initialize_network=True, save_initial_network=True, verbose=verbose, **kwargs)

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
        f_single_simulation = partial(run_single_simulation, verbose=verbose, **kwargs)
        for cfg in p_uimap(f_single_simulation, cfgs, num_cpus=num_cores) :
            update_database(db_cfg, q, cfg)

    return N_files