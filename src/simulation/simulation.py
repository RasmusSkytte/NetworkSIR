import numpy as np
import pandas as pd

from numba.typed import List
from numba.core.errors import (
    NumbaTypeSafetyWarning,
    NumbaExperimentalFeatureWarning,
    NumbaPendingDeprecationWarning, # TODO : Delete line
)



from pathlib import Path
import os

import h5py
import warnings
import datetime
from contexttimer import Timer

from tinydb import Query

from functools import partial

from tqdm import tqdm
from p_tqdm import p_umap, p_uimap


check_distributions = False

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
from src.utils import file_loaders

from src.simulation import nb_simulation
from src.simulation import nb_jitclass
from src.simulation import nb_network

hdf5_kwargs = dict(track_order=True)
np.set_printoptions(linewidth=200)


class Simulation :

    def __init__(self, cfg, verbose=False) :

        self.verbose = verbose

        self.cfg = cfg.deepcopy()
        self.cfg.pop("hash")

        self.N_tot = cfg.network.N_tot

        self.hash = cfg.hash
        self.my = nb_jitclass.initialize_My(self.cfg.deepcopy())

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

        coordinates_raw   = utils.df_coordinates_to_coordinates(self.df_coordinates)
        self.kommune_dict = utils.df_coordinates_to_kommune_dict(self.df_coordinates)

        if self.verbose :
            print(f"\nINITIALIZE VERSION {self.cfg.version} NETWORK")


        household_size_distribution, age_distribution_in_households = file_loaders.load_household_data(self.kommune_dict)

        self.N_ages = len(age_distribution_in_households[0, 0])
        kommune_ids = np.array(self.df_coordinates.idx)


        if self.verbose :
            print("Connect Household") #was household and families are used interchangebly. Most places it is changed to house(hold) since it just is people living at the same adress.

        (
            mu_counter,
            counter_ages,
            agents_in_age_group,
        ) = nb_network.place_and_connect_families_kommune_specific(
            self.my,
            household_size_distribution,
            age_distribution_in_households,
            coordinates_raw,
            kommune_ids,
            self.N_ages,
            verbose=self.verbose)


        if self.verbose :
            print("Connecting work and others, currently slow, please wait")

        nb_network.connect_work_and_others(
            self.my,
            self.N_ages,
            mu_counter,
            np.array(self.cfg.network.work_matrix),
            np.array(self.cfg.network.other_matrix),
            agents_in_age_group,
            verbose=self.verbose)


        self.agents_in_age_group = agents_in_age_group

    def _save_initialized_network(self, filename) :
        if self.verbose :
            print(f"Saving initialized network to {filename}", flush=True)
        file_loaders.make_sure_folder_exist(filename)
        my_hdf5ready = file_loaders.jitclass_to_hdf5_ready_dict(self.my)

        with h5py.File(filename, "w", **hdf5_kwargs) as f :
            group_my = f.create_group("my")
            file_loaders.save_jitclass_hdf5ready(group_my, my_hdf5ready)
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

            my_hdf5ready = file_loaders.load_jitclass_to_dict(f["my"])
            self.my = file_loaders.load_My_from_dict(my_hdf5ready, self.cfg.deepcopy())
        self.df_coordinates = utils.load_df_coordinates(self.N_tot, self.cfg.network.ID)
        self.kommune_dict   = utils.df_coordinates_to_kommune_dict(self.df_coordinates)


        # Update connection weights
        for agent in range(self.cfg.network.N_tot) :
            nb_network.set_infection_weight(self.my, agent)


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

        elif not file_loaders.file_exists(filename) :
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


    def intialize_interventions(self, verbose_interventions=None) :

        if self.verbose :
            print("\nINITIALING INTERVENTIONS")

        if self.cfg.labels.lower() == "kommune" :
            labels = self.my.kommune

        elif self.cfg.labels.lower() == "custom" :
            labels_raw = self.my.kommune
            labels = np.zeros(np.shape(labels_raw))

            for new_label, label_group in enumerate(self.cfg['label_map']) :
                labels[np.isin(labels_raw, self.kommune_dict['name_to_id'][label_group])] = new_label + 1

        elif self.cfg.labels.lower() == "none" :
            labels = np.zeros(np.shape(self.my.kommune))

        else :
            raise ValueError(f'Label name: {self.cfg.labels.lower()} not known')


        if verbose_interventions is None :
            verbose_interventions = self.verbose

        # Load the projected vaccination schedule
        # TODO: This should properably be done at cfg generation for consistent hashes
        vaccinations_per_age_group, vaccination_schedule = file_loaders.load_vaccination_schedule(self.cfg)

        # Load the restriction contact matrices
        # TODO: This should properably be done at cfg generation for consistent hashes
        work_matrix_restrict  = []
        other_matrix_restrict = []

        for scenario in self.cfg.Intervention_contact_matrices_name :
            tmp_work_matrix_restrict, tmp_other_matrix_restrict, _, _ = file_loaders.load_contact_matrices(scenario=scenario, N_labels=len(np.unique(labels)))

            # Check the loaded contact matrices have the right size
            if not len(tmp_other_matrix_restrict) == len(np.unique(labels)) :
                raise ValueError(f'Number of labels ({len(np.unique(labels))}) does not match the number of contact matrices ({len(tmp_other_matrix_restrict)}) for scenario: {scenario} and label: {self.cfg.labels}')

            work_matrix_restrict.append(tmp_work_matrix_restrict)
            other_matrix_restrict.append(tmp_other_matrix_restrict)


        wm = np.array(work_matrix_restrict)
        om = np.array(other_matrix_restrict)

        for s in range(len(self.cfg.Intervention_contact_matrices_name)) :
            for l in range(len(np.unique(labels))) :
                    wm[s,l,:,:] *= self.cfg.label_betas[l]
                    om[s,l,:,:] *= self.cfg.label_betas[l]

        # Store the labels in my
        self.my.initialize_labels(labels)

        self.intervention = nb_jitclass.Intervention(
            self.my.cfg,
            self.my.cfg_network,
            labels = labels,
            vaccinations_per_age_group = vaccinations_per_age_group,
            vaccination_schedule = vaccination_schedule,
            work_matrix_restrict = wm,
            other_matrix_restrict = om,
            verbose=verbose_interventions)


    def initialize_states(self) :
        utils.set_numba_random_seed(utils.hash_to_seed(self.hash))

        if self.verbose :
            print("\nINITIAL INFECTIONS")

        np.random.seed(utils.hash_to_seed(self.hash))

        self.N_states = 9  # number of states
        self.N_infectious_states = 4  # This means the 5'th state

        self.nts = 0.1  # Time step (0.1 - ten times a day)
        self.initial_ages_exposed = np.arange(self.N_ages)  # means that all ages are exposed

        self.state_total_counts            = np.zeros(self.N_states, dtype=np.uint32)
        self.stratified_infection_counts   = np.zeros((self.intervention.N_labels, 2, self.N_ages), dtype=np.uint32)
        self.stratified_vaccination_counts = np.zeros(self.N_ages, dtype=np.uint32)

        self.agents_in_state = utils.initialize_nested_lists(self.N_states, dtype=np.uint32)

        self.g = nb_jitclass.Gillespie(self.my, self.N_states, self.N_infectious_states)

        # Find the possible agents
        possible_agents = nb_simulation.find_possible_agents(self.my, self.initial_ages_exposed, self.agents_in_age_group)

        # Load the age distribution for infected
        age_distribution_infected, age_distribution_immunized = file_loaders.load_infection_age_distributions(self.cfg.initial_infection_distribution, self.N_ages)

        # Adjust for the untested fraction
        age_distribution_infected  /= self.cfg.testing_penetration
        age_distribution_immunized /= self.cfg.testing_penetration

        # Convert to probability
        age_distribution_infected  /= age_distribution_infected.sum()
        age_distribution_immunized /= age_distribution_immunized.sum()

        # Set the probability to choose agents
        if self.cfg.initialize_at_kommune_level :

            infected_per_kommune, immunized_per_kommune = file_loaders.load_kommune_infection_distribution(self.cfg.initial_infection_distribution, self.kommune_dict)

            infected_per_kommune  /= infected_per_kommune.sum()
            immunized_per_kommune /= immunized_per_kommune.sum()

            # Choose the a number of initially infected and immunized per kommune
            kommune_ids = np.arange(len(infected_per_kommune))
            N_kommune   = np.zeros(np.shape(kommune_ids), dtype=int)
            R_kommune   = np.zeros(np.shape(kommune_ids), dtype=int)

            N_inds, N_counts = np.unique(np.random.choice(kommune_ids, size=self.my.cfg.N_init, p=infected_per_kommune),  return_counts=True)
            R_inds, R_counts = np.unique(np.random.choice(kommune_ids, size=self.my.cfg.R_init, p=immunized_per_kommune), return_counts=True)

            N_kommune[N_inds] += N_counts
            R_kommune[R_inds] += R_counts


            initialization_subgroups = []

            # Loop over kommuner
            for kommune_id, N, R in zip(kommune_ids, N_kommune, R_kommune) :

                agents_in_kommune = np.array([agent for agent in possible_agents if self.my.kommune[agent] == kommune_id])

                # Check if any are to be infectd
                if N == 0 and R == 0 :
                    continue

                # Check if kommune is valid
                if len(agents_in_kommune) == 0:
                    warnings.warn(f"Agents selected for initialization in a kommune {kommune_id} : {self.kommune_dict['id_to_name'][kommune_id]} but no agents exists")
                    continue

                # Check if too many have been selected
                if len(agents_in_kommune) < (N + R) :
                    warnings.warn(f"{N+R} agents selected for initialization in a kommune {kommune_id} : {self.kommune_dict['id_to_name'][kommune_id]} but only {len(agents_in_kommune)} agents exists")
                    N = int(len(agents_in_kommune) * N / (N + R))
                    R = int(len(agents_in_kommune) * R / (N + R))

                # Determine the age distribution in the simulation
                ages_in_kommune = self.my.age[agents_in_kommune]
                agent_age_distribution = np.zeros(self.N_ages)
                age_inds, age_counts = np.unique(ages_in_kommune, return_counts=True)
                agent_age_distribution[age_inds] += age_counts

                # Adjust for age distribution of the populaiton
                prior_infected  = age_distribution_infected[ages_in_kommune]  / agent_age_distribution[ages_in_kommune]
                prior_immunized = age_distribution_immunized[ages_in_kommune] / agent_age_distribution[ages_in_kommune]

                # Convert to probability
                prior_infected  /= prior_infected.sum()
                prior_immunized /= prior_immunized.sum()

                kommune_beta    = self.my.cfg.label_betas[self.my.label[agents_in_kommune[0]]]
                kommune_UK_frac = self.my.cfg.label_frac[self.my.label[agents_in_kommune[0]]]

                initialization_subgroups.append((agents_in_kommune, N, R, prior_infected, prior_immunized, kommune_beta, kommune_UK_frac))

        else :

            # Determine the age distribution in the simulation
            ages = self.my.age[possible_agents]
            _, agent_age_distribution = np.unique(ages, return_counts=True)

            # Compute prior and adjust for age distribution of the populaiton
            prior_infected  = age_distribution_infected[ages]  / agent_age_distribution[ages]
            prior_immunized = age_distribution_immunized[ages] / agent_age_distribution[ages]

            # Convert to probability
            prior_infected  /= prior_infected.sum()
            prior_immunized /= prior_immunized.sum()

            initialization_subgroups = [(possible_agents, self.my.cfg.N_init, self.my.cfg.R_init, prior_infected, prior_immunized, self.my.cfg.label_betas[0], self.my.cfg.label_frac[0])]


        # Loop over subgroups and initialize
        for subgroup in tqdm(initialization_subgroups, total=len(initialization_subgroups), disable=(not self.verbose), position=0, leave=True) :

            agents_in_subgroup, N_subgroup, R_subgroup, prior_infected_subgroup, prior_immunized_subgroup, subgroup_beta_multiplier, subgroup_UK_frac = subgroup

            nb_simulation.initialize_states(
                self.my,
                self.g,
                self.intervention,
                self.state_total_counts,
                self.stratified_infection_counts,
                self.agents_in_state,
                subgroup_beta_multiplier,
                subgroup_UK_frac,
                agents_in_subgroup,
                N_subgroup,
                R_subgroup,
                prior_infected_subgroup,
                prior_immunized_subgroup,
                verbose=self.verbose)


        if check_distributions:
            ages = [self.my.age[agent] for agent in possible_agents if self.my.agent_is_infectious(agent)]
            _, dist = np.unique(ages, return_counts=True)
            dist = dist / dist.sum()

            print("Deviation of distribution for infected per age group (percentage points)")
            print(np.round(100 * (dist - age_distribution_infected), 1))


            ages = [self.my.age[agent] for agent in possible_agents if self.my.state[agent] == self.N_states - 1]
            _, dist = np.unique(ages, return_counts=True)
            dist = dist / dist.sum()

            print("Deviation of distribution for immunized per age group (percentage points)")
            print(np.round(100 * (dist - age_distribution_immunized), 1))


            if self.my.cfg.initialize_at_kommune_level :

                kommune = [self.my.kommune[agent] for agent in possible_agents if self.my.agent_is_infectious(agent)]
                dist = np.zeros(np.shape(infected_per_kommune))
                for k in kommune :
                    dist[k] += 1
                dist /= dist.sum()

                print("Deviation of distribution for infected per kommune (percentage points)")
                print(np.round(100 * (dist - infected_per_kommune), 1))

                if self.my.cfg.R_init > 0 :

                    kommune = [self.my.kommune[agent] for agent in possible_agents if self.my.state[agent] == self.N_states]
                    dist = np.zeros(np.shape(infected_per_kommune))
                    for k in kommune :
                        dist[k] += 1
                    dist /= dist.sum()

                    print("Deviation of distribution for immunized per kommune (percentage points)")
                    print(np.round(100 * (dist - immunized_per_kommune), 1))


    def run_simulation(self) :
        utils.set_numba_random_seed(utils.hash_to_seed(self.hash))

        if self.verbose :
            print("\nRUN SIMULATION")

        res = nb_simulation.run_simulation(
            self.my,
            self.g,
            self.intervention,
            self.state_total_counts,
            self.stratified_infection_counts,
            self.stratified_vaccination_counts,
            self.agents_in_state,
            self.nts,
            self.verbose)

        out_time, out_state_counts, out_stratified_infection_counts, out_stratified_vaccination_counts, out_my_state, intervention = res

        self.out_time = out_time
        self.my_state = np.array(out_my_state)
        self.df = utils.counts_to_df(out_time, out_state_counts, out_stratified_infection_counts, out_stratified_vaccination_counts, self.cfg)
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


    def _add_cfg_to_hdf5_file(self, f, cfg=None) :
        if cfg is None :
            cfg = self.cfg

        utils.add_cfg_to_hdf5_file(f, cfg)


    def _save_dataframe(self, save_csv=False, save_hdf5=True) :

        # Save CSV
        if save_csv :
            filename_csv = self._get_filename(name='ABM', filetype='csv')
            file_loaders.make_sure_folder_exist(filename_csv)
            self.df.to_csv(filename_csv, index=False)

        if save_hdf5 :
            filename_hdf5 = self._get_filename(name='ABM', filetype='hdf5')
            file_loaders.make_sure_folder_exist(filename_hdf5)
            with h5py.File(filename_hdf5, 'w', **hdf5_kwargs) as f :  #
                f.create_dataset('df', data=utils.dataframe_to_hdf5_format(self.df))
                self._add_cfg_to_hdf5_file(f)


    def _save_simulation_results(self, save_only_ID_0=False, time_elapsed=None) :

        if save_only_ID_0 and self.cfg.network.ID != 0 :
            return

        filename_hdf5 = self._get_filename(name="network", filetype="hdf5")
        file_loaders.make_sure_folder_exist(filename_hdf5)

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


    def save(self, save_csv=False, save_hdf5=True, save_only_ID_0=False, time_elapsed=None) :
        self._save_cfg()
        self._save_dataframe(save_csv=save_csv, save_hdf5=save_hdf5)
        self._save_simulation_results(save_only_ID_0=save_only_ID_0, time_elapsed=time_elapsed)




def run_single_simulation(
    cfg,
    verbose=False,
    force_rerun=False,
    only_initialize_network=False,
    save_initial_network=False,
    save_csv=False) :

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

        simulation.intialize_interventions()

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

    db_counts = np.array([db_cfg.count((q.hash == cfg.hash) & (q.network.ID == cfg.network.ID)) for cfg in cfgs_all])

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