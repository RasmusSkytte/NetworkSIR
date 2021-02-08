from numba.core.types.scalars import Boolean
import numpy as np
from pathlib import Path
import os
import numba as nb
from numba.experimental import jitclass
from numba import njit, typeof
from numba.typed import List, Dict
from numba.types import ListType, DictType, unicode_type

from src.utils import utils


@njit
def set_numba_random_seed(seed):
    np.random.seed(seed)


#%%

#      ██ ██ ████████      ██████ ██       █████  ███████ ███████ ███████ ███████
#      ██ ██    ██        ██      ██      ██   ██ ██      ██      ██      ██
#      ██ ██    ██        ██      ██      ███████ ███████ ███████ █████   ███████
# ██   ██ ██    ██        ██      ██      ██   ██      ██      ██ ██           ██
#  █████  ██    ██         ██████ ███████ ██   ██ ███████ ███████ ███████ ███████
#
# http://patorjk.com/software/taag/#p=display&f=ANSI%20Regular&t=Version%202%0A%20

#%%

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # cfg Config file # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

spec_cfg = {
    # Default parameters
    "version": nb.float32,
    "N_tot": nb.uint32,
    "beta": nb.float32,
    "sigma_beta": nb.float32,
    "beta_connection_type": nb.float32[:],
    "algo": nb.uint8,
    "N_init": nb.uint16,
    "N_init_UK_frac": nb.float64,
    "lambda_E": nb.float32,
    "lambda_I": nb.float32,
    # other
    "day_max": nb.float32,
    "make_random_initial_infections": nb.boolean,
    "weighted_random_initial_infections": nb.boolean,
    "make_initial_infections_at_kommune": nb.boolean,
    "make_restrictions_at_kommune_level": nb.boolean,
    "clustering_connection_retries": nb.uint32,
    "beta_UK_multiplier": nb.float32,
    "outbreak_position_UK": nb.types.unicode_type,
    "burn_in": nb.int64,
    "start_date_offset" : nb.int64,
    "days_of_vacci_start": nb.int64,
    # events
    "N_events": nb.uint16,
    "event_size_max": nb.uint16,
    "event_size_mean": nb.float32,
    "event_beta_scaling": nb.float32,
    "event_weekend_multiplier": nb.float32,
    # lockdown-related / interventions
    "do_interventions": nb.boolean,
    "threshold_type": nb.int8, # which thing set off restrictions: 0: certain date. 1: "real" incidens rate 2: measured incidens rate
    "restriction_thresholds": nb.int64[:], # len == 2*nr of different thresholds, on the form [start stop start stop etc.]
    "threshold_interventions_to_apply": ListType(nb.int64),
    "list_of_threshold_interventions_effects": nb.float64[:, :, :],
    "continuous_interventions_to_apply": ListType(nb.int64),
    "f_daily_tests": nb.float32,
    "test_delay_in_clicks": nb.int64[:],
    "results_delay_in_clicks": nb.int64[:],
    "chance_of_finding_infected": nb.float64[:],
    "days_looking_back": nb.int64,
    #"masking_rate_reduction": nb.float64[:, ::1],  # to make the type C instead if A
    #"lockdown_rate_reduction": nb.float64[:, ::1],  # to make the type C instead if A
    "isolation_rate_reduction": nb.float64[:],
    "tracking_rates": nb.float64[:],
    "tracking_delay": nb.int64,
    "intervention_removal_delay_in_clicks": nb.int32,
    # ID
    "ID": nb.uint16,
}

@jitclass(spec_cfg)
class Config(object):
    def __init__(self):

        # Default parameters
        self.version = 2.0
        self.N_tot = 580_000
        self.beta = 0.01
        self.sigma_beta = 0.0
        self.algo = 2
        self.N_init = 100
        self.N_init_UK_frac = 0
        self.lambda_E = 1.0
        self.lambda_I = 1.0

        # other
        self.make_random_initial_infections = True
        self.weighted_random_initial_infections = False
        self.make_initial_infections_at_kommune = False
        self.make_restrictions_at_kommune_level = True
        self.day_max = 0
        self.clustering_connection_retries = 0
        self.beta_UK_multiplier = 1.0
        self.burn_in = 20 # burn in period, -int how many days the sim shall run before

        # events
        self.N_events = 0
        self.event_size_max = 0
        self.event_size_mean = 50
        self.event_beta_scaling = 10
        self.event_weekend_multiplier = 1.0
        # Interventions / Lockdown
        self.do_interventions = True

        self.ID = 0


nb_cfg_type = Config.class_type.instance_type

spec_network = {
    # Default parameters
    "rho": nb.float32,
    "epsilon_rho": nb.float32,
    "mu": nb.float32,
    "sigma_mu": nb.float32,
    "contact_matrices_name": nb.types.unicode_type,
    "work_matrix": nb.float64[:, :],
    "other_matrix": nb.float64[:, :],
    "work_other_ratio": nb.float32,  # 0.2 = 20% work, 80% other
    "N_contacts_max": nb.uint16,
}

@jitclass(spec_network)
class Network(object):
    def __init__(self):

        # Default parameters
        self.rho = 0.0
        self.epsilon_rho = 0.04
        self.mu = 40.0
        self.sigma_mu = 0.0
        self.work_matrix  = np.ones((8, 8), dtype=np.float64)
        self.other_matrix = np.ones((8, 8), dtype=np.float64)
        self.work_other_ratio = 0.5
        self.N_contacts_max = 0
        
nb_cfg_network_type = Network.class_type.instance_type

def initialize_nb_cfg(obj, cfg, spec):
    for key, val in cfg.items():
        if isinstance(val, list):
            if isinstance(spec[key], nb.types.ListType):
                val = List(val)
            elif isinstance(spec[key], nb.types.Array):
                val = np.array(val, dtype=spec[key].dtype.name)
            else:
                raise AssertionError(f"Got {key}: {val}, not working")
        elif isinstance(val, dict) :
            continue
        setattr(obj, key, val)
    return obj



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # My object # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#%%

spec_my = {
    "age": nb.uint8[:],
    "connections": ListType(ListType(nb.uint32)),
    "connections_type": ListType(ListType(nb.uint8)),
    "beta_connection_type": nb.float32[:],
    "coordinates": nb.float32[:, :],
    "connection_weight": nb.float32[:],
    "infection_weight": nb.float64[:],
    "number_of_contacts": nb.uint16[:],
    "state": nb.int8[:],
    "tent": nb.uint16[:],
    "kommune": nb.uint8[:],
    "infectious_states": ListType(nb.int64),
    "corona_type": nb.uint8[:],
    "vaccination_type": nb.uint8[:],
    "restricted_status": nb.uint8[:],
    "cfg": nb_cfg_type,
    "cfg_network": nb_cfg_network_type,
}


# "Nested/Mutable" Arrays are faster than list of arrays which are faster than lists of lists
@jitclass(spec_my)
class My(object):
    def __init__(self, nb_cfg, nb_cfg_network):
        N_tot = nb_cfg.N_tot
        self.age = np.zeros(N_tot, dtype=np.uint8)
        self.coordinates = np.zeros((N_tot, 2), dtype=np.float32)
        self.connections = utils.initialize_nested_lists(N_tot, np.uint32)
        self.connections_type = utils.initialize_nested_lists(N_tot, np.uint8)
        self.beta_connection_type = np.array(
            [3.0, 1.0, 1.0, 1.0], dtype=np.float32
        )  # beta multiplier for [House, work, others, events]
        self.connection_weight = np.ones(N_tot, dtype=np.float32)
        self.infection_weight = np.ones(N_tot, dtype=np.float64)
        self.number_of_contacts = np.zeros(N_tot, dtype=nb.uint16)
        self.state = np.full(N_tot, fill_value=-1, dtype=np.int8)
        self.tent = np.zeros(N_tot, dtype=np.uint16)
        self.kommune = np.zeros(N_tot, dtype=np.uint8)
        self.infectious_states = List([4, 5, 6, 7])
        self.corona_type = np.zeros(N_tot, dtype=np.uint8)
        self.vaccination_type = np.zeros(N_tot, dtype=np.uint8)
        self.restricted_status = np.zeros(N_tot, dtype=np.uint8)
        self.cfg = nb_cfg
        self.cfg_network = nb_cfg_network

    def dist(self, agent1, agent2):
        point1 = self.coordinates[agent1]
        point2 = self.coordinates[agent2]
        return utils.haversine_scipy(point1, point2)

    def dist_coordinate(self, agent, coordinate):
        return utils.haversine_scipy(self.coordinates[agent], coordinate[::-1])

    def dist_accepted(self, agent1, agent2, rho_tmp):
        if np.exp(-self.dist(agent1, agent2) * rho_tmp) > np.random.rand():
            return True
        else:
            return False

    def dist_accepted_coordinate(self, agent, coordinate, rho_tmp):
        if np.exp(-self.dist_coordinate(agent, coordinate) * rho_tmp) > np.random.rand():
            return True
        else:
            return False

    def agent_is_susceptable(self, agent):
        return self.state[agent] == -1

    def agent_is_infectious(self, agent):
        return self.state[agent] in self.infectious_states

    def agent_is_not_infectious(self, agent):
        return not self.agent_is_infectious(agent)


def initialize_My(cfg):
    nb_cfg         = initialize_nb_cfg(Config(),  cfg,         spec_cfg)
    nb_cfg_network = initialize_nb_cfg(Network(), cfg.network, spec_network)
    return My(nb_cfg, nb_cfg_network)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # Gillespie Algorithm # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#%%

spec_g = {
    "N_tot": nb.uint32,
    "N_states": nb.uint8,
    "total_sum": nb.float64,
    "total_sum_infections": nb.float64,
    "total_sum_of_state_changes": nb.float64,
    "cumulative_sum": nb.float64,
    "cumulative_sum_of_state_changes": nb.float64[:],
    "cumulative_sum_infection_rates": nb.float64[:],
    "rates": ListType(nb.float64[::1]),  # ListType[array(float64, 1d, C)] (C vs. A)
    "sum_of_rates": nb.float64[:],
}


@jitclass(spec_g)
class Gillespie(object):
    def __init__(self, my, N_states):
        self.N_states = N_states
        self.total_sum = 0.0
        self.total_sum_infections = 0.0
        self.total_sum_of_state_changes = 0.0
        self.cumulative_sum = 0.0
        self.cumulative_sum_of_state_changes = np.zeros(N_states, dtype=np.float64)
        self.cumulative_sum_infection_rates = np.zeros(N_states, dtype=np.float64)
        self._initialize_rates(my)

    def _initialize_rates(self, my):
        rates = List()
        for i in range(my.cfg.N_tot):
            # x = np.full(
            #     shape=my.number_of_contacts[i],
            #     fill_value=my.infection_weight[i],
            #     dtype=np.float64,
            # )
            x = np.zeros(my.number_of_contacts[i])
            for j in range(my.number_of_contacts[i]):
                x[j] = my.beta_connection_type[my.connections_type[i][j]] * my.infection_weight[i]

            rates.append(x)
        self.rates = rates
        self.sum_of_rates = np.zeros(my.cfg.N_tot, dtype=np.float64)

    def update_rates(self, my, rate, agent):
        self.total_sum_infections += rate
        self.sum_of_rates[agent] += rate
        self.cumulative_sum_infection_rates[my.state[agent] :] += rate


#%%

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # Intervention Class  # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

spec_intervention = {
    "cfg": nb_cfg_type,
    "labels": nb.uint8[:],  # affilitation? XXX
    "label_counter": nb.uint32[:],
    "N_labels": nb.uint32,
    "freedom_impact": nb.float64[:],
    "freedom_impact_list": ListType(nb.float64),
    "R_true_list": ListType(nb.float64),
    "R_true_list_brit": ListType(nb.float64),
    "day_found_infected": nb.int32[:],
    "reason_for_test": nb.int8[:],
    "positive_test_counter": nb.uint32[:],
    "clicks_when_tested": nb.int32[:],
    "clicks_when_tested_result": nb.int32[:],
    "clicks_when_isolated": nb.int32[:],
    "clicks_when_restriction_stops": nb.int32[:],
    "types": nb.uint8[:],
    "started_as": nb.uint8[:],
    "vaccinations_per_age_group": nb.int64[:, :],
    "vaccination_schedule": nb.int64[:],
    "work_matrix_restrict": nb.float64[:, :],
    "other_matrix_restrict": nb.float64[:, :],

    "verbose": nb.boolean,

}


@jitclass(spec_intervention)
class Intervention(object):
    """
    - N_labels: Number of labels. "Label" here can refer to either tent or kommune.
    - labels: a label or ID which is either the nearest tent or the kommune which the agent belongs to
    - label_counter: count how many agent belong to a particular label

    - day_found_infected: -1 if not infected, otherwise the day of infection

    - reason_for_test:
         0: symptoms
         1: random_test
         2: tracing,
        -1: No reason yet (or no impending tests). You can still be tested again later on (if negative test)

    - positive_test_counter: counter of how many were found tested positive due to reasom 0, 1 or 2

    - clicks_when_tested: When you were tested measured in clicks (10 clicks = 1 day)

    - clicks_when_tested_result: When you get your test results measured in clicks

    - clicks_when_isolated: when you were told to go in isolation and be tested

    - threshold_interventions: array to keep count of which intervention are at place at which label
        0: Do nothing
        1: lockdown (cut some contacts and reduce the rest),
        2: Masking (reduce some contacts),
        3: Matrix based (used loaded contact matrices. )

    - continuous_interventions: array to keep count of which intervention are at place at which label
        # 0: Do nothing
        # 1: Tracking (infected and their connections)
        # 2: Test people with symptoms
        # 3: Isolate
        # 4: Random Testing
        # 5: vaccinations

    - started_as: describes whether or not an intervention has been applied. If 0, no intervention has been applied.

    - verbose: Prints status of interventions and removal of them

    """

    def __init__(
        self,
        nb_cfg,
        labels,
        vaccinations_per_age_group,
        vaccination_schedule,
        work_matrix_restrict,
        other_matrix_restrict,
        verbose=False,
    ):

        self.cfg = nb_cfg

        self._initialize_labels(labels)

        self.day_found_infected = np.full(self.cfg.N_tot, fill_value=-1, dtype=np.int32)
        self.freedom_impact = np.full(self.cfg.N_tot, fill_value=0.0, dtype=np.float64)
        self.freedom_impact_list = List([0.0])
        self.R_true_list = List([0.0])
        self.R_true_list_brit = List([0.0])
        self.reason_for_test = np.full(self.cfg.N_tot, fill_value=-1, dtype=np.int8)
        self.positive_test_counter = np.zeros(3, dtype=np.uint32)
        self.clicks_when_tested = np.full(self.cfg.N_tot, fill_value=-1, dtype=np.int32)
        self.clicks_when_tested_result = np.full(self.cfg.N_tot, fill_value=-1, dtype=np.int32)
        self.clicks_when_isolated = np.full(self.cfg.N_tot, fill_value=-1, dtype=np.int32)
        self.clicks_when_restriction_stops = np.full(self.N_labels, fill_value=-1, dtype=np.int32)
        self.types = np.zeros(self.N_labels, dtype=np.uint8)
        self.started_as = np.zeros(self.N_labels, dtype=np.uint8)
        self.vaccinations_per_age_group = vaccinations_per_age_group
        self.vaccination_schedule = vaccination_schedule
        self.work_matrix_restrict = work_matrix_restrict
        self.other_matrix_restrict = other_matrix_restrict

        self.verbose = verbose

    def _initialize_labels(self, labels):
        self.labels = np.asarray(labels, dtype=np.uint8)
        unique, counts = utils.numba_unique_with_counts(labels)
        self.label_counter = np.asarray(counts, dtype=np.uint32)
        self.N_labels = len(unique)

    def agent_has_not_been_tested(self, agent):
        return self.day_found_infected[agent] == -1

    def agent_has_been_tested(self, agent):
        return not self.agent_has_not_been_tested(agent)

    @property
    def apply_interventions(self):
        return self.cfg.do_interventions

    @property
    def apply_interventions_on_label(self):
        return (
            (1 in self.cfg.threshold_interventions_to_apply)
            or (2 in self.cfg.threshold_interventions_to_apply)
            or (3 in self.cfg.threshold_interventions_to_apply)
        )

    @property
    def apply_tracking(self):
        return 1 in self.cfg.continuous_interventions_to_apply

    @property
    def apply_symptom_testing(self):
        return 2 in self.cfg.continuous_interventions_to_apply

    @property
    def apply_isolation(self):
        return 3 in self.cfg.continuous_interventions_to_apply

    @property
    def apply_random_testing(self):
        return 4 in self.cfg.continuous_interventions_to_apply

    @property
    def apply_matrix_restriction(self):
        return 3 in self.cfg.threshold_interventions_to_apply

    @property
    def apply_vaccinations(self):
        return 5 in self.cfg.continuous_interventions_to_apply

    @property
    def start_interventions_by_day(self):
        return self.cfg.threshold_type == 0

    @property
    def start_interventions_by_real_incidens_rate(self):
        return self.cfg.threshold_type == 1

    @property
    def start_interventions_by_meassured_incidens_rate(self):
        return self.cfg.threshold_type == 2

#%%
# ██    ██ ███████ ██████  ███████ ██  ██████  ███    ██      ██
# ██    ██ ██      ██   ██ ██      ██ ██    ██ ████   ██     ███
# ██    ██ █████   ██████  ███████ ██ ██    ██ ██ ██  ██      ██
#  ██  ██  ██      ██   ██      ██ ██ ██    ██ ██  ██ ██      ██
#   ████   ███████ ██   ██ ███████ ██  ██████  ██   ████      ██
#
#%%


@njit
def v1_initialize_my(my, coordinates_raw):
    for agent in range(my.cfg.N_tot):
        set_connection_weight(my, agent)
        set_infection_weight(my, agent)
        my.coordinates[agent] = coordinates_raw[agent]


@njit
def v1_run_algo_1(my, PP, rho_tmp):
    """ Algo 1: density independent connection algorithm """
    agent1 = np.uint32(np.searchsorted(PP, np.random.rand()))
    while True:
        agent2 = np.uint32(np.searchsorted(PP, np.random.rand()))
        rho_tmp *= 0.9995
        do_stop = update_node_connections(
            my,
            rho_tmp,
            agent1,
            agent2,
            connection_type=-1,
            code_version=1,
        )
        if do_stop:
            break


@njit
def v1_run_algo_2(my, PP, rho_tmp):
    """ Algo 2: increases number of connections in high-density ares """
    while True:
        agent1 = np.uint32(np.searchsorted(PP, np.random.rand()))
        agent2 = np.uint32(np.searchsorted(PP, np.random.rand()))
        do_stop = update_node_connections(
            my,
            rho_tmp,
            agent1,
            agent2,
            connection_type=-1,
            code_version=1,
        )
        if do_stop:
            break


@njit
def v1_connect_nodes(my):
    """ v1 of connecting nodes. No age dependence, and a specific choice of Algo """
    if my.cfg.algo == 2:
        run_algo = v1_run_algo_2
    else:
        run_algo = v1_run_algo_1
    PP = np.cumsum(my.connection_weight) / np.sum(my.connection_weight)
    for _ in range(my.cfg.mu / 2 * my.cfg.N_tot):
        if np.random.rand() > my.cfg.epsilon_rho:
            rho_tmp = my.cfg.rho
        else:
            rho_tmp = 0.0
        run_algo(my, PP, rho_tmp)


#%%
# ██    ██ ███████ ██████  ███████ ██  ██████  ███    ██     ██████
# ██    ██ ██      ██   ██ ██      ██ ██    ██ ████   ██          ██
# ██    ██ █████   ██████  ███████ ██ ██    ██ ██ ██  ██      █████
#  ██  ██  ██      ██   ██      ██ ██ ██    ██ ██  ██ ██     ██
#   ████   ███████ ██   ██ ███████ ██  ██████  ██   ████     ███████
#
#%%

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # INITIALIZATION  # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@njit
def set_connection_weight(my, agent):
    """ How introvert / extrovert you are. How likely you are at having many contacts in your network.
        Function is used determine the distribution of number of contacts. A larger my.cfg.sigma_mu gives a larger variance in number of contacts.
        Parameters:
            my (class): Class of parameters describing the system
            agent (int): ID of agent
    """
    if np.random.rand() < my.cfg_network.sigma_mu:
        my.connection_weight[agent] = -np.log(np.random.rand())
    else:
        my.connection_weight[agent] = 1.0


@njit
def set_infection_weight(my, agent):
    """ How much of a super sheader are you?
        Function is used determine the distribution of number of contacts. A larger my.cfg.sigma_beta gives a larger variance individual betas
        if my.cfg.sigma_beta == 0 everybody is equally infectious.
        Parameters:
            my (class): Class of parameters describing the system
            agent (int): ID of agent
    """
    if np.random.rand() < my.cfg.sigma_beta:
        my.infection_weight[agent] = -np.log(np.random.rand()) * my.cfg.beta
    else:
        my.infection_weight[agent] = my.cfg.beta


@njit
def computer_number_of_cluster_retries(my, agent1, agent2):
    """ Number of times to (re)try to connect two agents. Function is used to cluster the network more.
        A higher my.cfg.clustering_connection_retries gives higher cluster coefficient.
        Parameters:
            my (class): Class of parameters describing the system
            agent1 (int): ID of first agent
            agent2 (int): ID of second agent
        returns:
           connectivity_factor (int): Number of tries to connect to agents.
    """
    connectivity_factor = 1
    for contact in my.connections[agent1]:
        if contact in my.connections[agent2]:
            connectivity_factor += my.cfg.clustering_connection_retries
    return connectivity_factor


@njit
def cluster_retry_succesfull(my, agent1, agent2, rho_tmp):
    """" (Re)Try to connect two agents. Returns True if succesful, else False.
        Parameters:
            my (class): Class of parameters describing the system
            agent1 (int): ID of first agent
            agent2 (int): ID of second agent
            rho_tmp (float): Characteristic distance of connections
        returns:
           Bool: Is any on the (re)tries succesfull

    """
    if my.cfg.clustering_connection_retries == 0:
        return False
    connectivity_factor = computer_number_of_cluster_retries(my, agent1, agent2)
    for _ in range(connectivity_factor):
        if my.dist_accepted(agent1, agent2, rho_tmp):
            return True
    return False


@njit
def update_node_connections(
    my,
    rho_tmp,
    agent1,
    agent2,
    connection_type,
    code_version=2,
):
    """ Returns True if two agents should be connected, else False
        Parameters:
            my (class): Class of parameters describing the system
            agent1 (int): ID of first agent
            agent2 (int): ID of second agent
            rho_tmp (float): Characteristic distance of connections
            connection_type (int): ID for connection type ([House, work, other])

    """

    # checks if the two agents are the same, if they are, they can not be connected
    if agent1 == agent2:
        return False

    # Tries to connect to agents, depending on distance.
    # if rho_tmp == 0 all distance are accepted (well mixed approximation)
    # TODO: add connection weights
    dist_accepted = rho_tmp == 0 or my.dist_accepted(agent1, agent2, rho_tmp)
    if not dist_accepted:
        # try and reconnect to increase clustering effect
        if not cluster_retry_succesfull(my, agent1, agent2, rho_tmp):
            return False

    # checks if the two agents are already connected
    already_added = agent1 in my.connections[agent2] or agent2 in my.connections[agent1]
    if already_added:
        return False

    #checks if one contact have exceeded the contact limit. Default is no contact limit. This check is incorporated to see effect of extreme tails in N_contact distribution
    N_contacts_max = my.cfg_network.N_contacts_max
    maximum_contacts_exceeded = (N_contacts_max > 0) and (
        (len(my.connections[agent1]) >= N_contacts_max)
        or (len(my.connections[agent2]) >= N_contacts_max)
    )
    if maximum_contacts_exceeded:
        return False

    #Store the connection
    my.connections[agent1].append(np.uint32(agent2))
    my.connections[agent2].append(np.uint32(agent1))

    # store connection type
    if code_version >= 2:
        connection_type = np.uint8(connection_type)
        my.connections_type[agent1].append(connection_type)
        my.connections_type[agent2].append(connection_type)

    # keep track of number of contacts
    my.number_of_contacts[agent1] += 1
    my.number_of_contacts[agent2] += 1

    return True


@njit
def place_and_connect_families(
    my, people_in_household, age_distribution_per_people_in_household, coordinates_raw
):
    """ Place agents into household, including assigning coordinates and making connections. First step in making the network.
        Parameters:
            my (class): Class of parameters describing the system
            people_in_household (list): distribution of number of people in households. Input data from file - source: danish statistics
            age_distribution_per_people_in_household (list): Age distribution of households as a function of number of people in household. Input data from file - source: danish statistics
            coordinates_raw: list of coordinates drawn from population density distribution. Households are placed at these coordinates
        returns:
            mu_counter (int): How many connections are made in households
            counter_ages(list): Number of agents in each age group
            agents_in_age_group(nested list): Which agents are in each age group

    """
    N_tot = my.cfg.N_tot

    #Shuffle indicies
    all_indices = np.arange(N_tot, dtype=np.uint32)
    np.random.shuffle(all_indices)

    N_dim_people_in_household, N_ages = age_distribution_per_people_in_household.shape
    assert N_dim_people_in_household == len(people_in_household)
    people_index_to_value = np.arange(1, N_dim_people_in_household + 1)

    #initialize lists to keep track of number of agents in each age group
    counter_ages = np.zeros(N_ages, dtype=np.uint32)
    agents_in_age_group = utils.initialize_nested_lists(N_ages, dtype=np.uint32)

    mu_counter = 0
    agent = 0
    do_continue = True
    while do_continue:

        agent0 = agent

        house_index = all_indices[agent]

        #Draw size of household form distribution
        N_people_in_house_index = utils.rand_choice_nb(people_in_household)
        N_people_in_house = people_index_to_value[N_people_in_house_index]

        # if N_in_house would increase agent to over N_tot,
        # set N_people_in_house such that it fits and break loop
        if agent + N_people_in_house >= N_tot:
            N_people_in_house = N_tot - agent
            do_continue = False

        # Initilaze the agents and assign them to households
        for _ in range(N_people_in_house):

            age_index = utils.rand_choice_nb(
                age_distribution_per_people_in_household[N_people_in_house_index]
            )

            #set age for agent
            age = age_index  # just use age index as substitute for age
            my.age[agent] = age
            counter_ages[age_index] += 1
            agents_in_age_group[age_index].append(np.uint32(agent))

            #set coordinate for agent
            my.coordinates[agent] = coordinates_raw[house_index]

            # set weights determining extro/introvert and supersheader
            set_connection_weight(my, agent)
            set_infection_weight(my, agent)

            agent += 1

        # add agents to each others networks (connections). All people in a household know eachother
        for agent1 in range(agent0, agent0 + N_people_in_house):
            for agent2 in range(agent1, agent0 + N_people_in_house):
                if agent1 != agent2:
                    my.connections[agent1].append(np.uint32(agent2))
                    my.connections[agent2].append(np.uint32(agent1))
                    my.connections_type[agent1].append(np.uint8(0))
                    my.connections_type[agent2].append(np.uint8(0))
                    my.number_of_contacts[agent1] += 1
                    my.number_of_contacts[agent2] += 1
                    mu_counter += 1

    agents_in_age_group = utils.nested_lists_to_list_of_array(agents_in_age_group)

    return mu_counter, counter_ages, agents_in_age_group

@njit
def place_and_connect_families_kommune_specific(
    my, people_in_household, age_distribution_per_people_in_household, coordinates_raw, Kommune_ids, N_ages
):
    """ Place agents into household, including assigning coordinates and making connections. First step in making the network.
        Parameters:
            my (class): Class of parameters describing the system
            people_in_household (list): distribution of number of people in households. Input data from file - source: danish statistics
            age_distribution_per_people_in_household (list): Age distribution of households as a function of number of people in household. Input data from file - source: danish statistics
            coordinates_raw: list of coordinates drawn from population density distribution. Households are placed at these coordinates
        returns:
            mu_counter (int): How many connections are made in households
            counter_ages(list): Number of agents in each age group
            agents_in_age_group(nested list): Which agents are in each age group

    """
    N_tot = my.cfg.N_tot

    #Shuffle indicies
    all_indices = np.arange(N_tot, dtype=np.uint32)
    np.random.shuffle(all_indices)
    people_index_to_value = np.arange(1, 7) # household are between 1-6 people

    #initialize lists to keep track of number of agents in each age group
    counter_ages = np.zeros(N_ages, dtype=np.uint32)
    agents_in_age_group = utils.initialize_nested_lists(N_ages, dtype=np.uint32)
    house_sizes = np.zeros(len(people_index_to_value), dtype=np.int64)
    mu_counter = 0
    agent = 0
    do_continue = True
    while do_continue:

        agent0 = agent
        house_index = all_indices[agent]
        coordinates = coordinates_raw[house_index]
        kommune = Kommune_ids[house_index]

        #Draw size of household form distribution
        people_in_household_kom = people_in_household[kommune,:]
        N_people_in_house_index = utils.rand_choice_nb(people_in_household_kom)
        N_people_in_house = people_index_to_value[N_people_in_house_index]
        house_sizes[N_people_in_house_index] += 1
        # if N_in_house would increase agent to over N_tot,
        # set N_people_in_house such that it fits and break loop
        if agent + N_people_in_house >= N_tot:
            N_people_in_house = N_tot - agent
            do_continue = False

        # Initilaze the agents and assign them to households
        age_dist = age_distribution_per_people_in_household[kommune, N_people_in_house_index,:]
        for _ in range(N_people_in_house):
            age_index = utils.rand_choice_nb(
                age_dist
            )

            #set age for agent
            age = age_index  # just use age index as substitute for age
            my.age[agent] = age
            counter_ages[age_index] += 1
            agents_in_age_group[age_index].append(np.uint32(agent))

            #set coordinate for agent
            my.coordinates[agent] = coordinates_raw[house_index]

            # set weights determining extro/introvert and supersheader
            set_connection_weight(my, agent)
            set_infection_weight(my, agent)

            agent += 1

        # add agents to each others networks (connections). All people in a household know eachother
        for agent1 in range(agent0, agent0 + N_people_in_house):
            for agent2 in range(agent1, agent0 + N_people_in_house):
                if agent1 != agent2:
                    my.connections[agent1].append(np.uint32(agent2))
                    my.connections[agent2].append(np.uint32(agent1))
                    my.connections_type[agent1].append(np.uint8(0))
                    my.connections_type[agent2].append(np.uint8(0))
                    my.number_of_contacts[agent1] += 1
                    my.number_of_contacts[agent2] += 1
                    mu_counter += 1

    agents_in_age_group = utils.nested_lists_to_list_of_array(agents_in_age_group)
    print(house_sizes)
    return mu_counter, counter_ages, agents_in_age_group

@njit
def run_algo_work(my, agents_in_age_group, age1, age2, rho_tmp):
    """ Make connection of work type. Algo locks choice of agent1, and then tries different agent2's until one is accepted.
        This algorithm gives an equal number of connections independent of local population density.
        The sssumption here is that the size of peoples workplaces is independent on where they live.
        Parameters:
            my (class): Class of parameters describing the system
            agents_in_age_group (nested list): list of which agents are in which age groups.
            age1 (int): Which age group should agent1 be drawn from.
            age2 (int): Which age group should agent2 be drawn from.
            rho_tmp(float): characteristic distance parameter
    """

    # TODO: Add connection weights
    agent1 = np.random.choice(agents_in_age_group[age1])

    while True:
        agent2 = np.random.choice(agents_in_age_group[age2])
        rho_tmp *= 0.9995 # lowers the threshold for accepting for each try, primarily used to make sure small simulations terminate.
        do_stop = update_node_connections(
            my,
            rho_tmp,
            agent1,
            agent2,
            connection_type=1,
            code_version=2,
        )

        if do_stop:
            break


@njit
def run_algo_other(my, agents_in_age_group, age1, age2, rho_tmp):
    """ Make connection of other type. Algo tries different combinations of agent1 and agent2 until one combination is accepted.
        This algorithm gives more connections to people living in high populations densitity areas. This is the main driver of outbreaks being stronger in cities.
        Assumption is that you meet more people if you live in densely populated areas.
        Parameters:
            my (class): Class of parameters describing the system
            agents_in_age_group (nested list): list of which agents are in which age groups.
            age1 (int): Which age group should agent1 be drawn from.
            age2 (int): Which age group should agent2 be drawn from.
            rho_tmp(float): characteristic distance parameter
    """
    while True:

        # TODO: Add connection weights
        agent1 = np.random.choice(agents_in_age_group[age1])
        agent2 = np.random.choice(agents_in_age_group[age2])
        do_stop = update_node_connections(
            my,
            rho_tmp,
            agent1,
            agent2,
            connection_type=2,
            code_version=2,
        )
        if do_stop:
            break


@njit
def find_two_age_groups(N_ages, matrix):
    """ Find two ages from an age connections matrix.
        Parameters:
            N_ages(int): Number of age groups, default 9 (TODO: Should this be 10?)
            matrix: Connection matrix, how often does different age group interact.
    """
    a = 0
    ra = np.random.rand()
    for i in range(N_ages):
        for j in range(N_ages):
            a += matrix[i, j]
            if a > ra:
                age1, age2 = i, j
                return age1, age2
    raise AssertionError("find_two_age_groups couldn't find two age groups")


@njit
def connect_work_and_others(
    my,
    N_ages,
    mu_counter,
    matrix_work,
    matrix_other,
    agents_in_age_group,
    verbose=True,
):
    """ Overall loop to make all non household connections.
        Parameters:
            my (class): Class of parameters describing the system
            N_ages(int): Number of age groups, default 9 (TODO: Should this be 10?)
            matrix_work: Connection matrix, how often does different age group interact at workplaces. Combination of school and work.
            matrix_other: Connection matrix, how often does different age group interact in other section.
            agents_in_age_group(nested list): list of which agents are in which age groups
            verbose: prints to terminal, how far the process of connecting the network is.
    """
    progress_delta_print = 0.1  # 10 percent
    progress_counter = 1

    matrix_work  = matrix_work / matrix_work.sum()
    matrix_other  = matrix_other / matrix_other.sum()
    mu_tot = my.cfg_network.mu / 2 * my.cfg.N_tot # total number of connections in the network, when done
    while mu_counter < mu_tot: # continue until all connections are made

        # determining if next connections is work or other.
        ra_work_other = np.random.rand()
        if ra_work_other < my.cfg_network.work_other_ratio:
            matrix = matrix_work
            run_algo = run_algo_work
        else:
            matrix = matrix_other
            run_algo = run_algo_other

        #draw ages from connectivity matrix
        age1, age2 = find_two_age_groups(N_ages, matrix)

        #some number small number of connections is independent on distance. eg. you now some people on the other side of the country. Data is based on pendling data.
        if np.random.rand() > my.cfg_network.epsilon_rho:
            rho_tmp = my.cfg_network.rho
        else:
            rho_tmp = 0.0

        #make connection
        run_algo(
            my,
            agents_in_age_group,
            age1,
            age2,
            rho_tmp,
        )

        mu_counter += 1
        if verbose:
            progress = mu_counter / mu_tot
            if progress > progress_counter * progress_delta_print:
                progress_counter += 1
                print("Connected ", round(progress * 100), r"% of work and others")


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # INITIAL INFECTIONS  # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@njit
def single_random_choice(x):
    return np.random.choice(x, size=1)[0]


@njit
def set_to_array(input_set):
    out = List()
    for s in input_set:
        out.append(s)
    return np.asarray(out)


@njit
def nb_random_choice(arr, prob, size=1, replace=False):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :param size: Integer describing the size of the output.
    :return: A random sample from the given array with a given probability.
    """

    assert len(arr) == len(prob)
    assert size < len(arr)

    prob = prob / np.sum(prob)
    if replace:
        ra = np.random.random(size=size)
        idx = np.searchsorted(np.cumsum(prob), ra, side="right")
        return arr[idx]
    else:
        if size / len(arr) > 0.5:
            print(
                "Warning: choosing more than 50% of the input array with replacement, can be slow."
            )
        out = set()
        while len(out) < size:
            ra = np.random.random()
            idx = np.searchsorted(np.cumsum(prob), ra, side="right")
            x = arr[idx]
            if not x in out:
                out.add(x)
        return set_to_array(out)

@njit
def exp_func(x, a, b, c):

    return a * np.exp(b * x) + c
@njit
def make_random_initial_infections(my, possible_agents):
    if my.cfg.weighted_random_initial_infections:
        probs =np.array([7.09189651e+00, 7.21828639e+00, 7.35063322e+00, 7.48921778e+00,
           7.63433406e+00, 7.78628991e+00, 7.94540769e+00, 8.11202496e+00,
           8.28649517e+00, 8.46918845e+00, 8.66049237e+00, 8.86081276e+00,
           9.07057458e+00, 9.29022282e+00, 9.52022345e+00, 9.76106439e+00,
           1.00132566e+01, 1.02773350e+01, 1.05538598e+01, 1.08434177e+01,
           1.11466230e+01, 1.14641189e+01, 1.17965788e+01, 1.21447081e+01,
           1.25092454e+01, 1.28909638e+01, 1.32906734e+01, 1.37092218e+01,
           1.41474972e+01, 1.46064292e+01, 1.50869914e+01, 1.55902033e+01,
           1.61171325e+01, 1.66688966e+01, 1.72466664e+01, 1.78516673e+01,
           1.84851830e+01, 1.91485573e+01, 1.98431976e+01, 2.05705774e+01,
           2.13322397e+01, 2.21298006e+01, 2.29649517e+01, 2.38394649e+01,
           2.47551954e+01, 2.57140858e+01, 2.67181702e+01, 2.77695789e+01,
           2.88705421e+01, 3.00233956e+01, 3.12305849e+01, 3.24946711e+01,
           3.38183358e+01, 3.52043869e+01, 3.66557650e+01, 3.81755489e+01,
           3.97669627e+01, 4.14333825e+01, 4.31783434e+01, 4.50055472e+01,
           4.69188701e+01, 4.89223712e+01, 5.10203005e+01, 5.32171087e+01,
           5.55174561e+01, 5.79262227e+01, 6.04485185e+01, 6.30896942e+01,
           6.58553529e+01, 6.87513617e+01, 7.17838642e+01, 7.49592935e+01,
           7.82843861e+01, 8.17661957e+01, 8.54121088e+01, 8.92298599e+01,
           9.32275478e+01, 9.74136534e+01, 1.01797057e+02, 1.06387058e+02,
           1.11193393e+02, 1.16226259e+02, 1.21496331e+02, 1.27014792e+02,
           1.32793346e+02, 1.38844253e+02, 1.45180349e+02, 1.51815076e+02,
           1.58762509e+02, 1.66037386e+02, 1.73655140e+02, 1.81631931e+02,
           1.89984681e+02, 1.98731110e+02, 2.07889773e+02, 2.17480099e+02,
           2.27522433e+02, 2.38038079e+02, 2.49049344e+02, 2.60579589e+02,
           2.72653273e+02, 2.85296010e+02, 2.98534620e+02, 3.12397188e+02,
           3.26913121e+02, 3.42113214e+02, 3.58029713e+02, 3.74696383e+02,
           3.92148580e+02, 4.10423328e+02, 4.29559396e+02, 4.49597378e+02,
           4.70579783e+02, 4.92551123e+02, 5.15558010e+02, 5.39649249e+02,
           5.64875947e+02, 5.91291622e+02, 6.18952312e+02, 6.47916695e+02,
           6.78246218e+02, 7.10005221e+02, 7.43261078e+02, 7.78084340e+02,
           8.14548879e+02, 8.52732052e+02, 8.92714861e+02, 9.34582126e+02,
           9.78422665e+02, 1.02432948e+03, 1.07239996e+03, 1.12273608e+03,
           1.17544463e+03, 1.23063741e+03, 1.28843153e+03, 1.34894957e+03,
           1.41231994e+03, 1.47867705e+03, 1.54816168e+03, 1.62092124e+03,
           1.69711008e+03, 1.77688982e+03, 1.86042971e+03, 1.94790698e+03,
           2.03950719e+03, 2.13542468e+03, 2.23586291e+03, 2.34103497e+03,
           2.45116395e+03, 2.56648350e+03, 2.68723825e+03, 2.81368437e+03,
           2.94609011e+03, 3.08473635e+03, 3.22991721e+03, 3.38194069e+03,
           3.54112929e+03, 3.70782070e+03, 3.88236856e+03, 4.06514315e+03,
           4.25653221e+03, 4.45694175e+03, 4.66679693e+03, 4.88654292e+03,
           5.11664591e+03, 5.35759404e+03, 5.60989844e+03, 5.87409437e+03,
           6.15074229e+03, 6.44042909e+03, 6.74376930e+03, 7.06140644e+03,
           7.39401435e+03, 7.74229861e+03, 8.10699809e+03, 8.48888646e+03,
           8.88877386e+03, 9.30750861e+03, 9.74597902e+03, 1.02051153e+04,
           1.06858914e+04, 1.11893272e+04, 1.17164909e+04, 1.22685006e+04,
           1.28465275e+04, 1.34517977e+04, 1.40855953e+04, 1.47492649e+04,
           1.54442143e+04, 1.61719178e+04, 1.69339191e+04, 1.77318349e+04,
           1.85673577e+04, 1.94422602e+04, 2.03583982e+04, 2.13177153e+04,
           2.23222466e+04, 2.33741232e+04, 2.44755764e+04, 2.56289429e+04])

        proba = np.zeros(my.cfg.N_tot,dtype=np.float64)
        for agent in range(my.cfg.N_tot):
            proba[agent] = probs[my.number_of_contacts[agent]]
        return nb_random_choice(
            possible_agents,
            prob=proba,
            size=my.cfg.N_init,
            replace=False,
        )
    else:
        return np.random.choice(
            possible_agents,
            size=my.cfg.N_init,
            replace=False,
        )


@njit
def compute_initial_agents_to_infect(my, possible_agents):

    ##  Standard outbreak type, infecting randomly
    if my.cfg.make_random_initial_infections:
        return make_random_initial_infections(my, possible_agents)

    # Local outbreak type, infecting around a point:
    else:

        rho_init_local_outbreak = 0.1

        outbreak_agent = single_random_choice(possible_agents)  # this is where the outbreak starts

        initial_agents_to_infect = List()
        initial_agents_to_infect.append(outbreak_agent)

        while len(initial_agents_to_infect) < (my.cfg.N_init):
            proposed_agent = single_random_choice(possible_agents)

            if my.dist_accepted(outbreak_agent, proposed_agent, rho_init_local_outbreak):
                if proposed_agent not in initial_agents_to_infect:
                    initial_agents_to_infect.append(proposed_agent)
        return np.asarray(initial_agents_to_infect, dtype=np.uint32)


@njit
def compute_initial_agents_to_infect_from_kommune(
    my,
    infected_per_kommune_start,
    kommune_names,
    my_kommune,
    verbose=False,
):
    fraction_found = 0.5  # estimate of size of fraction of positive we find. roughly speaking "mørketallet" is the inverse of this times n_found_positive- TODO: make this a function based on N_daily_test, tracking_rates and symptomatics
    contact_number_init = (
        1.05  # used to estimate how many people are in the E state, from how many found positive.
    )
    N_tot_frac = my.cfg.N_tot / 5_800_000
    time_inf = 1 / my.cfg.lambda_I
    time_e = 1 / my.cfg.lambda_E
    norm_factor = contact_number_init / fraction_found * N_tot_frac * (1 + time_e / time_inf)

    initial_agents_to_infect = List()
    # num_infected = 0

    for num_of_infected_in_kommune, kommune in zip(infected_per_kommune_start, kommune_names):
        num_of_infected_normed = np.int32(np.ceil(num_of_infected_in_kommune * norm_factor))
        if verbose:
            print(
                "kommune",
                kommune,
                "num_infected",
                num_of_infected_normed,
                "non normed",
                num_of_infected_in_kommune,
            )
        list_of_agent_in_kommune = List()
        for agent, agent_kommune in zip(range(my.cfg.N_tot), my_kommune):
            if kommune == agent_kommune:
                list_of_agent_in_kommune.append(agent)
        if num_of_infected_normed != 0:
            if verbose:
                print("kommune", kommune, "num_infected", num_of_infected_normed)
            liste = np.random.choice(
                np.asarray(list_of_agent_in_kommune), size=num_of_infected_normed, replace=False
            )
            for entry in liste:
                initial_agents_to_infect.append(entry)
    E_I_ratio = contact_number_init * time_e / (contact_number_init * time_e + time_inf)
    return initial_agents_to_infect, E_I_ratio


@njit
def find_outbreak_agent(my, possible_agents, coordinate, rho, max_tries=10_000):
    counter = 0
    while True:
        outbreak_agent = single_random_choice(possible_agents)
        if my.dist_accepted_coordinate(outbreak_agent, coordinate, rho):
            return outbreak_agent
        counter += 1
        if counter >= max_tries:
            raise AssertionError("Couldn't find any outbreak agent!")


@njit
def calc_E_I_dist(my, r_guess):
    p_E = 4/my.cfg.lambda_E
    p_I = 4/my.cfg.lambda_I
    gen = p_E + p_I
    E_I_weight_list = np.ones(8,dtype=np.float64)
    for i in range(1,4):
        E_I_weight_list[i:] = E_I_weight_list[i:]* r_guess**(1/my.cfg.lambda_I/(p_E*p_I))
    for i in range(4,8):
        E_I_weight_list[i:] = E_I_weight_list[i:]* r_guess**(1/my.cfg.lambda_E/(p_E*p_I))
    E_I_weight_list = E_I_weight_list[::-1]
    return E_I_weight_list




@njit
def make_initial_infections(
    my,
    g,
    state_total_counts,
    agents_in_state,
    SIR_transition_rates,
    agents_in_age_group,
    initial_ages_exposed,
    # N_infectious_states,
    N_states,
):

    # version 2 has age groups
    if my.cfg.version >= 2:
        possible_agents = List()
        for age_exposed in initial_ages_exposed:
            for agent in agents_in_age_group[age_exposed]:
                possible_agents.append(agent)
        possible_agents = np.asarray(possible_agents, dtype=np.uint32)
    # version 1 has no age groups
    else:
        possible_agents = np.arange(my.cfg.N_tot, dtype=np.uint32)

    initial_agents_to_infect = compute_initial_agents_to_infect(my, possible_agents)

    ##  Now make initial infections
    for _, agent in enumerate(initial_agents_to_infect):
        weights = calc_E_I_dist(my, 1)
        states = np.arange(N_states - 1, dtype=np.int8)
        new_state = nb_random_choice(states, weights)[0]  # E1-E4 or I1-I4, uniformly distributed
        my.state[agent] = new_state
        if np.random.rand() < my.cfg.N_init_UK_frac:
            my.corona_type[agent] = 1  # IMPORTANT LINE!

        agents_in_state[new_state].append(np.uint32(agent))
        state_total_counts[new_state] += 1

        g.total_sum_of_state_changes += SIR_transition_rates[new_state]
        g.cumulative_sum_of_state_changes[new_state:] += SIR_transition_rates[new_state]

        # Moves TO infectious State from non-infectious
        if my.agent_is_infectious(agent):
            for contact, rate in zip(my.connections[agent], g.rates[agent]):
                # update rates if contact is susceptible
                if my.agent_is_susceptable(contact):
                    g.update_rates(my, +rate, agent)

        update_infection_list_for_newly_infected_agent(my, g, agent)

    # English Corona Type TODO

    #if my.cfg.N_init_UK > 0: init uk, as outbreak
    if False:

        rho_init_local_outbreak = 0.1

        possible_agents_UK = np.arange(my.cfg.N_tot, dtype=np.uint32)

        # this is where the outbreak starts

        if my.cfg.outbreak_position_UK.lower() == "københavn":
            coordinate = (55.67594, 12.56553)

            outbreak_agent_UK = find_outbreak_agent(
                my,
                possible_agents_UK,
                coordinate,
                rho_init_local_outbreak,
                max_tries=10_000,
            )

            # print("København", outbreak_agent_UK, my.coordinates[outbreak_agent_UK])

        elif my.cfg.outbreak_position_UK.lower() == "nordjylland":
            coordinate = (57.36085, 10.09901)  # "Vendsyssel" på Google Maps

            outbreak_agent_UK = find_outbreak_agent(
                my,
                possible_agents_UK,
                coordinate,
                rho_init_local_outbreak,
                max_tries=10_000,
            )
            # print("nordjylland", outbreak_agent_UK, my.coordinates[outbreak_agent_UK])

        # elif "," in my.cfg.outbreak_position_UK:
        # pass
        else:
            outbreak_agent_UK = single_random_choice(possible_agents_UK)
            # print("random", outbreak_agent_UK, my.coordinates[outbreak_agent_UK])

        initial_agents_to_infect_UK = List()
        initial_agents_to_infect_UK.append(outbreak_agent_UK)

        while len(initial_agents_to_infect_UK) < my.cfg.N_init_UK:
            proposed_agent_UK = single_random_choice(possible_agents_UK)

            if my.dist_accepted(outbreak_agent_UK, proposed_agent_UK, rho_init_local_outbreak):
                if proposed_agent_UK not in initial_agents_to_infect_UK:
                    if my.agent_is_susceptable(proposed_agent_UK):
                        initial_agents_to_infect_UK.append(proposed_agent_UK)

        initial_agents_to_infect_UK = np.asarray(initial_agents_to_infect_UK, dtype=np.uint32)

        ##  Now make initial UK infections
        for _, agent in enumerate(initial_agents_to_infect_UK):
            weights = calc_E_I_dist(my, 1)
            states = np.arange(N_states - 1, dtype=np.int8)
            new_state = nb_random_choice(states, weights)[0]
            my.state[agent] = new_state
            my.corona_type[agent] = 1  # IMPORTANT LINE!

            agents_in_state[new_state].append(np.uint32(agent))
            state_total_counts[new_state] += 1

            g.total_sum_of_state_changes += SIR_transition_rates[new_state]
            g.cumulative_sum_of_state_changes[new_state:] += SIR_transition_rates[new_state]

            # Moves TO infectious State from non-infectious
            if my.agent_is_infectious(agent):
                for contact, rate in zip(my.connections[agent], g.rates[agent]):
                    # update rates if contact is susceptible
                    if my.agent_is_susceptable(contact):
                        g.update_rates(my, +rate, agent)

            update_infection_list_for_newly_infected_agent(my, g, agent)

    return None


@njit
def make_initial_infections_from_kommune_data(
    my,
    g,
    state_total_counts,
    agents_in_state,
    SIR_transition_rates,
    agents_in_age_group,
    initial_ages_exposed,
    # N_infectious_states,
    N_states,
    infected_per_kommune_ints,
    kommune_names,
    my_kommune,
    verbose=False,
):

    # version 2 has age groups
    if my.cfg.version >= 2:
        possible_agents = List()
        for age_exposed in initial_ages_exposed:
            for agent in agents_in_age_group[age_exposed]:
                possible_agents.append(agent)
        possible_agents = np.asarray(possible_agents, dtype=np.uint32)
    # version 1 has no age groups
    else:
        possible_agents = np.arange(my.cfg.N_tot, dtype=np.uint32)

    initial_agents_to_infect, E_I_ratio = compute_initial_agents_to_infect_from_kommune(
        my,
        infected_per_kommune_ints,
        kommune_names,
        my_kommune,
        verbose,
    )
    # initial_agents_to_infect.flatten()
    g.total_sum_of_state_changes = 0.0

    ##  Now make initial infections
    for _, agent in enumerate(initial_agents_to_infect):
        new_state = np.random.randint(N_states - 1)  # E1-E4 or I1-I4, uniformly distributed
        if np.random.rand() < E_I_ratio:
            new_state += 4
        my.state[agent] = new_state

        agents_in_state[new_state].append(np.uint32(agent))
        state_total_counts[new_state] += 1

        g.total_sum_of_state_changes += SIR_transition_rates[new_state]
        g.cumulative_sum_of_state_changes[new_state:] += SIR_transition_rates[new_state]

        # if my.state[agent] >= N_infectious_states:
        if my.agent_is_infectious(agent):
            for contact, rate in zip(my.connections[agent], g.rates[agent]):
                # update rates if contact is susceptible
                if my.agent_is_susceptable(contact):
                    g.update_rates(my, +rate, agent)

        update_infection_list_for_newly_infected_agent(my, g, agent)

    return None


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # PRE SIMULATION  # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@njit
def do_bug_check(
    my,
    g,
    step_number,
    day,
    continue_run,
    verbose,
    state_total_counts,
    N_states,
    accept,
    ra1,
    s,
    x,
):

    if my.cfg.day_max > 0 and day >= my.cfg.day_max:
        if verbose:
            print("day exceeded day_max")
        continue_run = False

    elif day > 10_000:
        print("day exceeded 10_000")
        continue_run = False

    elif step_number > 100_000_000:
        print("step_number > 100_000_000")
        continue_run = False

    elif (g.total_sum_infections + g.total_sum_of_state_changes < 0.0001) and (
        g.total_sum_of_state_changes + g.total_sum_infections > -0.00001
    ):
        continue_run = False
        if verbose:
            print("Equilibrium")
            print(day, my.cfg.day_max, my.cfg.day_max > 0, day > my.cfg.day_max)

    elif state_total_counts[N_states - 1] > my.cfg.N_tot - 10:
        if verbose:
            print("2/3 through")
        continue_run = False

    # Check for bugs
    elif not accept:
        print("\nNo Chosen rate")
        print("s: \t", s)
        print("g.total_sum_infections: \t", g.total_sum_infections)
        print("g.cumulative_sum_infection_rates: \t", g.cumulative_sum_infection_rates)
        print("g.cumulative_sum_of_state_changes: \t", g.cumulative_sum_of_state_changes)
        print("x: \t", x)
        print("ra1: \t", ra1)
        continue_run = False

    elif (g.total_sum_of_state_changes < 0) and (g.total_sum_of_state_changes > -0.001):
        g.total_sum_of_state_changes = 0

    elif (g.total_sum_infections < 0) and (g.total_sum_infections > -0.001):
        g.total_sum_infections = 0

    elif (g.total_sum_of_state_changes < 0) or (g.total_sum_infections < 0):
        print("\nNegative Problem", g.total_sum_of_state_changes, g.total_sum_infections)
        print("s: \t", s)
        print("g.total_sum_infections: \t", g.total_sum_infections)
        print("g.cumulative_sum_infection_rates: \t", g.cumulative_sum_infection_rates)
        print("g.cumulative_sum_of_state_changes: \t", g.cumulative_sum_of_state_changes)
        print("x: \t", x)
        print("ra1: \t", ra1)
        continue_run = False

    return continue_run


#%%


@njit
def update_infection_list_for_newly_infected_agent(my, g, agent_getting_infected):

    # Here we update infection lists so that newly infected cannot be infected again

    # loop over contacts of the newly infected agent in order to:
    # 1) remove newly infected agent from contact list (find_myself) by setting rate to 0
    # 2) remove rates from contacts gillespie sums (only if they are in infections state (I))
    for contact_of_agent_getting_infected in my.connections[agent_getting_infected]:

        # loop over indexes of the contact to find_myself and set rate to 0
        for ith_contact_of_agent_getting_infected in range(
            my.number_of_contacts[contact_of_agent_getting_infected]
        ):

            find_myself = my.connections[contact_of_agent_getting_infected][
                ith_contact_of_agent_getting_infected
            ]

            # check if the contact found is myself
            if find_myself == agent_getting_infected:

                rate = g.rates[contact_of_agent_getting_infected][
                    ith_contact_of_agent_getting_infected
                ]

                # set rates to myself to 0 (I cannot get infected again)
                g.rates[contact_of_agent_getting_infected][
                    ith_contact_of_agent_getting_infected
                ] = 0

                # if the contact can infect, then remove the rates from the overall gillespie accounting
                if my.agent_is_infectious(contact_of_agent_getting_infected):
                    g.update_rates(my, -rate, contact_of_agent_getting_infected)

                break


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # SIMULATION  # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@njit
def run_simulation(
    my,
    g,
    intervention,
    state_total_counts,
    agents_in_state,
    N_states,
    SIR_transition_rates,
    N_infectious_states,
    nts,
    verbose,
):
    print("apply intervention", intervention.apply_interventions)

    out_time = List()
    out_state_counts = List()
    out_my_state = List()

    daily_counter = 0
    day = -1 * my.cfg.burn_in
    #day = 0
    click = nts * day
    step_number = 0
    real_time = 1.0 * day
    print(day, click, real_time)

    s_counter = np.zeros(4)
    where_infections_happened_counter = np.zeros(4)

    days_of_vacci_start = my.cfg.days_of_vacci_start


    # Run the simulation ################################
    continue_run = True
    while continue_run:

        s = 0

        step_number += 1
        g.total_sum = g.total_sum_of_state_changes + g.total_sum_infections

        dt = -np.log(np.random.rand()) / g.total_sum
        real_time += dt

        g.cumulative_sum = 0.0
        ra1 = np.random.rand()

        #######/ Here we move between infected between states
        accept = False
        if g.total_sum_of_state_changes / g.total_sum > ra1:

            s = 1

            x = g.cumulative_sum_of_state_changes / g.total_sum
            state_now = np.searchsorted(x, ra1)
            state_after = state_now + 1

            agent = utils.numba_random_choice_list(agents_in_state[state_now])

            # We have chosen agent to move -> here we move it
            agents_in_state[state_after].append(agent)
            agents_in_state[state_now].remove(agent)

            my.state[agent] += 1

            state_total_counts[state_now] -= 1
            state_total_counts[state_after] += 1

            g.total_sum_of_state_changes -= SIR_transition_rates[state_now]
            g.total_sum_of_state_changes += SIR_transition_rates[state_after]

            g.cumulative_sum_of_state_changes[state_now] -= SIR_transition_rates[state_now]
            g.cumulative_sum_of_state_changes[state_after:] += (
                SIR_transition_rates[state_after] - SIR_transition_rates[state_now]
            )

            g.cumulative_sum_infection_rates[state_now] -= g.sum_of_rates[agent]

            accept = True

            if intervention.apply_interventions and intervention.apply_symptom_testing and day >= 0:
                apply_symptom_testing(my, intervention, agent, click)

            # Moves TO infectious State from non-infectious
            if my.state[agent] == N_infectious_states:
                # for i, (contact, rate) in enumerate(zip(my.connections[agent], g.rates[agent])):
                for i, contact in enumerate(my.connections[agent]):
                    # update rates if contact is susceptible
                    if my.agent_is_susceptable(contact):
                        if my.corona_type[agent] == 1:
                            g.rates[agent][i] *= my.cfg.beta_UK_multiplier
                        rate = g.rates[agent][i]
                        g.update_rates(my, +rate, agent)

            # If this moves to Recovered state
            if my.state[agent] == N_states - 1:
                for contact, rate in zip(my.connections[agent], g.rates[agent]):
                    # update rates if contact is susceptible
                    if my.agent_is_susceptable(contact):
                        g.update_rates(my, -rate, agent)

        #######/ Here we infect new states
        else:
            s = 2

            x = (g.total_sum_of_state_changes + g.cumulative_sum_infection_rates) / g.total_sum
            state_now = np.searchsorted(x, ra1)
            g.cumulative_sum = (
                g.total_sum_of_state_changes + g.cumulative_sum_infection_rates[state_now - 1]
            ) / g.total_sum  # important change from [state_now] to [state_now-1]

            agent_getting_infected = -1
            for agent in agents_in_state[state_now]:

                # suggested cumulative sum
                suggested_cumulative_sum = g.cumulative_sum + g.sum_of_rates[agent] / g.total_sum

                if suggested_cumulative_sum > ra1:
                    ith_contact = 0
                    for rate, contact in zip(g.rates[agent], my.connections[agent]):

                        # if contact is susceptible
                        if my.state[contact] == -1:

                            g.cumulative_sum += rate / g.total_sum

                            # here agent infect contact
                            if g.cumulative_sum > ra1:
                                where_infections_happened_counter[
                                    my.connections_type[agent][ith_contact]
                                ] += 1
                                my.state[contact] = 0

                                my.corona_type[contact] = my.corona_type[agent]

                                agents_in_state[0].append(np.uint32(contact))
                                state_total_counts[0] += 1
                                g.total_sum_of_state_changes += SIR_transition_rates[0]
                                g.cumulative_sum_of_state_changes += SIR_transition_rates[0]
                                accept = True
                                agent_getting_infected = contact
                                break
                            ith_contact += 1
                else:
                    g.cumulative_sum = suggested_cumulative_sum

                if accept:
                    break

            if agent_getting_infected == -1:
                print(
                    "Error! Not choosing any agent getting infected.",
                    *("\naccept:", accept),
                    *("\nagent_getting_infected: ", agent_getting_infected),
                    *("\nstep_number", step_number),
                    "\ncfg:",
                )
                # my.cfg.print()
                break

            # Here we update infection lists so that newly infected cannot be infected again
            update_infection_list_for_newly_infected_agent(my, g, agent_getting_infected)

        ################

        while nts * click + abs(2*my.cfg.burn_in) < real_time + abs(2*my.cfg.burn_in):
            # if nts * click < real_time:

            daily_counter += 1
            if ((len(out_time) == 0) or (real_time != out_time[-1])) and day >= 0:
                out_time.append(real_time)
                out_state_counts.append(state_total_counts.copy())

            if daily_counter >= 10:
                day += 1
                if intervention.apply_interventions and intervention.apply_interventions_on_label and day >= 0:
                    apply_interventions_on_label(my, g, intervention, day, click)

                if intervention.apply_random_testing:
                    apply_random_testing(my, intervention, click)

                if my.cfg.N_events > 0:
                    add_daily_events(
                        my,
                        g,
                        day,
                        agents_in_state,
                        state_total_counts,
                        SIR_transition_rates,
                        where_infections_happened_counter,
                    )

                daily_counter = 0

                if day >= 0:
                    out_my_state.append(my.state.copy())
                if verbose:
                    print("day", day, "n_inf", np.sum(where_infections_happened_counter) )
                    print("R_true", intervention.R_true_list[-1])
                    print("freedom_impact", intervention.freedom_impact_list[-1])
                    print("R_true_list_brit", intervention.R_true_list_brit[-1])

                if intervention.apply_vaccinations:
                    if days_of_vacci_start > 0:
                        for day in range(days_of_vacci_start):
                            vaccinate(my, g, intervention, agents_in_state, state_total_counts, day)
                        intervention.vaccination_schedule - days_of_vacci_start
                        days_of_vacci_start = 0

                    vaccinate(my, g, intervention, agents_in_state, state_total_counts, day)

            if intervention.apply_interventions:
                test_tagged_agents(my, g, intervention, day, click)
            if day >= 0:
                intervention.R_true_list.append(calculate_R_True(my, g))
                intervention.freedom_impact_list.append(calculate_population_freedom_impact(intervention))
                intervention.R_true_list_brit.append(calculate_R_True_brit(my, g))


            click += 1

        continue_run = do_bug_check(
            my,
            g,
            step_number,
            day,
            continue_run,
            verbose,
            state_total_counts,
            N_states,
            accept,
            ra1,
            s,
            x,
        )

        s_counter[s] += 1

    if verbose:
        print("Simulation step_number, ", step_number)
        print("s_counter", s_counter)
        print("Where", where_infections_happened_counter)
        print("positive_test_counter", intervention.positive_test_counter)
        print("n_found", np.sum(np.array([1 for day_found in intervention.day_found_infected if day_found>=0])))
        #print(list(calc_contact_dist(my,2)))
        # frac_inf = np.zeros((2,200))
        # for agent in range(my.cfg.N_tot):
        #     n_con = my.number_of_contacts[agent]
        #     frac_inf[1,n_con] +=1
        #     if my.state[agent]>=0 and my.state[agent] < 8:
        #         frac_inf[0,n_con] +=1
        #print(frac_inf[0,:]/frac_inf[1,:])
        # print("N_daily_tests", intervention.N_daily_tests)
        # print("N_positive_tested", N_positive_tested)

    return out_time, out_state_counts, out_my_state, intervention


#%%
# ███    ███  █████  ██████  ████████ ██ ███    ██ ██    ██
# ████  ████ ██   ██ ██   ██    ██    ██ ████   ██  ██  ██
# ██ ████ ██ ███████ ██████     ██    ██ ██ ██  ██   ████
# ██  ██  ██ ██   ██ ██   ██    ██    ██ ██  ██ ██    ██
# ██      ██ ██   ██ ██   ██    ██    ██ ██   ████    ██
#
#%%
@njit
def calc_contact_dist(my, contact_type):
    contact_dist = np.zeros(100)
    for agent in range(my.cfg.N_tot):
        agent_sum = 0
        for ith_contact in range(len(my.connections[agent])):
            if my.connections_type[agent][ith_contact] == contact_type:
                agent_sum += 1
        contact_dist[agent_sum] += 1
    return contact_dist



@njit
def vaccinate(my, g, intervention, agents_in_state, state_total_counts, day):

    # try to vaccinate everyone, but only do vaccinate susceptable agents
    possible_agents_to_vaccinate = np.arange(my.cfg.N_tot, dtype=np.uint32)
    # agent = utils.numba_random_choice_list(agents_in_state[state_now])

    R_state = g.N_states - 1  # 8

    # Check if all vaccines have been given
    if day > intervention.vaccination_schedule[-1] :
        return

    # Check if any vaccines are effective yet:
    if day >= intervention.vaccination_schedule[0] :

        # Get the number of new effective vaccines
        N = intervention.vaccinations_per_age_group[day - intervention.vaccination_schedule[0]]

        # Compute probability for each agent being infected
        probabilities = np.array([N[my.age[agent]] for agent in possible_agents_to_vaccinate])

        # Distribute the effective vaccines among the population
        agents = nb_random_choice(possible_agents_to_vaccinate, probabilities, size = int(np.sum(N)))
        for agent in agents:

            # pick agent if it is susceptible (in S state)
            if my.agent_is_susceptable(agent):
                # "vaccinate agent"
                my.vaccination_type[agent] = 1

                # set agent to recovered, instantly
                my.state[agent] = R_state

                agents_in_state[R_state].append(np.uint32(agent))
                state_total_counts[R_state] += 1

                # remove rates into agent from its infectios contacts
                update_infection_list_for_newly_infected_agent(my, g, agent)

@njit
def calculate_R_True(my, g):
    lambda_I = my.cfg.lambda_I
    rate_sum = g.total_sum_infections
    N_infected = 0
    for agent in range(my.cfg.N_tot):
        if my.agent_is_infectious(agent):
            N_infected += 1
    return rate_sum / lambda_I / np.maximum(N_infected, 1.0) * 4

@njit
def calculate_R_True_brit(my, g):
    lambda_I = my.cfg.lambda_I
    rate_sum = 0
    N_infected = 0
    for agent in range(my.cfg.N_tot):
        if my.agent_is_infectious(agent) and my.corona_type[agent] == 1:
            N_infected += 1
            rate_sum += g.sum_of_rates[agent]
    return rate_sum / lambda_I / np.maximum(N_infected, 1.0) * 4

@njit
def calculate_population_freedom_impact(intervention):
    return np.mean(intervention.freedom_impact)

@njit
def calculate_pandemic_control(my, intervention):
    I_crit = 2000.0*my.cfg.N_tot/5_800_000/my.cfg.lambda_I*4
    b = np.log(2)/I_crit
    N_infected = 0
    for agent in range(my.cfg.N_tot):
        if my.agent_is_infectious(agent):
            N_infected += 1
    return (1.0/(1.0+np.exp(-b*(N_infected-I_crit)))-(1/3))*3/2




@njit
def compute_my_cluster_coefficient(my):
    """calculates cluster cooefficent
    (np.mean of the first output gives cluster coeff for whole network ).
    This function is somewhat slow, since it loops over all connections. Could just random sample, but here we get the exact distribution.
    """

    cluster_coefficient = np.zeros(my.cfg.N_tot, dtype=np.float32)
    for agent1 in range(my.cfg.N_tot):
        counter = 0
        total = 0
        for j, contact in enumerate(my.connections[agent1]):
            for k in range(j + 1, my.number_of_contacts[agent1]):
                if contact in my.connections[my.connections[agent1][k]]:
                    counter += 1
                    break
                total += 1
        cluster_coefficient[agent1] = counter / total
    return cluster_coefficient


@njit
def initialize_tents(my, N_tents):

    """Pick N_tents tents in random positions and compute which tent each
    person is nearest to and connect that person to that tent.
    """

    N_tot = my.cfg.N_tot

    tent_positions = np.zeros((N_tents, 2), np.float32)
    for i in range(N_tents):
        tent_positions[i] = my.coordinates[np.random.randint(N_tot)]

    tent_counter = np.zeros(N_tents, np.uint32)
    for agent in range(N_tot):
        distances = np.array(
            [utils.haversine_scipy(my.coordinates[agent], p_tent) for p_tent in tent_positions]
        )
        closest_tent = np.argmin(distances)
        my.tent[agent] = closest_tent
        tent_counter[closest_tent] += 1

    return tent_positions, tent_counter


# @njit
def initialize_kommuner(my, df_coordinates):
    my.kommune = np.array(df_coordinates["idx"].values, dtype=np.uint8)
    kommune_counter = df_coordinates["idx"].value_counts().sort_index()
    kommune_counter = np.array(kommune_counter, dtype=np.uint32)
    return kommune_counter


@njit
def test_if_label_needs_intervention(
    intervention,
    day,
    intervention_type_to_init,
    threshold=0.02,  # threshold is the fraction that need to be positive.
):
    infected_per_label = np.zeros_like(intervention.label_counter, dtype=np.uint32)

    for agent, day_found in enumerate(intervention.day_found_infected):
        if day_found > max(0, day - intervention.cfg.days_looking_back):
            infected_per_label[intervention.labels[agent]] += 1

    it = enumerate(
        zip(
            infected_per_label,
            intervention.label_counter,
            intervention.types,
        )
    )
    for i_label, (N_infected, N_inhabitants, my_intervention_type) in it:
        if N_infected / N_inhabitants > threshold and my_intervention_type == 0:
            if intervention.verbose:
                print(
                    *("lockdown at label", i_label),
                    *("at day", day),
                    *("the num of infected is", N_infected),
                    *("/", N_inhabitants),
                )

            intervention.types[i_label] = intervention_type_to_init

    return None

@njit
def test_if_label_needs_intervention_multi(
    intervention,
    day,
    threshold_info,
):
    infected_per_label = np.zeros_like(intervention.label_counter, dtype=np.uint32)

    for agent, day_found in enumerate(intervention.day_found_infected):
        if day_found > max(0, day - intervention.cfg.days_looking_back):
            infected_per_label[intervention.labels[agent]] += 1


    it = enumerate(
        zip(
            infected_per_label,
            intervention.label_counter,
            intervention.types,
        )
    )
    for i_label, (N_infected, N_inhabitants, my_intervention_type) in it:
        for ith_intervention in range(0, len(threshold_info) + 1):
            if my_intervention_type == 0:
                possible_interventions = [1, 2, 7]
            elif my_intervention_type == 1:
                possible_interventions = [9001] # random integer that doesn't mean anything,
            elif my_intervention_type == 2:
                possible_interventions = [1,7]
            elif my_intervention_type == 7:
                possible_interventions = [9001] # random integer that doesn't mean anything,

            if N_infected / N_inhabitants > threshold_info[ith_intervention+1][0]/100_000.0 and threshold_info[0][ith_intervention] in possible_interventions:
                if intervention.verbose:
                    intervention_type_name = ["nothing","lockdown","masking","error","error","error","error","matrix_based"]
                    print(
                        *(intervention_type_name[threshold_info[0][ith_intervention]]," at label", i_label),
                        *("at day", day),
                        *("the num of infected is", N_infected),
                        *("/", N_inhabitants),
                    )

                intervention.types[i_label] = threshold_info[0][ith_intervention]
                break

    return None

@njit
def reset_rates_of_agent(my, g, agent, intervention, connection_type_weight=None):

    if connection_type_weight is None:
        # reset infection rate to origin times this number for [home, job, other]
        connection_type_weight = np.ones(3, dtype=np.float32)

    agent_update_rate = 0.0
    for ith_contact, contact in enumerate(my.connections[agent]):

        infection_rate = (
            my.infection_weight[agent]
            * connection_type_weight[my.connections_type[agent][ith_contact]]
        )

        rate = infection_rate - g.rates[agent][ith_contact]
        intervention.freedom_impact[agent] = 0
        g.rates[agent][ith_contact] = infection_rate

        if my.agent_is_infectious(agent) and my.agent_is_susceptable(contact):
            agent_update_rate += rate

        # loop over indexes of the contact to find_myself and set rate to 0
        for ith_contact_of_contact, possible_agent in enumerate(my.connections[contact]):

            # check if the contact found is myself
            if agent == possible_agent:
                infection_rate = (
                    my.infection_weight[contact]
                    * connection_type_weight[my.connections_type[contact][ith_contact_of_contact]]                )

                # update rates from contact to agent.
                c_rate = my.infection_weight[contact] - g.rates[contact][ith_contact_of_contact]
                intervention.freedom_impact[contact] = 0
                g.rates[contact][ith_contact_of_contact] = my.infection_weight[contact]

                # updates to gillespie sums, if contact is infectious and agent is susceptible
                if my.agent_is_infectious(contact) and my.agent_is_susceptable(agent):
                    g.update_rates(my, c_rate, contact)
                break

    # actually updates to gillespie sums
    g.update_rates(my, +agent_update_rate, agent)
    return None


@njit
def remove_intervention_at_label(my, g, intervention, ith_label):
    for agent in range(my.cfg.N_tot):
        if intervention.labels[agent] == ith_label and my.restricted_status[agent] == 1:
            reset_rates_of_agent(my, g, agent, intervention, connection_type_weight=None)
            my.restricted_status[agent] = 0
    return None


@njit
def test_if_intervention_on_labels_can_be_removed(my, g, intervention, day, threshold=0.001):

    infected_per_label = np.zeros(intervention.N_labels, dtype=np.int32)
    for agent, day_found in enumerate(intervention.day_found_infected):
        if day_found > day - intervention.cfg.days_looking_back:
            infected_per_label[intervention.labels[agent]] += 1

    it = enumerate(
        zip(
            infected_per_label,
            intervention.label_counter,
            intervention.types,
        )
    )
    for i_label, (N_infected, N_inhabitants, my_intervention_type) in it:
        if (N_infected / N_inhabitants) < threshold and my_intervention_type == 0:

            remove_intervention_at_label(my, g, intervention, i_label)

            intervention.types[i_label] = 0
            intervention.started_as[i_label] = 0
            if intervention.verbose:
                print(
                    *("remove lockdown at num of infected", i_label),
                    *("at day", day),
                    *("the num of infected is", N_infected),
                    *("/", N_inhabitants),
                )

    return None

@njit
def test_if_intervention_on_labels_can_be_removed_multi(my, g, intervention, day, click,  threshold_info):

    infected_per_label = np.zeros(intervention.N_labels, dtype=np.int32)
    for agent, day_found in enumerate(intervention.day_found_infected):
        if day_found > day - intervention.cfg.days_looking_back:
            infected_per_label[intervention.labels[agent]] += 1

    it = enumerate(
        zip(
            infected_per_label,
            intervention.label_counter,
            intervention.types,
        )
    )
    for i_label, (N_infected, N_inhabitants, my_intervention_type) in it:
        for ith_intervention in range(0, len(threshold_info)-1):
            if N_infected / N_inhabitants < threshold_info[ith_intervention + 1][1]/100_000.0 and my_intervention_type == threshold_info[0][ith_intervention]:
                intervention.clicks_when_restriction_stops[i_label] = click + my.cfg.intervention_removal_delay_in_clicks


    return None

@njit
def test_if_intervention_on_labels_can_be_removed_multi_old(my, g, intervention, day, threshold_info):

    infected_per_label = np.zeros(intervention.N_labels, dtype=np.int32)
    for agent, day_found in enumerate(intervention.day_found_infected):
        if day_found > day - intervention.cfg.days_looking_back:
            infected_per_label[intervention.labels[agent]] += 1

    it = enumerate(
        zip(
            infected_per_label,
            intervention.label_counter,
            intervention.types,
        )
    )
    for i_label, (N_infected, N_inhabitants, my_intervention_type) in it:
        for ith_intervention in range(0, len(threshold_info)-1):
            if N_infected / N_inhabitants < threshold_info[ith_intervention + 1][1]/100_000.0 and my_intervention_type == threshold_info[0][ith_intervention]:
                remove_intervention_at_label(my, g, intervention, i_label)

                intervention.types[i_label] = 0
                intervention.started_as[i_label] = 0
                if intervention.verbose:
                    intervention_type_name = ["nothing","lockdown","masking"]
                    print(
                        *("remove ", intervention_type_name[threshold_info[0][ith_intervention]], " at num of infected", i_label),
                        *("at day", day),
                        *("the num of infected is", N_infected),
                        *("/", N_inhabitants),
                    )

    return None


@njit
def loop_update_rates_of_contacts(
    my, g, intervention, agent, contact, rate, agent_update_rate, rate_reduction
):

    # updates to gillespie sums, if agent is infected and contact is susceptible
    if my.agent_is_infectious(agent) and my.agent_is_susceptable(contact):
        agent_update_rate += rate

    # loop over indexes of the contact to find_myself and set rate to 0
    for ith_contact_of_contact, possible_agent in enumerate(my.connections[contact]):

        # check if the contact found is myself
        if agent == possible_agent:

            # update rates from contact to agent. Rate_reduction makes it depending on connection type
            c_rate = (
                g.rates[contact][ith_contact_of_contact]
                * rate_reduction[my.connections_type[contact][ith_contact_of_contact]]
            )
            intervention.freedom_impact[contact] += rate_reduction[my.connections_type[contact][ith_contact_of_contact]]/2/my.number_of_contacts[contact]
            g.rates[contact][ith_contact_of_contact] -= c_rate

            # updates to gillespie sums, if contact is infectious and agent is susceptible
            if my.agent_is_infectious(contact) and my.agent_is_susceptable(agent):
                g.update_rates(my, -c_rate, contact)
            break

    return agent_update_rate


@njit
def cut_rates_of_agent(my, g, intervention, agent, rate_reduction):

    agent_update_rate = 0.0

    # step 1 loop over all of an agents contact
    for ith_contact, contact in enumerate(my.connections[agent]):

        # update rates from agent to contact. Rate_reduction makes it depending on connection type

        rate = g.rates[agent][ith_contact] * rate_reduction[my.connections_type[agent][ith_contact]]
        intervention.freedom_impact[contact] += rate_reduction[my.connections_type[agent][ith_contact]]/2/my.number_of_contacts[agent]

        g.rates[agent][ith_contact] -= rate

        agent_update_rate = loop_update_rates_of_contacts(
            my,
            g,
            intervention,
            agent,
            contact,
            rate,
            agent_update_rate,
            rate_reduction=rate_reduction,
        )

    # actually updates to gillespie sums
    g.update_rates(my, -agent_update_rate, agent)
    return None


@njit
def reduce_frac_rates_of_agent(my, g, intervention, agent, rate_reduction):
    # rate reduction is 2 3-vectors. is used for masking interventions
    agent_update_rate = 0.0
    remove_rates = rate_reduction[0]
    reduce_rates = rate_reduction[1]

    # step 1 loop over all of an agents contact
    for ith_contact, contact in enumerate(my.connections[agent]):

        # update rates from agent to contact. Rate_reduction makes it depending on connection type
        if np.random.rand() > remove_rates[my.connections_type[agent][ith_contact]]:
            act_rate_reduction = np.array([0, 0, 0], dtype=np.float64)
        else:
            act_rate_reduction = reduce_rates

        rate = (
            g.rates[agent][ith_contact]
            * act_rate_reduction[my.connections_type[agent][ith_contact]]
        )
        intervention.freedom_impact[contact] += act_rate_reduction[my.connections_type[agent][ith_contact]]/2/my.number_of_contacts[agent]
        g.rates[agent][ith_contact] -= rate

        agent_update_rate = loop_update_rates_of_contacts(
            my,
            g,
            intervention,
            agent,
            contact,
            rate,
            agent_update_rate,
            rate_reduction=act_rate_reduction,
        )

    # actually updates to gillespie sums
    g.update_rates(my, -agent_update_rate, agent)
    return None


@njit
def remove_and_reduce_rates_of_agent(my, g, intervention, agent, rate_reduction):
    # rate reduction is 2 3-vectors. is used for lockdown interventions
    agent_update_rate = 0.0
    remove_rates = rate_reduction[0]
    reduce_rates = rate_reduction[1]

    # step 1 loop over all of an agents contact
    for ith_contact, contact in enumerate(my.connections[agent]):

        # update rates from agent to contact. Rate_reduction makes it depending on connection type
        act_rate_reduction = reduce_rates
        if np.random.rand() < remove_rates[my.connections_type[agent][ith_contact]]:
            act_rate_reduction = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        rate = (
            g.rates[agent][ith_contact]
            * act_rate_reduction[my.connections_type[agent][ith_contact]]
        )

        g.rates[agent][ith_contact] -= rate
        intervention.freedom_impact[agent] += act_rate_reduction[my.connections_type[agent][ith_contact]]/2/my.number_of_contacts[agent]

        agent_update_rate = loop_update_rates_of_contacts(
            my,
            g,
            intervention,
            agent,
            contact,
            rate,
            agent_update_rate,
            rate_reduction=act_rate_reduction,
        )

    # actually updates to gillespie sums
    g.update_rates(my, -agent_update_rate, agent)
    return None

@njit
def remove_and_reduce_rates_of_agent_matrix(my, g, intervention, agent):
    # rate reduction is 2 3-vectors. is used for lockdown interventions
    agent_update_rate = 0.0
    act_rate_reduction = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    # step 1 loop over all of an agents contact
    for ith_contact, contact in enumerate(my.connections[agent]):
        act_rate_reduction = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        if not my.connections_type[agent][ith_contact] == 0:
            if my.connections_type[agent][ith_contact] == 1:
                mr = intervention.work_matrix_restrict
                mi = my.cfg_network.work_matrix
            elif my.connections_type[agent][ith_contact] == 2:
                mr = intervention.other_matrix_restrict
                mi = my.cfg_network.other_matrix

            mr_single = mr[my.age[agent], my.age[contact]]
            mi_single = mi[my.age[agent], my.age[contact]]

            if mr_single > mi_single:
                print(my.age[agent])
                print(my.age[contact])
                print(my.connections_type[agent][ith_contact])
                assert mr_single < mi_single

            p = 1 - np.sqrt(4 - 4 * (1 - min(mr_single / mi_single, 1))) / 2
            if np.random.rand() < p:
                act_rate_reduction = np.array([0.0, 1.0, 1.0], dtype=np.float64)

        rate = (
            g.rates[agent][ith_contact]
            * act_rate_reduction[my.connections_type[agent][ith_contact]]
        )

        g.rates[agent][ith_contact] -= rate
        intervention.freedom_impact[agent] += act_rate_reduction[my.connections_type[agent][ith_contact]]/2/my.number_of_contacts[agent]

        agent_update_rate = loop_update_rates_of_contacts(
            my,
            g,
            intervention,
            agent,
            contact,
            rate,
            agent_update_rate,
            rate_reduction=act_rate_reduction,
        )

    # actually updates to gillespie sums
    g.update_rates(my, -agent_update_rate, agent)
    return None

@njit
def lockdown_on_label(my, g, intervention, label, rate_reduction):
    # lockdown on all agent with a certain label (tent or municipality, or whatever else you define). Rate reduction is two vectors of length 3. First is the fraction of [home, job, others] rates to set to 0.
    # second is the fraction of reduction of the remaining [home, job, others] rates.
    # ie: [[0,0.8,0.8],[0,0.8,0.8]] means that 80% of your contacts on job and other is set to 0, and the remaining 20% is reduced by 80%.
    # loop over all agents
    for agent in range(my.cfg.N_tot):
        if intervention.labels[agent] == label:
            my.restricted_status[agent] = 1
            remove_and_reduce_rates_of_agent(my, g, intervention, agent, rate_reduction)


@njit
def masking_on_label(my, g, intervention, label, rate_reduction):
    # masking on all agent with a certain label (tent or municipality, or whatever else you define). Rate reduction is two vectors of length 3. First is the fraction of [home, job, others] rates to be effected by masks.
    # second is the fraction of reduction of the those [home, job, others] rates.
    # ie: [[0,0.2,0.2],[0,0.8,0.8]] means that your wear mask when around 20% of job and other contacts, and your rates to those is reduced by 80%
    # loop over all agents
    for agent in range(my.cfg.N_tot):
        if intervention.labels[agent] == label:
            my.restricted_status[agent] = 1
            reduce_frac_rates_of_agent(my, g, intervention, agent, rate_reduction)

@njit
def matrix_restriction_on_label(my, g, intervention, label):
    # masking on all agent with a certain label (tent or municipality, or whatever else you define). Rate reduction is two vectors of length 3. First is the fraction of [home, job, others] rates to be effected by masks.
    # second is the fraction of reduction of the those [home, job, others] rates.
    # ie: [[0,0.2,0.2],[0,0.8,0.8]] means that your wear mask when around 20% of job and other contacts, and your rates to those is reduced by 80%
    # loop over all agents
    for agent in range(my.cfg.N_tot):
        if intervention.labels[agent] == label:
            my.restricted_status[agent] = 1
            remove_and_reduce_rates_of_agent_matrix(my, g, intervention, agent)


@njit
def test_a_person(my, g, intervention, agent, click):
    # if agent is infectious and hasn't been tested before
    if my.agent_is_infectious(agent) and intervention.agent_has_not_been_tested(agent):
        intervention.clicks_when_tested_result[agent] = (
            click + intervention.cfg.results_delay_in_clicks[intervention.reason_for_test[agent]]
        )
        intervention.positive_test_counter[
            intervention.reason_for_test[agent]
        ] += 1  # count reason found infected

        # check if tracking is on
        if intervention.apply_tracking:
            # loop over contacts
            for ith_contact, contact in enumerate(my.connections[agent]):
                if (
                    np.random.rand()
                    < intervention.cfg.tracking_rates[my.connections_type[agent][ith_contact]]
                    and intervention.clicks_when_tested[contact] == -1
                ):
                    intervention.reason_for_test[contact] = 2
                    intervention.clicks_when_tested[contact] = (
                        click + intervention.cfg.test_delay_in_clicks[2]
                    )
                    intervention.clicks_when_isolated[contact] = click + my.cfg.tracking_delay

    # this should only trigger if they have gone into isolation after contact tracing
    elif (
        my.agent_is_not_infectious(agent)
        and intervention.agent_has_been_tested(agent)
        and click > intervention.clicks_when_isolated[agent]
    ):
        reset_rates_of_agent(my, g, agent, intervention, connection_type_weight=None)

    intervention.clicks_when_tested[agent] = -1
    intervention.reason_for_test[agent] = -1
    intervention.clicks_when_isolated[agent] = -1

    return None


@njit
def apply_symptom_testing(my, intervention, agent, click):

    # if my.state[agent] >= N_infectious_states: # old line
    if my.agent_is_infectious(agent):  # TODO: XXX which one of these to lines

        prob = intervention.cfg.chance_of_finding_infected[my.state[agent] - 4]
        randomly_selected = np.random.rand() < prob
        not_tested_before = intervention.clicks_when_tested[agent] == -1

        if randomly_selected and not_tested_before:
            # testing in n_clicks for symptom checking
            intervention.clicks_when_tested[agent] = (
                click + intervention.cfg.test_delay_in_clicks[0]
            )
            # set the reason for testing to symptoms (0)
            intervention.reason_for_test[agent] = 0


@njit
def apply_random_testing(my, intervention, click):
    # choose N_daily_test people at random to test
    N_daily_test = int(intervention.cfg.f_daily_tests * my.cfg.N_tot)
    agents = np.arange(my.cfg.N_tot, dtype=np.uint32)
    random_agents_to_be_tested = np.random.choice(agents, N_daily_test)
    intervention.clicks_when_tested[random_agents_to_be_tested] = (
        click + intervention.cfg.test_delay_in_clicks[1]
    )
    # specify that random test is the reason for test
    intervention.reason_for_test[random_agents_to_be_tested] = 1


@njit
def apply_interventions_on_label(my, g, intervention, day, click):
    if intervention.start_interventions_by_real_incidens_rate or intervention.start_interventions_by_meassured_incidens_rate:
        threshold_info = np.array([[1, 2], [200, 100], [20, 20]]) #TODO: remove
        test_if_intervention_on_labels_can_be_removed_multi(my, g, intervention, day, click, threshold_info)
        for i_label, clicks_when_restriction_stops in enumerate(intervention.clicks_when_restriction_stops):
            if clicks_when_restriction_stops == click:
                remove_intervention_at_label(my, g, intervention, i_label)
                intervention.clicks_when_restriction_stops[i_label] = -1
                intervention_type_n = intervention.types[i_label]
                intervention.types[i_label] = 0
                intervention.started_as[i_label] = 0
                if intervention.verbose:
                    intervention_type_name = ["nothing","lockdown","masking","error","error","error","error","matrix_based"]
                    print(
                        *("remove ", intervention_type_name[intervention_type_n], " at num of infected", i_label),
                        *("at day", day)
                    )

        test_if_label_needs_intervention_multi(
            intervention,
            day,
            threshold_info,
        )


        for ith_label, intervention_type in enumerate(intervention.types):
            if intervention_type in intervention.cfg.threshold_interventions_to_apply:
                intervention_has_not_been_applied = intervention.started_as[ith_label] == 0

                apply_lockdown = intervention_type == 1
                if apply_lockdown and intervention_has_not_been_applied:
                    intervention.started_as[ith_label] = 1
                    lockdown_on_label(
                        my,
                        g,
                        intervention,
                        label=ith_label,
                        rate_reduction=intervention.cfg.list_of_threshold_interventions_effects[0],
                    )

                apply_masking = intervention_type == 2
                if apply_masking and intervention_has_not_been_applied:
                    intervention.started_as[ith_label] = 2
                    masking_on_label(
                        my,
                        g,
                        intervention,
                        label=ith_label,
                        rate_reduction=intervention.cfg.list_of_threshold_interventions_effects[0],
                    )

                apply_matrix_restriction = intervention_type == 7
                if apply_matrix_restriction and intervention_has_not_been_applied:
                    intervention.started_as[ith_label] = 7
                    matrix_restriction_on_label(
                        my,
                        g,
                        intervention,
                        label=ith_label,
                    )

    elif intervention.start_interventions_by_day:
        if day in intervention.cfg.restriction_thresholds:
            for i, intervention_date in enumerate(intervention.cfg.restriction_thresholds):
                if day == intervention_date:
                    if i % 2 == 0:
                        # just looping over all labels. intervention type is not necesary with intervention by day
                        for ith_label, intervention_type in enumerate(intervention.types):

                            # if lockdown
                            if intervention.cfg.threshold_interventions_to_apply[int(i/2)] == 1:
                                print("lockdown")
                                lockdown_on_label(
                                    my,
                                    g,
                                    intervention,
                                    label=ith_label,
                                    rate_reduction=intervention.cfg.list_of_threshold_interventions_effects[int(i/2)]
                                )
                            # if masking
                            if intervention.cfg.threshold_interventions_to_apply[int(i/2)] == 2:
                                print("masks")
                                masking_on_label(
                                    my,
                                    g,
                                    intervention,
                                    label=ith_label,
                                    rate_reduction=intervention.cfg.list_of_threshold_interventions_effects[int(i/2)]
                                )
                            # if matrix restriction
                            if intervention.cfg.threshold_interventions_to_apply[int(i/2)] == 3:
                                print("matrix restriction ")
                                matrix_restriction_on_label(
                                    my,
                                    g,
                                    intervention,
                                    label=ith_label,
                                )
                    else:
                        for i_label, intervention_type in enumerate(intervention.types):
                            print("intervention removed")
                            remove_intervention_at_label(my, g, intervention, i_label)





@njit
def test_tagged_agents(my, g, intervention, day, click):
    # test everybody whose counter say we should test
    for agent in range(my.cfg.N_tot):
        # testing everybody who should be tested
        if intervention.clicks_when_tested[agent] == click:
            test_a_person(my, g, intervention, agent, click)

        if intervention.clicks_when_isolated[agent] == click:
            cut_rates_of_agent(
                my,
                g,
                intervention,
                agent,
                rate_reduction=intervention.cfg.isolation_rate_reduction,
            )

        # getting results for people
        if intervention.clicks_when_tested_result[agent] == click:
            intervention.clicks_when_tested_result[agent] = -1
            intervention.day_found_infected[agent] = day
            if intervention.apply_isolation:
                cut_rates_of_agent(
                    my,
                    g,
                    intervention,
                    agent,
                    rate_reduction=intervention.cfg.isolation_rate_reduction,
                )


#%%
# ███████ ██    ██ ███████ ███    ██ ████████ ███████
# ██      ██    ██ ██      ████   ██    ██    ██
# █████   ██    ██ █████   ██ ██  ██    ██    ███████
# ██       ██  ██  ██      ██  ██ ██    ██         ██
# ███████   ████   ███████ ██   ████    ██    ███████
#


@njit
def add_daily_events(
    my,
    g,
    day,
    agents_in_state,
    state_total_counts,
    SIR_transition_rates,
    where_infections_happened_counter,
):
    N_tot = my.cfg.N_tot
    event_size_max = my.cfg.event_size_max
    # if no max, set it to N_tot
    if event_size_max == 0:
        event_size_max = N_tot

    # if weekend increase number of events
    if (day % 7) == 0 or (day % 7) == 1:
        my.cfg.N_events = int(
            my.cfg.N_events * my.cfg.event_weekend_multiplier
        )  #  randomness x XXX

    # agents_in_event = List()

    agents_getting_infected_at_any_event = List()

    for _ in range(my.cfg.N_events):

        event_size = min(
            int(10 - np.log(np.random.rand()) * my.cfg.event_size_mean),
            event_size_max,
        )
        event_duration = -np.log(np.random.rand()) * 2 / 24  # event duration in days

        event_id = np.random.randint(N_tot)

        agents_in_this_event = List()
        while len(agents_in_this_event) < event_size:
            guest = np.random.randint(N_tot)
            rho_tmp = 0.5
            epsilon_rho_tmp = 4 / 100
            if my.dist_accepted(event_id, guest, rho_tmp) or np.random.rand() < epsilon_rho_tmp:
                agents_in_this_event.append(np.uint32(guest))

        # extract all agents that are infectious and then
        for agent in agents_in_this_event:
            if my.agent_is_infectious(agent):
                for guest in agents_in_this_event:
                    if guest != agent and my.agent_is_susceptable(guest):
                        time = np.random.uniform(0, event_duration)
                        probability = my.infection_weight[agent] * time * my.cfg.event_beta_scaling
                        if np.random.rand() < probability:
                            if guest not in agents_getting_infected_at_any_event:
                                agents_getting_infected_at_any_event.append(np.uint32(guest))

    for agent_getting_infected_at_event in agents_getting_infected_at_any_event:

        # XXX this update was needed
        my.state[agent_getting_infected_at_event] = 0
        where_infections_happened_counter[3] += 1
        agents_in_state[0].append(np.uint32(agent_getting_infected_at_event))
        state_total_counts[0] += 1
        g.total_sum_of_state_changes += SIR_transition_rates[0]
        g.cumulative_sum_of_state_changes += SIR_transition_rates[0]

        update_infection_list_for_newly_infected_agent(my, g, agent_getting_infected_at_event)
