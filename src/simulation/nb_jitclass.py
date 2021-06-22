
import numpy as np
import numba as nb
from numba.experimental import jitclass
from numba.typed import List
from numba.types import ListType, DictType

from src.utils import utils



 ######  ########  ######
##    ## ##       ##    ##
##       ##       ##
##       ######   ##   ####
##       ##       ##    ##
##    ## ##       ##    ##
 ######  ##        ######

spec_cfg = {
    # Default parameters
    'version' : nb.float32,
    'beta' : nb.float32,
    'sigma_beta' : nb.float32,
    'beta_connection_type' : nb.float32[:],
    'R_guess': nb.float32,
    'algo' : nb.uint8,
    'N_init' : nb.uint16,
    'N_init_UK_frac' : nb.float32,
    'R_init' : nb.uint16,
    'initial_infection_distribution' : nb.types.unicode_type,
    'lambda_E' : nb.float32,
    'lambda_I' : nb.float32,
    # other
    'day_max' : nb.int32,
    'make_random_initial_infections' : nb.boolean,
    'weighted_random_initial_infections' : nb.boolean,
    'initialize_at_kommune_level' : nb.boolean,
    'stratified_labels' : nb.types.unicode_type,
    'simulated_tests' : nb.boolean,
    'incidence_labels' : ListType(ListType(nb.types.unicode_type)),
    'incidence_threshold' : ListType(nb.float64[: :1]), # to make the type C instead of A
    'infection_threshold' : ListType(nb.int64[: :1]), # to make the type C instead of A
    'percentage_threshold' : ListType(nb.float64[: :1]), # to make the type C instead of A
    'incidence_intervention_effect' : ListType(nb.float64[:, : :1]), # to make the type C instead of A
    'test_reference' : nb.float64,
    'incidence_reference' : nb.float64,
    'testing_exponent' : nb.float64,
    'matrix_labels' : nb.types.unicode_type,
    'matrix_label_multiplier' : nb.float32[:],
    'matrix_label_frac' : nb.float32[:],
    'initial_infected_label_weight' : nb.float32[:],
    'clustering_connection_retries' : nb.uint32,
    'beta_UK_multiplier' : nb.float32,
    'outbreak_position_UK' : nb.types.unicode_type,
    'start_date_offset' : nb.int16,
    # events
    'N_events' : nb.uint16,
    'event_size_max' : nb.uint32[:, :],
    'event_size_mean' : nb.float32[:],
    'event_beta_scaling' : nb.float32,
    'event_weekend_multiplier' : nb.float32,
    'event_rho' : nb.float32,
    'event_epsilon_rho' : nb.float32,
    'outdoor_indoor_event_ratio' : nb.float32,
    'outdoor_beta_scaling' : nb.float32,

    # lockdown-related / interventions
    'do_interventions' : nb.boolean,
    'incidence_interventions_to_apply' : nb.int8[:],
    'planned_restriction_dates' : nb.uint16[:],
    'planned_restriction_types' : nb.uint8[:],
    'continuous_interventions_to_apply' : nb.int8[:],
    'test_delay_in_clicks' : nb.int64[:],
    'results_delay_in_clicks' : nb.int64[:],
    'chance_of_finding_infected' : nb.float64[:],
    'days_looking_back' : nb.int64,
    'testing_penetration' : nb.float32[:],
    #'masking_rate_reduction' : nb.float64[ :, : :1],  # to make the type C instead if A
    #'lockdown_rate_reduction' : nb.float64[ :, : :1],  # to make the type C instead if A
    'isolation_rate_multiplier' : nb.float64[:],
    'tracing_rates' : nb.float64[:],
    'tracing_delay' : nb.int64,
    'intervention_removal_delay_in_clicks' : nb.int32,
    'Intervention_contact_matrices_name' : ListType(nb.types.unicode_type),
    'Intervention_vaccination_schedule_name' : nb.types.unicode_type,
    'Intervention_vaccination_effect_delays' : nb.int16[:],
    'Intervention_vaccination_efficacies' : nb.float32[:],
    'seasonal_list_name' : nb.types.unicode_type,
    'seasonal_strength' : nb.float64
}

@jitclass(spec_cfg)
class Config(object) :
    def __init__(self) :

        # Default parameters
        self.version = 2.0
        self.beta = 0.01
        self.sigma_beta = 0.0
        self.R_guess = 1.0
        self.algo = 2
        self.N_init = 100
        self.N_init_UK_frac = 0
        self.R_init = 0
        self.initial_infection_distribution = 'random'
        self.lambda_E = 1.0
        self.lambda_I = 1.0

        # other
        self.make_random_initial_infections     = True
        self.weighted_random_initial_infections = False
        self.initialize_at_kommune_level        = False
        self.stratified_labels                  = 'land'
        self.incidence_labels                   = List([List(['land'])])
        self.incidence_threshold                = List([np.array( [2_000.0],        dtype=np.float64)])
        self.infection_threshold                = List([np.array( [0],              dtype=np.int64)])
        self.percentage_threshold               = List([np.array( [0.0],            dtype=np.float64)])
        self.incidence_intervention_effect      = List([np.array([[1.0, 0.9, 0.9]], dtype=np.float64)])
        self.test_reference                     = 0.017         # 1.7 % percent of the population is tested daily
        self.incidence_reference                = 100_000
        self.testing_exponent                   = 0.55
        self.matrix_labels                      = 'land'
        self.matrix_label_multiplier            = np.array([1.0], dtype=np.float32)
        self.matrix_label_frac                  = np.array([0.0], dtype=np.float32)
        self.initial_infected_label_weight      = np.array([1.0], dtype=np.float32)
        self.day_max                            = 0
        self.clustering_connection_retries      = 0
        self.beta_UK_multiplier                 = 1.0

        # events
        self.N_events = 0
        self.event_size_max  = np.array([[50, 50]], dtype=np.uint32)
        self.event_size_mean = np.array([50, 50],   dtype=np.float32)
        self.event_beta_scaling = 10
        self.event_weekend_multiplier = 1.0
        self.event_rho = 0.1
        self.event_epsilon_rho = 0.04
        self.outdoor_indoor_event_ratio = 0.5
        self.outdoor_beta_scaling = 1

        # Interventions / Lockdown
        self.do_interventions = True
        self.planned_restriction_types = np.arange(0, dtype=np.uint8) # Trick to yield empty list

        self.Intervention_contact_matrices_name = List(['' for x in range(0)]) # Trick to yield empty list

        self.Intervention_vaccination_schedule_name = 'None'
        self.Intervention_vaccination_effect_delays = np.array([0.0], dtype=np.int16)
        self.Intervention_vaccination_efficacies    = np.array([0.0], dtype=np.float32)

        # Season effects
        self.seasonal_list_name = 'None'
        self.seasonal_strength = 0


nb_cfg_type = Config.class_type.instance_type

def initialize_nb_cfg(obj, cfg, spec) :
    for key, val in cfg.items() :
        if isinstance(val, list) :
            if isinstance(spec[key], nb.types.ListType) :

                # Check for nested list
                if len(val) > 0 :
                    if any(isinstance(v, list) for v in val) :

                        nested_type = getattr(obj, key)[0]

                        for ind in range(len(val)) :
                            if isinstance(nested_type, nb.typed.typedlist.List) :
                                val[ind] = List(val[ind])
                            elif isinstance(nested_type, np.ndarray) :
                                val[ind] = np.array([l for l in val[ind]])  # Prevents reflected list

                val = List(val)

            elif isinstance(spec[key], nb.types.Array) :
                val = np.array(val, dtype=spec[key].dtype.name)

            else :
                raise AssertionError(f'Got {key} : {val}, not working')

        elif isinstance(val, dict) :
            continue

        setattr(obj, key, val)
    return obj




##    ## ######## ######## ##      ##  #######  ########  ##    ##
###   ## ##          ##    ##  ##  ## ##     ## ##     ## ##   ##
####  ## ##          ##    ##  ##  ## ##     ## ##     ## ##  ##
## ## ## ######      ##    ##  ##  ## ##     ## ########  #####
##  #### ##          ##    ##  ##  ## ##     ## ##   ##   ##  ##
##   ### ##          ##    ##  ##  ## ##     ## ##    ##  ##   ##
##    ## ########    ##     ###  ###   #######  ##     ## ##    ##


spec_network = {
    # Default parameters
    'N_tot' : nb.uint32,
    'rho' : nb.float32,
    'epsilon_rho' : nb.float32,
    'mu' : nb.float32,
    'sigma_mu' : nb.float32,
    'contact_matrices_name' : nb.types.unicode_type,
    'work_matrix' : nb.float64[:, :],
    'school_matrix' : nb.float64[:, :],
    'other_matrix' : nb.float64[:, :],
    'matrix_weights' : nb.float64[:],
    'N_contacts_max' : nb.uint16,
    # ID
    'ID' : nb.uint16,
}

@jitclass(spec_network)
class Network(object) :
    def __init__(self) :

        # Default parameters
        self.N_tot = 580_000
        self.rho = 0.0
        self.epsilon_rho = 0.04
        self.mu = 40.0
        self.sigma_mu = 0.0
        self.work_matrix   = np.ones((8, 8), dtype=np.float64)
        self.school_matrix = np.ones((8, 8), dtype=np.float64)
        self.other_matrix  = np.ones((8, 8), dtype=np.float64)
        self.matrix_weights = np.array([0.33, 0.33], dtype=np.float64)
        self.N_contacts_max = 0
        self.ID = 0

nb_cfg_network_type = Network.class_type.instance_type




##     ## ##    ##
###   ###  ##  ##
#### ####   ####
## ### ##    ##
##     ##    ##
##     ##    ##
##     ##    ##


spec_my = {
    'age' : nb.uint8[:],
    'connections' : ListType(ListType(nb.uint32)),
    'connection_status' : ListType(ListType(nb.boolean)),
    'connection_type' : ListType(ListType(nb.uint8)),
    'beta_connection_type' : nb.float32[:],
    'coordinates' : nb.float32[:, :],
    'connection_weight' : nb.float32[:],
    'infection_weight' : nb.float64[:],
    'number_of_contacts' : nb.uint16[:],
    'state' : nb.int8[:],
    'sogn' : nb.uint16[:],
    'kommune' : nb.uint16[:],
    'N_sogne' : nb.uint16,
    'N_kommuner' : nb.uint16,
    'infectious_states' : ListType(nb.int64),
    'corona_type' : nb.uint8[:],
    'vaccination_type' : nb.int8[:],
    'testing_probability' : nb.float64[:],
    'restricted_status' : nb.uint8[:],
    'cfg' : nb_cfg_type,
    'cfg_network' : nb_cfg_network_type,
}


# "Nested/Mutable" Arrays are faster than list of arrays which are faster than lists of lists
@jitclass(spec_my)
class My(object) :
    def __init__(self, nb_cfg, nb_cfg_network) :
        N_tot = nb_cfg_network.N_tot
        self.age = np.zeros(N_tot, dtype=np.uint8)
        self.coordinates = np.zeros((N_tot, 2), dtype=np.float32)
        self.connections = utils.initialize_nested_lists(N_tot, np.uint32)
        self.connection_status = utils.initialize_nested_lists(N_tot, nb.boolean)
        self.connection_type = utils.initialize_nested_lists(N_tot, np.uint8)
        #self.beta_connection_type = np.array([0.7, 1.0, 1.4, 1.4, 1.0], dtype=np.float32)  # beta multiplier for [House, work, school, others, events]
        self.beta_connection_type = np.array([0.7, 0.5, 0.7, 1.4, 1.0], dtype=np.float32)  # beta multiplier for [House, work, school, others, events] (FASE 4 matrices)
        self.connection_weight = np.ones(N_tot, dtype=np.float32)
        self.infection_weight = np.ones(N_tot, dtype=np.float64)
        self.number_of_contacts = np.zeros(N_tot, dtype=nb.uint16)
        self.state = np.full(N_tot, fill_value=-1, dtype=np.int8)
        self.sogn = np.zeros(N_tot, dtype=np.uint16)
        self.kommune = np.zeros(N_tot, dtype=np.uint16)
        self.N_sogne = 1
        self.N_kommuner = 1
        self.infectious_states = List([4, 5, 6, 7])
        self.corona_type = np.zeros(N_tot, dtype=np.uint8)
        self.vaccination_type = np.zeros(N_tot, dtype=np.uint8)
        self.testing_probability = np.zeros(N_tot, dtype=nb.float64)
        self.restricted_status = np.zeros(N_tot, dtype=np.uint8)
        self.cfg = nb_cfg
        self.cfg_network = nb_cfg_network

    def dist(self, agent1, agent2) :
        point1 = self.coordinates[agent1]
        point2 = self.coordinates[agent2]
        return utils.haversine_scipy(point1, point2)

    def dist_coordinate(self, agent, coordinate) :
        return utils.haversine_scipy(self.coordinates[agent], coordinate[ : :-1])

    def dist_accepted(self, agent1, agent2, rho_tmp) :
        if np.exp(-self.dist(agent1, agent2) * rho_tmp) > np.random.rand() :
            return True
        else :
            return False

    def dist_accepted_coordinate(self, agent, coordinate, rho_tmp) :
        if np.exp(-self.dist_coordinate(agent, coordinate) * rho_tmp) > np.random.rand() :
            return True
        else :
            return False

    def agent_is_susceptible(self, agent) :
        return (self.state[agent] == -1) and (not self.agent_is_protected_by_vaccine(agent))

    def agent_is_recovered(self, agent) :
        return self.state[agent] == 8

    def agent_is_vaccinated(self, agent) :
        return self.vaccination_type[agent] != 0

    def agent_is_not_vaccinated(self, agent) :
        return self.vaccination_type[agent] == 0

    def agent_is_protected_by_vaccine(self, agent) :
        return self.vaccination_type[agent] > 0

    def agent_is_infectious(self, agent) :
        return self.state[agent] in self.infectious_states

    def agent_is_not_infectious(self, agent) :
        return not self.agent_is_infectious(agent)

    def agent_is_connected(self, agent, ith_contact) :
        return self.connection_status[agent][ith_contact]

def initialize_My(cfg) :
    nb_cfg         = initialize_nb_cfg(Config(),  cfg,         spec_cfg)
    nb_cfg_network = initialize_nb_cfg(Network(), cfg.network, spec_network)
    return My(nb_cfg, nb_cfg_network)








 ######   #### ##       ##       ########  ######  ########  #### ########
##    ##   ##  ##       ##       ##       ##    ## ##     ##  ##  ##
##         ##  ##       ##       ##       ##       ##     ##  ##  ##
##   ####  ##  ##       ##       ######    ######  ########   ##  ######
##    ##   ##  ##       ##       ##             ## ##         ##  ##
##    ##   ##  ##       ##       ##       ##    ## ##         ##  ##
 ######   #### ######## ######## ########  ######  ##        #### ########


spec_g = {
    'N_tot' : nb.uint32,
    'N_states' : nb.uint8,
    'N_infectious_states' : nb.uint8,
    'total_sum_infections' : nb.float64,
    'total_sum_of_state_changes' : nb.float64,
    'cumulative_sum' : nb.float64,
    'cumulative_sum_of_state_changes' : nb.float64[:],
    'cumulative_sum_infection_rates' : nb.float64[:],
    'transition_rates' : nb.float64[:],
    'rates' : ListType(nb.float64[ : :1]),  # ListType[array(float64, 1d, C)] (C vs. A)
    'sum_of_rates' : nb.float64[:],
    'seasonal_model' : nb.float64[:],
    'seasonal_strength' : nb.float64
}


@jitclass(spec_g)
class Gillespie(object) :
    def __init__(self, my, N_states, N_infectious_states, seasonal_model, seasonal_strength) :
        self.N_states = N_states
        self.N_infectious_states = N_infectious_states
        self.total_sum_infections = 0.0
        self.total_sum_of_state_changes = 0.0
        self.cumulative_sum = 0.0
        self.cumulative_sum_of_state_changes = np.zeros(N_states, dtype=np.float64)
        self.cumulative_sum_infection_rates  = np.zeros(N_states, dtype=np.float64)
        self.seasonal_model = seasonal_model
        self.seasonal_strength = seasonal_strength
        self._initialize_rates(my)
        self._initialize_SIR_rates(my)

    def _initialize_rates(self, my) :
        rates = List()
        for i in range(my.cfg_network.N_tot) :
            x = np.zeros(my.number_of_contacts[i])
            for j in range(my.number_of_contacts[i]) :
                x[j] = my.beta_connection_type[my.connection_type[i][j]] * my.infection_weight[i]

            rates.append(x)
        self.rates = rates
        self.sum_of_rates = np.zeros(my.cfg_network.N_tot, dtype=np.float64)

    def _initialize_SIR_rates(self, my) :
        self.transition_rates = np.zeros(self.N_states, dtype=np.float64)
        self.transition_rates[:self.N_infectious_states] = my.cfg.lambda_E
        self.transition_rates[self.N_infectious_states : 2 * self.N_infectious_states] = my.cfg.lambda_I

    def update_rates(self, my, rate, agent) :
        self.total_sum_infections += rate
        self.sum_of_rates[agent] += rate
        self.cumulative_sum_infection_rates[my.state[agent] :] += rate

    def seasonality(self, day) :
        return 1 + (self.seasonal_model[day] - 1) * self.seasonal_strength

#### ##    ## ######## ######## ########  ##     ## ######## ##    ## ######## ####  #######  ##    ##
 ##  ###   ##    ##    ##       ##     ## ##     ## ##       ###   ##    ##     ##  ##     ## ###   ##
 ##  ####  ##    ##    ##       ##     ## ##     ## ##       ####  ##    ##     ##  ##     ## ####  ##
 ##  ## ## ##    ##    ######   ########  ##     ## ######   ## ## ##    ##     ##  ##     ## ## ## ##
 ##  ##  ####    ##    ##       ##   ##    ##   ##  ##       ##  ####    ##     ##  ##     ## ##  ####
 ##  ##   ###    ##    ##       ##    ##    ## ##   ##       ##   ###    ##     ##  ##     ## ##   ###
#### ##    ##    ##    ######## ##     ##    ###    ######## ##    ##    ##    ####  #######  ##    ##

spec_intervention = {
    # Configs
    'cfg' : nb_cfg_type,
    'cfg_network' : nb_cfg_network_type,
    # Measures
    'freedom_impact' : nb.float64[:],
    'freedom_impact_list' : ListType(nb.float64),
    'R_true_list' : ListType(nb.float64),
    'R_true_list_brit' : ListType(nb.float64),
    # Testing
    'daily_pcr_tests' : nb.int64[:],     # TODO: depreciate
    'daily_antigen_tests' : nb.int64[:],     # TODO: depreciate
    'pcr_to_antigen_test_ratio' : nb.float64[:],
    'daily_test_probability' : nb.float64[:],
    'random_tests' : nb.int64[:],    # TODO: depreciate
    'day_found_infected' : nb.int32[:],
    'reason_for_test' : nb.int8[:],
    'result_of_test' : nb.int8[:],
    'positive_test_counter' : nb.uint32[:],
    'test_counter' : nb.uint32[:],
    'clicks_when_tested' : nb.int32[:],
    'clicks_when_tested_result' : nb.int32[:],
    'clicks_when_isolated' : nb.int32[:],
    'isolated' : nb.boolean[:],
    # Incidence
    'N_incidence_labels' : DictType(nb.types.unicode_type, nb.uint16),
    'incidence_labels' : ListType(nb.types.unicode_type),
    'incidence_threshold' : nb.float64[:],
    'infection_threshold' : nb.int64[:],
    'percentage_threshold' : nb.float64[:],
    'incidence_intervention_effect' : nb.float64[:, :],
    'incidence_label_map' : DictType(nb.types.unicode_type, DictType(nb.uint16, nb.uint16)),
    'inverse_incidence_label_map' : DictType(nb.types.unicode_type, DictType(nb.uint16, nb.uint16[:])),
    'agents_per_incidence_label' : DictType(nb.types.unicode_type, nb.float32[:]),
    'incidence_per_kommune' : nb.float64[:],
    'types' : nb.int8[:],
    'clicks_when_restriction_changes' : nb.int32[:],
    'clicks_looking_back' : nb.int64,
    # Ouputs
    'stratified_label_map' : DictType(nb.uint16, nb.uint16),
    # Vaccinations
    'vaccinations_per_age_group' : nb.int64[:, :, :],
    'vaccination_schedule' : nb.int32[:, :],
    # Contact matrcies
    'N_matrix_labels' : nb.uint16,
    'matrix_label_map' : DictType(nb.uint16, nb.uint16),
    'work_matrix_restrict' : nb.float64[:, :, :, :],
    'school_matrix_restrict' : nb.float64[:, :, :, :],
    'other_matrix_restrict' : nb.float64[:, :, :, :],
    # Events
    'event_size_max' : nb.uint32[:],
    # Options
    'nts' : nb.float32,
    'verbose' : nb.boolean,
}


@jitclass(spec_intervention)
class Intervention(object) :
    """
    - N_incidence_labels : Number of incidence labels. "Label" here can be sogn, kommune, landsdel, region or land.
    - N_matrix_labels : Number of matrix labels. "Label" here can be sogn, kommune, landsdel, region or land.
    - agents_per_incidence_label : count how many agent belong to a particular label

    - day_found_infected : -10_000 if not infected, otherwise the day of infection

    - reason_for_test :
         0 : symptoms
         1 : random_test
         2 : tracing,
        -1 : No reason yet (or no impending tests). You can still be tested again later on (if negative test)

    - results_of_test :
         0 : negative
         1 : positive
        -1 : no result yet

    - test_counter : counter of how many were  tested due to reason 0, 1 or 2

    - positive_test_counter : counter of how many were found tested positive due to reasom 0, 1 or 2

    - clicks_when_tested : When you were tested measured in clicks (10 clicks = 1 day)

    - clicks_when_tested_result : When you get your test results measured in clicks

    - clicks_when_isolated : when you were told to go in isolation and be tested

    - isolated : boolean to store the state of the agent

    - threshold_interventions : array to keep count of which intervention are at place at which label
        0 : Do nothing
        1 : lockdown (cut some contacts and reduce the rest),
        2 : Masking (reduce some contacts),
        3 : Matrix based (used loaded contact matrices. )

    - continuous_interventions : array to keep count of which intervention are at place at which label
        # 0 : Do nothing
        # 1 : tracing (infected and their connections)
        # 2 : Test people with symptoms
        # 3 : Isolate
        # 4 : Random Testing
        # 5 : vaccinations

    - verbose : Prints status of interventions and removal of them

    """

    def __init__(
        self,
        my,
        incidence_label_map,
        inverse_incidence_label_map,
        N_incidence_labels,
        agents_per_incidence_label,
        matrix_label_map,
        N_matrix_labels,
        stratified_label_map,
        vaccinations_per_age_group,
        vaccination_schedule,
        work_matrix_restrict,
        school_matrix_restrict,
        other_matrix_restrict,
        daily_pcr_tests,
        daily_antigen_tests,
        daily_test_modifer,
        pcr_to_antigen_test_ratio,
        nts,
        verbose=False) :

        self.cfg         = my.cfg
        self.cfg_network = my.cfg_network


        self.freedom_impact                  = np.full(self.cfg_network.N_tot, fill_value=0.0, dtype=np.float64)
        self.freedom_impact_list             = List([0.0])
        self.R_true_list                     = List([0.0])
        self.R_true_list_brit                = List([0.0])


        self.day_found_infected              = np.full(self.cfg_network.N_tot, fill_value=-10_000, dtype=np.int32)
        self.reason_for_test                 = np.full(self.cfg_network.N_tot, fill_value=-1, dtype=np.int8)
        self.result_of_test                  = np.full(self.cfg_network.N_tot, fill_value=-1, dtype=np.int8)
        self.positive_test_counter           = np.zeros(3, dtype=np.uint32)
        self.test_counter                    = np.zeros(3, dtype=np.uint32)
        self.clicks_when_tested              = np.full(self.cfg_network.N_tot, fill_value=-10_000, dtype=np.int32)
        self.clicks_when_tested_result       = np.full(self.cfg_network.N_tot, fill_value=-10_000, dtype=np.int32)
        self.clicks_when_isolated            = np.full(self.cfg_network.N_tot, fill_value=-10_000, dtype=np.int32)
        self.isolated                        = np.full(self.cfg_network.N_tot, fill_value=False, dtype=nb.boolean)


        self.N_incidence_labels              = N_incidence_labels
        self.incidence_labels                = self.cfg.incidence_labels[0]
        self.incidence_threshold             = self.cfg.incidence_threshold[0]
        self.infection_threshold             = self.cfg.infection_threshold[0]
        self.percentage_threshold            = self.cfg.percentage_threshold[0]
        self.incidence_intervention_effect   = self.cfg.incidence_intervention_effect[0]
        self.incidence_label_map             = incidence_label_map
        self.inverse_incidence_label_map     = inverse_incidence_label_map


        self.agents_per_incidence_label      = agents_per_incidence_label
        self.incidence_per_kommune           = np.zeros(my.N_kommuner, dtype=np.float64)

        self.types                           = np.zeros(my.N_sogne, dtype=np.int8)
        self.clicks_when_restriction_changes = np.full(my.N_sogne,  fill_value=-1, dtype=np.int32)
        self.clicks_looking_back             = int(np.round(self.cfg.days_looking_back / nts))

        self.N_matrix_labels                 = N_matrix_labels
        self.matrix_label_map                = matrix_label_map
        self.work_matrix_restrict            = work_matrix_restrict
        self.school_matrix_restrict          = school_matrix_restrict
        self.other_matrix_restrict           = other_matrix_restrict

        self.daily_pcr_tests                 = daily_pcr_tests                  # TODO Depreciate
        self.daily_antigen_tests             = daily_antigen_tests              # TODO Depreciate
        self.random_tests                    = daily_pcr_tests + (0.5 * daily_antigen_tests).astype(daily_antigen_tests.dtype)      # TODO Depreciate
        self.daily_test_probability          = daily_test_modifer
        self.pcr_to_antigen_test_ratio       = pcr_to_antigen_test_ratio

        self.stratified_label_map            = stratified_label_map

        self.vaccinations_per_age_group      = vaccinations_per_age_group
        self.vaccination_schedule            = vaccination_schedule

        self.event_size_max                  = self.cfg.event_size_max[0]

        self.nts                             = nts

        self.verbose = verbose


    def agent_not_found_positive(self, agent) :
        return self.day_found_infected[agent] == -1

    def agent_found_positive(self, agent) :
        return not self.agent_not_found_positive(agent)

    @property
    def apply_interventions(self) :
        return self.cfg.do_interventions

    @property
    def apply_interventions_on_label(self) :
        return (self.apply_matrix_restriction or self.start_interventions_by_incidence)

    @property
    def apply_tracing(self) :
        return 1 in self.cfg.continuous_interventions_to_apply

    @property
    def apply_symptom_testing(self) :
        return 2 in self.cfg.continuous_interventions_to_apply

    @property
    def apply_isolation(self) :
        return 3 in self.cfg.continuous_interventions_to_apply

    @property
    def apply_random_testing(self) :
        return 4 in self.cfg.continuous_interventions_to_apply

    @property
    def apply_testing(self) :
        return self.apply_symptom_testing or self.apply_random_testing

    @property
    def apply_vaccinations(self) :
        return 5 in self.cfg.continuous_interventions_to_apply

    @property
    def apply_matrix_restriction(self) :
        return 1 in self.cfg.planned_restriction_types

    @property
    def start_interventions_by_day(self) :
        return len(self.cfg.planned_restriction_types) > 0

    @property
    def start_interventions_by_incidence(self) :
        return 0 not in self.cfg.incidence_interventions_to_apply

    @property
    def apply_lockdowns(self) :
        return 1 in self.cfg.incidence_interventions_to_apply

    @property
    def apply_increased_testing(self) :
        return 2 in self.cfg.incidence_interventions_to_apply

    @property
    def use_only_raw_incidence(self) :
        return 3 in self.cfg.incidence_interventions_to_apply