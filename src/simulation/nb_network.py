import numpy as np

from numba import njit

from src.utils import utils

#     ## ######## ########   ######  ####  #######  ##    ##          ##
##     ## ##       ##     ## ##    ##  ##  ##     ## ###   ##        ####
##     ## ##       ##     ## ##        ##  ##     ## ####  ##          ##
##     ## ######   ########   ######   ##  ##     ## ## ## ##          ##
 ##   ##  ##       ##   ##         ##  ##  ##     ## ##  ####          ##
  ## ##   ##       ##    ##  ##    ##  ##  ##     ## ##   ###          ##
   ###    ######## ##     ##  ######  ####  #######  ##    ##        ######


@njit
def v1_initialize_my(my, coordinates_raw) :
    for agent in range(my.cfg_network.N_tot) :
        set_connection_weight(my, agent)
        set_infection_weight(my, agent)
        my.coordinates[agent] = coordinates_raw[agent]


@njit
def v1_run_algo_1(my, PP, rho_tmp) :
    """ Algo 1 : density independent connection algorithm """
    agent1 = np.uint32(np.searchsorted(PP, np.random.rand()))
    while True :
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
        if do_stop :
            break


@njit
def v1_run_algo_2(my, PP, rho_tmp) :
    """ Algo 2 : increases number of connections in high-density ares """
    while True :
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
        if do_stop :
            break


@njit
def v1_connect_nodes(my) :
    """ v1 of connecting nodes. No age dependence, and a specific choice of Algo """
    if my.cfg.algo == 2 :
        run_algo = v1_run_algo_2
    else :
        run_algo = v1_run_algo_1
    PP = np.cumsum(my.connection_weight) / np.sum(my.connection_weight)
    for _ in range(my.cfg.mu / 2 * my.cfg_network.N_tot) :
        if np.random.rand() > my.cfg.epsilon_rho :
            rho_tmp = my.cfg.rho
        else :
            rho_tmp = 0.0
        run_algo(my, PP, rho_tmp)






##     ## ######## ########   ######  ####  #######  ##    ##        #######
##     ## ##       ##     ## ##    ##  ##  ##     ## ###   ##       ##     ##
##     ## ##       ##     ## ##        ##  ##     ## ####  ##              ##
##     ## ######   ########   ######   ##  ##     ## ## ## ##        #######
 ##   ##  ##       ##   ##         ##  ##  ##     ## ##  ####       ##
  ## ##   ##       ##    ##  ##    ##  ##  ##     ## ##   ###       ##
   ###    ######## ##     ##  ######  ####  #######  ##    ##       #########


@njit
def set_connection_weight(my, agent) :
    """ How introvert / extrovert you are. How likely you are at having many contacts in your network.
        Function is used determine the distribution of number of contacts. A larger my.cfg.sigma_mu gives a larger variance in number of contacts.
        Parameters :
            my (class) : Class of parameters describing the system
            agent (int) : ID of agent
    """
    if np.random.rand() < my.cfg_network.sigma_mu :
        my.connection_weight[agent] = -np.log(np.random.rand())
    else :
        my.connection_weight[agent] = 1.0


@njit
def set_infection_weight(my, agent) :
    """ How much of a super sheader are you?
        Function is used determine the distribution of number of contacts. A larger my.cfg.sigma_beta gives a larger variance individual betas
        if my.cfg.sigma_beta == 0 everybody is equally infectious.
        Parameters :
            my (class) : Class of parameters describing the system
            agent (int) : ID of agent
    """
    if np.random.rand() < my.cfg.sigma_beta :
        my.infection_weight[agent] = -np.log(np.random.rand()) * my.cfg.beta
    else :
        my.infection_weight[agent] = my.cfg.beta


@njit
def computer_number_of_cluster_retries(my, agent1, agent2) :
    """ Number of times to (re)try to connect two agents. Function is used to cluster the network more.
        A higher my.cfg.clustering_connection_retries gives higher cluster coefficient.
        Parameters :
            my (class) : Class of parameters describing the system
            agent1 (int) : ID of first agent
            agent2 (int) : ID of second agent
        returns :
           connectivity_factor (int) : Number of tries to connect to agents.
    """
    connectivity_factor = 1
    for contact in my.connections[agent1] :
        if contact in my.connections[agent2] :
            connectivity_factor += my.cfg.clustering_connection_retries
    return connectivity_factor


@njit
def cluster_retry_succesfull(my, agent1, agent2, rho_tmp) :
    """" (Re)Try to connect two agents. Returns True if succesful, else False.
        Parameters :
            my (class) : Class of parameters describing the system
            agent1 (int) : ID of first agent
            agent2 (int) : ID of second agent
            rho_tmp (float) : Characteristic distance of connections
        returns :
           Bool : Is any on the (re)tries succesfull

    """
    if my.cfg.clustering_connection_retries == 0 :
        return False
    connectivity_factor = computer_number_of_cluster_retries(my, agent1, agent2)
    for _ in range(connectivity_factor) :
        if my.dist_accepted(agent1, agent2, rho_tmp) :
            return True
    return False


@njit
def update_node_connections(
    my,
    rho_tmp,
    agent1,
    agent2,
    connection_type,
    code_version=2) :

    """ Returns True if two agents should be connected, else False
        Parameters :
            my (class) : Class of parameters describing the system
            agent1 (int) : ID of first agent
            agent2 (int) : ID of second agent
            rho_tmp (float) : Characteristic distance of connections
            connection_type (int) : ID for connection type ([House, work, other])

    """

    # checks if the two agents are the same, if they are, they can not be connected
    if agent1 == agent2 :
        return False

    # Tries to connect to agents, depending on distance.
    # if rho_tmp == 0 all distance are accepted (well mixed approximation)
    # TODO : add connection weights
    dist_accepted = rho_tmp == 0 or my.dist_accepted(agent1, agent2, rho_tmp)
    if not dist_accepted :
        # try and reconnect to increase clustering effect
        if not cluster_retry_succesfull(my, agent1, agent2, rho_tmp) :
            return False

    # checks if the two agents are already connected
    already_added = agent1 in my.connections[agent2] or agent2 in my.connections[agent1]
    if already_added :
        return False

    #checks if one contact have exceeded the contact limit. Default is no contact limit. This check is incorporated to see effect of extreme tails in N_contact distribution
    N_contacts_max = my.cfg_network.N_contacts_max
    maximum_contacts_exceeded = (N_contacts_max > 0) and (
        (len(my.connections[agent1]) >= N_contacts_max)
        or (len(my.connections[agent2]) >= N_contacts_max)
    )
    if maximum_contacts_exceeded :
        return False

    #Store the connection
    my.connections[agent1].append(np.uint32(agent2))
    my.connections[agent2].append(np.uint32(agent1))
    my.connection_status[agent1].append(True)
    my.connection_status[agent2].append(True)

    # store connection type
    if code_version >= 2 :
        connection_type = np.uint8(connection_type)
        my.connections_type[agent1].append(connection_type)
        my.connections_type[agent2].append(connection_type)

    # keep track of number of contacts
    my.number_of_contacts[agent1] += 1
    my.number_of_contacts[agent2] += 1

    return True


@njit
def place_and_connect_families(
    my, people_in_household, age_distribution_per_people_in_household, coordinates_raw) :
    """ Place agents into household, including assigning coordinates and making connections. First step in making the network.
        Parameters :
            my (class) : Class of parameters describing the system
            people_in_household (list) : distribution of number of people in households. Input data from file - source : danish statistics
            age_distribution_per_people_in_household (list) : Age distribution of households as a function of number of people in household. Input data from file - source : danish statistics
            coordinates_raw : list of coordinates drawn from population density distribution. Households are placed at these coordinates
        returns :
            mu_counter (int) : How many connections are made in households
            counter_ages(list) : Number of agents in each age group
            agents_in_age_group(nested list) : Which agents are in each age group

    """
    N_tot = my.cfg_network.N_tot

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
    while do_continue :

        agent0 = agent

        house_index = all_indices[agent]

        #Draw size of household form distribution
        N_people_in_house_index = utils.rand_choice_nb(people_in_household)
        N_people_in_house = people_index_to_value[N_people_in_house_index]

        # if N_in_house would increase agent to over N_tot,
        # set N_people_in_house such that it fits and break loop
        if agent + N_people_in_house >= N_tot :
            N_people_in_house = N_tot - agent
            do_continue = False

        # Initilaze the agents and assign them to households
        for _ in range(N_people_in_house) :

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
        for agent1 in range(agent0, agent0 + N_people_in_house) :
            for agent2 in range(agent1, agent0 + N_people_in_house) :
                if agent1 != agent2 :
                    my.connections[agent1].append(np.uint32(agent2))
                    my.connections[agent2].append(np.uint32(agent1))
                    my.connection_status[agent1].append(True)
                    my.connection_status[agent2].append(True)
                    my.connections_type[agent1].append(np.uint8(0))
                    my.connections_type[agent2].append(np.uint8(0))
                    my.number_of_contacts[agent1] += 1
                    my.number_of_contacts[agent2] += 1
                    mu_counter += 1

    agents_in_age_group = utils.nested_lists_to_list_of_array(agents_in_age_group)

    return mu_counter, counter_ages, agents_in_age_group

@njit
def place_and_connect_families_kommune_specific(
    my, people_in_household, age_distribution_per_people_in_household, coordinates_raw, kommune_ids, N_ages, verbose=False) :
    """ Place agents into household, including assigning coordinates and making connections. First step in making the network.
        Parameters :
            my (class) : Class of parameters describing the system
            people_in_household (list) : distribution of number of people in households. Input data from file - source : danish statistics
            age_distribution_per_people_in_household (list) : Age distribution of households as a function of number of people in household. Input data from file - source : danish statistics
            coordinates_raw : list of coordinates drawn from population density distribution. Households are placed at these coordinates
        returns :
            mu_counter (int) : How many connections are made in households
            counter_ages(list) : Number of agents in each age group
            agents_in_age_group(nested list) : Which agents are in each age group

    """
    N_tot = my.cfg_network.N_tot

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
    while do_continue :

        agent0 = agent
        house_index = all_indices[agent]
        coordinates = coordinates_raw[house_index]
        kommune = kommune_ids[house_index]

        #Draw size of household form distribution
        people_in_household_kommune = people_in_household[kommune, :]
        N_people_in_house_index = utils.rand_choice_nb(people_in_household_kommune)
        N_people_in_house = people_index_to_value[N_people_in_house_index]
        house_sizes[N_people_in_house_index] += 1
        # if N_in_house would increase agent to over N_tot,
        # set N_people_in_house such that it fits and break loop
        if agent + N_people_in_house >= N_tot :
            N_people_in_house = N_tot - agent
            do_continue = False

        # Initilaze the agents and assign them to households
        age_dist = age_distribution_per_people_in_household[kommune, N_people_in_house_index, :]
        for _ in range(N_people_in_house) :
            age_index = utils.rand_choice_nb(
                age_dist
            )

            #set age for agent
            age = age_index  # just use age index as substitute for age
            my.age[agent] = age
            my.kommune[agent] = kommune
            counter_ages[age_index] += 1
            agents_in_age_group[age_index].append(np.uint32(agent))

            #set coordinate for agent
            my.coordinates[agent] = coordinates_raw[house_index]

            # set weights determining extro/introvert and supersheader
            set_connection_weight(my, agent)
            set_infection_weight(my, agent)

            agent += 1

        # add agents to each others networks (connections). All people in a household know eachother
        for agent1 in range(agent0, agent0 + N_people_in_house) :
            for agent2 in range(agent1, agent0 + N_people_in_house) :
                if agent1 != agent2 :
                    my.connections[agent1].append(np.uint32(agent2))
                    my.connections[agent2].append(np.uint32(agent1))
                    my.connection_status[agent1].append(True)
                    my.connection_status[agent2].append(True)
                    my.connections_type[agent1].append(np.uint8(0))
                    my.connections_type[agent2].append(np.uint8(0))
                    my.number_of_contacts[agent1] += 1
                    my.number_of_contacts[agent2] += 1
                    mu_counter += 1

    agents_in_age_group = utils.nested_lists_to_list_of_array(agents_in_age_group)

    if verbose :
        print("House sizes :")
        print(house_sizes)

    return mu_counter, counter_ages, agents_in_age_group


@njit
def run_algo_work(my, agents_in_age_group, age1, age2, rho_tmp) :
    """ Make connection of work type. Algo locks choice of agent1, and then tries different agent2's until one is accepted.
        This algorithm gives an equal number of connections independent of local population density.
        The sssumption here is that the size of peoples workplaces is independent on where they live.
        Parameters :
            my (class) : Class of parameters describing the system
            agents_in_age_group (nested list) : list of which agents are in which age groups.
            age1 (int) : Which age group should agent1 be drawn from.
            age2 (int) : Which age group should agent2 be drawn from.
            rho_tmp(float) : characteristic distance parameter
    """

    # TODO : Add connection weights
    agent1 = np.random.choice(agents_in_age_group[age1])

    while True :
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

        if do_stop :
            break


@njit
def run_algo_other(my, agents_in_age_group, age1, age2, rho_tmp) :
    """ Make connection of other type. Algo tries different combinations of agent1 and agent2 until one combination is accepted.
        This algorithm gives more connections to people living in high populations densitity areas. This is the main driver of outbreaks being stronger in cities.
        Assumption is that you meet more people if you live in densely populated areas.
        Parameters :
            my (class) : Class of parameters describing the system
            agents_in_age_group (nested list) : list of which agents are in which age groups.
            age1 (int) : Which age group should agent1 be drawn from.
            age2 (int) : Which age group should agent2 be drawn from.
            rho_tmp(float) : characteristic distance parameter
    """
    while True :

        # TODO : Add connection weights
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
        if do_stop :
            break


@njit
def find_two_age_groups(N_ages, matrix) :
    """ Find two ages from an age connections matrix.
        Parameters :
            N_ages(int) : Number of age groups, default 9 (TODO : Should this be 10?)
            matrix : Connection matrix, how often does different age group interact.
    """
    a = 0
    ra = np.random.rand()
    for i in range(N_ages) :
        for j in range(N_ages) :
            a += matrix[i, j]
            if a > ra :
                age1, age2 = i, j
                return age1, age2
    raise AssertionError("find_two_age_groups couldn't find two age groups")


#@njit
def connect_work_and_others(
    my,
    N_ages,
    mu_counter,
    matrix_work,
    matrix_other,
    agents_in_age_group,
    verbose=True) :

    """ Overall loop to make all non household connections.
        Parameters :
            my (class) : Class of parameters describing the system
            N_ages(int) : Number of age groups, default 9 (TODO : Should this be 10?)
            matrix_work : Connection matrix, how often does different age group interact at workplaces. Combination of school and work.
            matrix_other : Connection matrix, how often does different age group interact in other section.
            agents_in_age_group(nested list) : list of which agents are in which age groups
            verbose : prints to terminal, how far the process of connecting the network is.
    """
    progress_delta_print = 0.1  # 10 percent
    progress_counter = 1

    # Store the activity (to choose the labels)
    activity_per_label = np.zeros(matrix_work.shape[0])

    # Record the activity and normalize the matrices
    for i in range(len(activity_per_label)) :
        activity_per_label[i] = np.sum(matrix_work[i, :, :]) + np.sum(matrix_other[i, :, :])
        matrix_work[i, :, :]  = matrix_work[i, :, :]  / np.sum(matrix_work[i, :, :])
        matrix_other[i, :, :] = matrix_other[i, :, :] / np.sum(matrix_other[i, :, :])

    # Convert activity to probability for choosing the label
    label_probability = np.cumsum(activity_per_label) / np.max(activity_per_label)

    mu_tot = my.cfg_network.mu / 2 * my.cfg_network.N_tot # total number of connections in the network, when done
    while mu_counter < mu_tot : # continue until all connections are made

        # Choose label
        label = np.argmax(np.random.rand() < label_probability)

        # determining if next connections is work or other.
        ra_work_other = np.random.rand()
        if ra_work_other < my.cfg_network.work_other_ratio[label] :
            matrix   = matrix_work[label, :, :]
            run_algo = run_algo_work
        else :
            matrix   = matrix_other[label, :, :]
            run_algo = run_algo_other

        #draw ages from connectivity matrix
        age1, age2 = find_two_age_groups(N_ages, matrix)

        #some number small number of connections is independent on distance. eg. you now some people on the other side of the country. Data is based on pendling data.
        if np.random.rand() > my.cfg_network.epsilon_rho :
            rho_tmp = my.cfg_network.rho
        else :
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
        if verbose :
            progress = mu_counter / mu_tot
            if progress > progress_counter * progress_delta_print :
                progress_counter += 1
                print("Connected ", round(progress * 100), r"% of work and others")