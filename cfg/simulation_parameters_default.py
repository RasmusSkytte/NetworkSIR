N_tot: 580_000  # Total number of nodes!
N_init: 100  # Initial Infected
rho: 0.0  # Spacial dependency. Average distance to connect with.
epsilon_rho: 0.04  # fraction of connections not depending on distance
mu: 40.0  # Average number of connections of a node (init: 20)
sigma_mu: 0.0  # Spread (skewness) in N connections
beta: 0.01  # Daily infection rate (SIR, init: 0-1, but beta = (2mu/N_tot)* betaSIR)
sigma_beta: 0.0  # Spread in rates, beta
lambda_E: 1.0  # E->I, Lambda(from E states)
lambda_I: 1.0  # I->R, Lambda(from I states)
algo: 2  # node connection algorithm
