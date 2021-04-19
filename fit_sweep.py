import numpy as np

from src.utils import utils
from src.simulation import simulation

from tqdm import tqdm
from contexttimer import Timer


params, start_date = utils.load_params("cfg/simulation_parameters_gatherings.yaml")

if utils.is_local_computer():
    f = 0.01
    n_steps = 1
    num_cores_max = 1
    N_runs = 3
else :
    f = 0.5
    n_steps = 3
    num_cores_max = 5
    N_runs = 1


if num_cores_max == 1 :
    verbose = True
else :
    verbose = False


if n_steps == 1 :
    noise = lambda m, d : np.round(m, 5)
else :
    noise = lambda m, d : np.round(m + np.linspace(-d, d, 2*(n_steps - 1) + 1), 5)

# Sweep around parameter set
params["beta"]               = noise(params["beta"], 0.005)
params["N_init"]             = noise(params["N_init"] * f, 500 * f)
params["N_init_UK_frac"]     = noise(params["N_init_UK_frac"], 1)

# Scale the population
params["N_tot"]  = int(params["N_tot"]  * f)
params["R_init"] = int(params["R_init"] * f)

N_files_total = 0
if __name__ == "__main__":
    with Timer() as t:

        N_files_total +=  simulation.run_simulations(params, N_runs=N_runs, num_cores_max=num_cores_max, verbose=verbose)

    print(f"\n{N_files_total:,} files were generated, total duration {utils.format_time(t.elapsed)}")
    print("Finished simulating!")



from src.analysis.helpers import *

logK, logK_sigma, beta, t_index = load_covid_index()

fraction, fraction_sigma, t_fraction = load_b117_fraction()

for subset in [{'Intervention_contact_matrices_name' : params['Intervention_contact_matrices_name'][0]}] :
    if __name__ == '__main__':
        # Load the ABM simulations
        abm_files = file_loaders.ABM_simulations(subset=subset, verbose=True)

        if len(abm_files.cfgs) == 0 :
            raise ValueError('No files found')

        lls_f     = []
        lls_s     = []

        for cfg in tqdm(
            abm_files.iter_cfgs(),
            desc='Calculating log-likelihoods',
            total=len(abm_files.cfgs)) :

            ll_f = []
            ll_s = []

            for filename in abm_files.cfg_to_filenames(cfg) :

                # Load
                I_tot_scaled, f, _, _, _, _= load_from_file(filename, start_date)

                start_date = datetime.datetime(2020, 12, 28) + datetime.timedelta(days=cfg.start_date_offset)
                end_date   = start_date + datetime.timedelta(days=cfg.day_max)

                t_tests, t_f = parse_time_ranges(start_date, end_date)

                # Evaluate
                tmp_ll_s = compute_loglikelihood((I_tot_scaled, t_tests), (logK, logK_sigma, t_index), transformation_function = lambda x : np.log(x) - beta * np.log(ref_tests))
                tmp_ll_f = compute_loglikelihood((f, t_f), (fraction, fraction_sigma, t_fraction))

                ll_s.append(tmp_ll_s)
                ll_f.append(tmp_ll_f)

            # Store loglikelihoods
            lls_s.append(np.mean(ll_s))
            lls_f.append(np.mean(ll_f))


        cfgs = [cfg for cfg in abm_files.iter_cfgs()]
        cfg_best = cfgs[np.nanargmax(lls_s)]
        ll_best = lls_s[np.nanargmax(lls_s)]
        print("--- Best parameters - Smitte ---")
        print(f"Weighted loglikelihood : {ll_best:.3f}")
        print(f"beta : {cfg_best.beta:.5f}")
        print(f"beta_UK_multiplier : {cfg_best.beta_UK_multiplier:.3f}")
        print(f"N_init : {cfg_best.N_init:.0f}")
        print(f"N_init_UK_frac : {cfg_best.N_init_UK_frac:.3f}")

        cfg_best = cfgs[np.nanargmax(lls_f)]
        ll_best = lls_f[np.nanargmax(lls_f)]
        print("--- Best parameters - B.1.1.7 ---")
        print(f"Weighted loglikelihood : {ll_best:.3f}")
        print(f"beta : {cfg_best.beta:.5f}")
        print(f"beta_UK_multiplier : {cfg_best.beta_UK_multiplier:.3f}")
        print(f"N_init : {cfg_best.N_init:.0f}")
        print(f"N_init_UK_frac : {cfg_best.N_init_UK_frac:.3f}")


        lls = np.array(lls_s) + np.array(lls_f)
        cfg_best = cfgs[np.nanargmax(lls)]
        ll_best = lls[np.nanargmax(lls)]
        print("--- Best parameters ---")
        print(f"Weighted loglikelihood : {ll_best:.3f}")
        print(f"beta : {cfg_best.beta:.5f}")
        print(f"beta_UK_multiplier : {cfg_best.beta_UK_multiplier:.3f}")
        print(f"N_init : {cfg_best.N_init:.0f}")
        print(f"N_init_UK_frac : {cfg_best.N_init_UK_frac:.3f}")

        betas     = np.array([cfg.beta                for cfg in cfgs])
        rel_betas = np.array([cfg.beta_UK_multiplier  for cfg in cfgs])

        N_init         = np.array([cfg.N_init         for cfg in cfgs])
        N_init_UK_frac = np.array([cfg.N_init_UK_frac for cfg in cfgs])

        best = lambda arr : np.array([np.mean(lls[arr == v]) for v in np.unique(arr)])
        err  = lambda arr : np.array([np.std( lls[arr == v]) for v in np.unique(arr)])


        def terminal_printer(name, arr, val, lls) :
            # Unique paramter values
            u_arr = np.unique(arr)

            # Higtest likelihood location
            I = np.argmax(u_arr == val)

            # Print the houtput
            out_string = "["
            for i in range(len(u_arr)) :
                if i == I :
                    out_string += " *" + str(u_arr[i]) + "*"
                else :
                    out_string += "  " + str(u_arr[i]) + " "
            out_string += f" ]"# sens. : {d_lls:.2g}"
            print(name + "\t" + out_string)

        print("--- Maximum likelihood value locations ---")
        terminal_printer("beta* :      ", betas,          cfg_best.beta                 , lls)
        terminal_printer("N_init* :    ", N_init,         cfg_best.N_init               , lls)
        terminal_printer("N_UK_frac* : ", N_init_UK_frac, cfg_best.N_init_UK_frac       , lls)
