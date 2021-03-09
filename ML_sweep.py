import numpy as np

from src.utils import utils
from src.simulation import simulation

from tqdm import tqdm
from contexttimer import Timer


params, start_date = utils.load_params("cfg/simulation_parameters_fit_2021_fase2.yaml")

if utils.is_local_computer():
    f = 0.01
    n_steps = 1
    num_cores_max = 1
    N_runs = 1
else :
    f = 0.2
    n_steps = 2
    num_cores_max = 15
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
#params["beta"]               = [0.02, 0.025, 0.03, 0.035, 0.04]
params["beta"]               = noise(params["beta"], 0.005)
#params["beta_UK_multiplier"] = [1.5]
#params["beta_UK_multiplier"] = noise(params["beta_UK_multiplier"], 0.1)

params["N_init"]             = noise(params["N_init"] * f, 500 * f)
#params["N_init"] = int(params["N_init"] * f)

#params["N_init_UK_frac"]     = [0.02, 0.025, 0.03]
#params["N_init_UK_frac"]     = noise(params["N_init_UK_frac"], 0.005)

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
        abm_files = file_loaders.ABM_simulations(base_dir='Output/ABM', subset=subset, verbose=True)

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
                I_tot_scaled, f, _, _, _= load_from_file(filename)


                start_date = datetime.datetime(2020, 12, 28) + datetime.timedelta(days=cfg.start_date_offset)
                end_date   = start_date + datetime.timedelta(days=cfg.day_max)

                t_tests = pd.date_range(start = start_date, end = end_date, freq = "D")
                t_f     = pd.date_range(start = start_date, end = end_date, freq = "W-SUN")[1:]


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

        betas     = np.array([cfg.beta               for cfg in cfgs])
        rel_betas = np.array([cfg.beta_UK_multiplier for cfg in cfgs])

        N_init         = np.array([cfg.N_init         for cfg in cfgs])
        N_init_UK_frac = np.array([cfg.N_init_UK_frac for cfg in cfgs])

        best = lambda arr : np.array([np.mean(lls[arr == v]) for v in np.unique(arr)])
        err  = lambda arr : np.array([np.std( lls[arr == v]) for v in np.unique(arr)])

        if False:# utils.is_local_computer() :
            rc_params.set_rc_params()

            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
            axes = axes.flatten()

            axes[0].errorbar(np.unique(betas), best(betas), yerr=err(betas), fmt="o")
            axes[0].scatter(cfg_best.beta, ll_best, fmt="x")
            axes[0].set_xlabel('beta')

            axes[1].errorbar(np.unique(N_init), best(N_init), yerr=err(N_init), fmt="o")
            axes[1].scatter(cfg_best.N_init, ll_best, fmt="x")
            axes[1].set_xlabel('N_init')

            axes[2].errorbar(np.unique(rel_betas), best(rel_betas), yerr=err(rel_betas), fmt="o")
            axes[2].scatter(cfg_best.beta_UK_multiplier, ll_best, fmt="x")
            axes[2].set_xlabel('rel. beta')

            axes[3].errorbar(np.unique(N_init_UK_frac), best(N_init_UK_frac), yerr=err(N_init_UK_frac), fmt="o")
            axes[3].scatter(cfg_best.N_init_UK_frac, ll_best, fmt="x")
            axes[3].set_xlabel('N_init_UK_frac')

            plt.savefig('Figures/LogLikelihood_parameters.png')


        def terminal_printer(name, arr, val, lls) :
            # Unique paramter values
            u_arr = np.unique(arr)

            # Average loglikelihood value for parameter sets
            #lls_param = np.zeros(np.shape(u_arr))
            #for i, val in enumerate(u_arr) :
            #    lls_param[i] = np.nanmean(lls[arr == val])

            #s_lls = np.array(sorted(lls_param))
            #s_lls = s_lls[~np.isnan(s_lls)]
            #d_lls = s_lls[-1] / s_lls[-2]

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
        #terminal_printer("rel_beta* :  ", rel_betas,      cfg_best.beta_UK_multiplier   , lls)
        #terminal_printer("N_init* :    ", N_init,         cfg_best.N_init               , lls)
        #terminal_printer("N_UK_frac* : ", N_init_UK_frac, cfg_best.N_init_UK_frac       , lls)
