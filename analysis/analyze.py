import pandas as pd
import os
import numpy as np
from functions import blckbstr_error, error, binder_cumulant, safe_block_k
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def analyze(args) :
    sim_name, alg, L, beta_i, beta_f, n_beta, position = args

    df = pd.DataFrame() # m

    # position > 0: each worker process gets its own fixed terminal row
    # (position 0 is reserved for the overall "L completati" bar in the
    # main process), the same idea as the per-thread rows in the C
    # simulation's progress bars.
    for i in tqdm(range(0,n_beta), desc=f"L={L:<4} beta", position=position, leave=False, dynamic_ncols=True) :

        if n_beta == 1:
            beta = beta_i
        else :
            beta = beta_i + i*(beta_f - beta_i) / (n_beta-1)

        file_name = f"{sim_name}/{alg}_L{L}_beta{beta:.4f}.csv"
        data = pd.read_csv("data/" + file_name)
        tag = f"{alg} L={L} beta={beta:.4f}"

        df.loc[i, "beta"] = beta

        df.loc[i, "m"] = data["m"].mean()
        df.loc[i, "err_m"], tau_m = error(data["m"], label=f"{tag} obs=m")

        # NOTE (bugfix): tau_m/tau_abs_m/tau_e used to be plugged straight
        # into np.floor(10*tau_x) as the bootstrap block size. When
        # tau_int_fft doesn't converge (tau_x is None -- most likely for
        # large L close to beta_c, exactly where critical slowing down is
        # worst), that raised a TypeError/ValueError and killed this whole
        # job, silently dropping every beta for that L from the results.
        # safe_block_k() falls back to a reasonable block size instead and
        # prints which (L, beta, observable) needed the fallback.
        k_m = safe_block_k(tau_m, len(data["m"]), label=f"{tag} obs=m")
        df.loc[i, "chi"] = np.var(data["m"])*beta*(L**2)
        df.loc[i, "err_chi"] = blckbstr_error(data["m"], lambda x: np.var(x)*beta*(L**2), k_m)

        df.loc[i, "abs_m"] = data["m"].abs().mean()
        df.loc[i, "err_abs_m"], tau_abs_m = error(data["m"].abs(), label=f"{tag} obs=|m|")

        k_abs_m = safe_block_k(tau_abs_m, len(data["m"]), label=f"{tag} obs=|m|")
        df.loc[i, "chi1"] = np.var(data["m"].abs())*beta*(L**2)
        df.loc[i, "err_chi1"] = blckbstr_error(data["m"].abs(), lambda x: np.var(x)*beta*(L**2), k_abs_m)

        df.loc[i, "e"] = data["E_per_site"].mean()
        df.loc[i, "err_e"], tau_e = error(data["E_per_site"], label=f"{tag} obs=E")

        k_e = safe_block_k(tau_e, len(data["E_per_site"]), label=f"{tag} obs=E")
        df.loc[i, "C"] = np.var(data["E_per_site"])*(beta*L)**2 ## should be C(v) instead of c(v)...
        df.loc[i, "err_C"] = blckbstr_error(data["E_per_site"], lambda x: np.var(x)*(beta*L)**2, k_e)

        df.loc[i, "U"] = binder_cumulant(data["m"])
        df.loc[i, "err_U"] = blckbstr_error(data["m"], binder_cumulant, k_m)

        df.to_csv(f"results/{sim_name}/L{L}.csv",index=False)
    return L

if __name__ == "__main__":

    sim_name = input("Simulation: ").strip()
    alg = input("Algorithm (metropolis/wolff): ").strip()
    filename = f"data/{sim_name}/metadata.csv"

    metadata = pd.read_csv(filename)
    jobs = []

    for i in range(0,metadata.shape[0]) :

        L = metadata["L"][i]
        beta_i = metadata["beta_i"][i]
        beta_f = metadata["beta_f"][i]
        n_beta = metadata["n_beta"][i]
        n_measures = metadata["n_measures"][i]

        jobs.append((sim_name, alg, L, beta_i, beta_f, n_beta))

    n_processes = 4

    # Round-robin a terminal row (1..n_processes) to each job so that up to
    # n_processes concurrent workers can each show their own "L=.. beta"
    # bar without clobbering each other; row 0 is reserved for the overall
    # bar below.
    jobs = [(*job, 1 + i % n_processes) for i, job in enumerate(jobs)]

    os.makedirs(f"results/{sim_name}",exist_ok=True)
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = [executor.submit(analyze, job) for job in jobs]
        # NOTE (bugfix, kept): a bare executor.map(...) iterator that's
        # never consumed silently discards exceptions raised inside a
        # worker process -- analyze.py would print nothing, exit 0, and
        # leave results/L*.csv missing or half-written for whichever L's
        # job crashed. Calling fut.result() below re-raises here instead.
        with tqdm(total=len(jobs), desc="L completati", position=0, dynamic_ncols=True) as outer:
            for fut in as_completed(futures):
                fut.result()
                outer.update(1)
