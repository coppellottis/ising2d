import pandas as pd 
import os
import numpy as np
from functions import blckbstr_error, error, binder_cumulant
from concurrent.futures import ProcessPoolExecutor

def analyze(args) :
    sim_name, alg, L, beta_i, beta_f, n_beta = args

    df = pd.DataFrame() # m

    for i in range(0,n_beta) :

        if n_beta == 1:
            beta = beta_i
        else :
            beta = beta_i + i*(beta_f - beta_i) / (n_beta-1)

        file_name = f"{sim_name}/{alg}_L{L}_beta{beta:.4f}.csv"
        data = pd.read_csv("data/" + file_name)

        df.loc[i, "beta"] = beta

        df.loc[i, "m"] = data["m"].mean()
        df.loc[i, "err_m"], tau_m = error(data["m"])

        df.loc[i, "chi"] = np.var(data["m"])*beta*(L**2)
        df.loc[i, "err_chi"] = blckbstr_error(data["m"], lambda x: np.var(x)*beta*(L**2), np.floor(10*tau_m))

        df.loc[i, "abs_m"] = data["m"].abs().mean()
        df.loc[i, "err_abs_m"], tau_abs_m = error(data["m"].abs())

        df.loc[i, "chi1"] = np.var(data["m"].abs())*beta*(L**2)      
        df.loc[i, "err_chi1"] = blckbstr_error(data["m"].abs(), lambda x: np.var(x)*beta*(L**2), np.floor(10*tau_abs_m))

        df.loc[i, "e"] = data["E_per_site"].mean()
        df.loc[i, "err_e"], tau_e = error(data["E_per_site"])

        df.loc[i, "C"] = np.var(data["E_per_site"])*(beta*L)**2 ## should be C(v) instead of c(v)...
        df.loc[i, "err_C"] = blckbstr_error(data["E_per_site"], lambda x: np.var(x)*(beta*L)**2, np.floor(10*tau_e))  

        df.loc[i, "U"] = binder_cumulant(data["m"])
        df.loc[i, "err_U"] = blckbstr_error(data["m"], binder_cumulant, np.floor(10*tau_m))
        
        df.to_csv(f"results/{sim_name}/L{L}.csv",index=False)
    return

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

    os.makedirs(f"results/{sim_name}",exist_ok=True)
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        executor.map(analyze, jobs)
    