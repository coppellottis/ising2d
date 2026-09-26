import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from functions import tau_int_fft, get_tau

sim_name = input("Simulation name: ")
alg = input("Algorithm (metropolis/wolff): ")

filename = f"data/{sim_name}/metadata.csv"

metadata = pd.read_csv(filename)

for i in tqdm(range(0,metadata.shape[0]), desc="Lattice sizes", position=0, dynamic_ncols=True) :
    L = metadata["L"][i]
    beta_i = metadata["beta_i"][i]
    beta_f = metadata["beta_f"][i]
    n_beta = metadata["n_beta"][i]
    n_measures = metadata["n_measures"][i]

    df = pd.DataFrame()

    for j in tqdm(range(0,n_beta), desc=f"L={L:<4} beta", position=1, leave=False, dynamic_ncols=True) :

        if n_beta == 1 : 
            beta = beta_i
        else:
            beta = np.linspace(beta_i, beta_f, n_beta)
            
        file_name = f"{sim_name}/{alg}_L{L}_beta{beta:.4f}.csv"
        data = pd.read_csv("data/" + file_name)

        df.loc[j, "beta"] = beta
        tag = f"{alg} L={L} beta={beta:.4f}"
        tau_abs_m, err_tau_abs_m = get_tau(data["m"].abs(), tau_int_fft, label=f"{tag} obs=|m|")
        df.loc[j, "tau_abs_m"] = tau_abs_m
        df.loc[j, "err_tau_abs_m"] = err_tau_abs_m
        tau_e, err_tau_e = get_tau(data["E_per_site"], tau_int_fft, label=f"{tag} obs=E")
        df.loc[j, "tau_E"] = tau_e
        df.loc[j, "err_tau_E"] = err_tau_e
        
        os.makedirs(f"results/{sim_name}",exist_ok=True)
        df.to_csv(f"results/{sim_name}/L{L}_tau.csv",index=False)