import pandas as pd 
import os
import numpy as np
from functions import tau_int_fft, get_tau

sim_name = input("Simulation name: ") 
alg = input("Algorithm (metropolis/wolff): ")

filename = f"data/{sim_name}/metadata.csv"

metadata = pd.read_csv(filename)

for i in range(0,metadata.shape[0]) :
    L = metadata["L"][i]
    beta_i = metadata["beta_i"][i]
    beta_f = metadata["beta_f"][i]
    n_beta = metadata["n_beta"][i]
    n_measures = metadata["n_measures"][i]

    df = pd.DataFrame()

    for i in range(0,n_beta) :

        beta = beta_i + i*(beta_f - beta_i) / (n_beta-1)

        file_name = f"{sim_name}/{alg}_L{L}_beta{beta:.4f}.csv"
        data = pd.read_csv("data/" + file_name)

        df.loc[i, "beta"] = beta
        tau_abs_m, err_tau_abs_m = get_tau(data["m"].abs(),tau_int_fft)
        df.loc[i, "tau_abs_m"] = tau_abs_m   
        df.loc[i, "err_tau_abs_m"] = err_tau_abs_m
        
        os.makedirs(f"results/{sim_name}",exist_ok=True)
        df.to_csv(f"results/{sim_name}/L{L}_tau.csv",index=False)