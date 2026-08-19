import pandas as pd
import numpy as np
from functions import get_tau, tau_int_fft
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os


def compute_errors(args):

    sim_name, alg, L, beta_i, beta_f, n_beta, kk = args

    beta_c = 0.5 * np.log(1 + np.sqrt(2))

    beta = np.linspace(beta_i, beta_f, n_beta)
    beta = beta[np.argmin(np.abs(beta - beta_c))]

    file_name = f"data/{sim_name}/{alg}_L{L}_beta{beta:.4f}.csv"

    data = pd.read_csv(file_name)

    os.makedirs(f"results/{sim_name}/blocking",exist_ok=True)
    path = f"results/{sim_name}/blocking/L{L}_tau_err.csv"

    for k, block_size in enumerate(kk):
        df = pd.DataFrame()            
        tau, tau_err = get_tau(data["m"].abs(),tau_int_fft,block_size)
        df.loc[k, "err_tau"] = tau_err
        df.loc[k, "k"] = block_size
        print(tau)

    df.to_csv(path,mode="a", header=not os.path.exists(path), index=False)

    return


if __name__ == "__main__":
    sim_name = input("Simulation name: ")
    alg = input("Algorithm (metropolis/wolff): ")

    metadata = pd.read_csv(f"data/{sim_name}/metadata.csv")

    k_i = int(input("Initial block size for binning (min 2): "))
    k_f = int(input("Final block size for binning: "))
    n_k = int(input("Number of windows: "))

    kk = np.linspace(k_i, k_f, n_k).astype(int)

    jobs = []

    for i in range(metadata.shape[0]):

        L = metadata["L"][i]
        beta_i = metadata["beta_i"][i]
        beta_f = metadata["beta_f"][i]
        n_beta = metadata["n_beta"][i]

        jobs.append(
            (
                sim_name,
                alg,
                L,
                beta_i,
                beta_f,
                n_beta,
                kk
            )
        )

    # number of core (RAM 8 Gi+2 Gi swap)
    n_processes = 2

    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        executor.map(compute_errors, jobs)