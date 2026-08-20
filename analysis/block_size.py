import pandas as pd
import numpy as np
from functions import blckbstr_error
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os


### DEPRECATED

def mean(x):
    return np.mean(x)


if __name__ == "__main__":
    sim_name = input("Simulation name: ")
    alg = input("Algorithm (metropolis/wolff): ")

    metadata = pd.read_csv(f"data/{sim_name}/metadata.csv")

    k_i = int(input("Initial block size (min 2): "))
    k_f = int(input("Final block size: "))
    n_k = int(input("Number of windows: "))

    kk = np.linspace(k_i, k_f, n_k).astype(int)

    jobs = []

    L = metadata["L"][0]
    beta = metadata["beta_i"][0]
    data = pd.read_csv(f"data/{sim_name}/{alg}_L{L}_beta{beta:.4f}.csv")
    
    for k in kk:
        jobs.append((data["m"].abs(), mean, k))

        # number of core (RAM 8 Gi+2 Gi swap)
        n_processes = 2

        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            errors = list(executor.map(blckbstr_error, jobs))


    
    # style
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 18,

        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,

        "axes.linewidth": 1.0,

        "xtick.direction": "in",
        "ytick.direction": "in",

        "xtick.top": True,
        "ytick.right": True,

        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,

        "legend.fontsize": 18,
    })

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(kk, errors, "o-")

    plt.show()