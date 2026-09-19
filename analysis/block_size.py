import pandas as pd
import numpy as np
from functions import blckbstr_error
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os


### DEPRECATED
# (kept as a manual diagnostic: plots the block-bootstrap error on <|m|>
# against block size k, useful for eyeballing whether blckbstr_error's
# default/auto block size has settled into the plateau it's supposed to.)

def mean(x):
    return np.mean(x)

# NOTE (bugfix): module-level (not nested) so it can be pickled and sent
# to worker processes. Unpacks each (x, f, k) job before calling
# blckbstr_error -- ProcessPoolExecutor.map(func, iterable) calls
# func(item), not func(*item), so passing blckbstr_error directly used to
# hand it a single 3-tuple as its first positional argument and raise a
# TypeError.
def _run_blckbstr_job(job):
    x, f, k = job
    return blckbstr_error(x, f, k)


if __name__ == "__main__":
    sim_name = input("Simulation name: ")
    alg = input("Algorithm (metropolis/wolff): ")

    metadata = pd.read_csv(f"data/{sim_name}/metadata.csv")

    k_i = int(input("Initial block size (min 2): "))
    k_f = int(input("Final block size: "))
    n_k = int(input("Number of windows: "))

    kk = np.linspace(k_i, k_f, n_k).astype(int)

    L = metadata["L"][0]
    beta = metadata["beta_i"][0]
    data = pd.read_csv(f"data/{sim_name}/{alg}_L{L}_beta{beta:.4f}.csv")

    jobs = [(data["m"].abs(), mean, k) for k in kk]

    # number of core (RAM 8 Gi+2 Gi swap)
    n_processes = 2

    # NOTE (bugfix): the pool used to be created *inside* the loop that
    # builds "jobs", so on iteration m it recomputed all m already-processed
    # k's again from scratch (O(n_k^2) total work for O(n_k) useful
    # results) instead of running once after the full job list was built.
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        errors = list(executor.map(_run_blckbstr_job, jobs))

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
