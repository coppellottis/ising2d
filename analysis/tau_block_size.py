import pandas as pd 
import numpy as np
from functions import get_tau, tau_int_fft
import matplotlib.pyplot as plt

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

sim_name = input("Simulation name: ") 
alg = input("Algorithm (metropolis/wolff): ")

filename = f"data/{sim_name}/metadata.csv"
metadata = pd.read_csv(filename)

k_i = int(input("Initial block size for binning (min 2): "))
k_f = int(input("Final block size for binning: "))
n_k = 20
kk = np.linspace(k_i,k_f,n_k).astype(int)

err = np.zeros(n_k)

beta_c = 0.5*np.log(1+np.sqrt(2))

for i in range(0,metadata.shape[0]) :
    L = metadata["L"][i]
    beta_i = metadata["beta_i"][i]
    beta_f = metadata["beta_f"][i]
    n_beta = metadata["n_beta"][i]

    beta = np.linspace(beta_i, beta_f, n_beta) 
    beta = beta[np.argmin(np.abs(beta-beta_c))]

    file_name = f"{sim_name}/{alg}_L{L}_beta{beta:.4f}.csv"

    data = pd.read_csv("data/" + file_name)

    for k in range(0,n_k):
        tau, tau_err = get_tau(data["m"].abs(), tau_int_fft, kk[k])
        err[k] = tau_err

    ax.scatter(
        kk, err,
        label = f"L = {L}"
    )

ax.set_xlabel(r"$k$")
ax.set_xlabel(r"$\sigma_\tau$")

fig.legend()

plt.show()