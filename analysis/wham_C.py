import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from tqdm import tqdm
from scipy.special import logsumexp
from scipy.optimize import curve_fit
from functions import format_error

def specific_heat(beta, E_bins, log_gE, N):

    log_w = log_gE - beta * E_bins
    f = logsumexp(log_w)

    w = np.exp(log_w - f) 
 
    e_mean = np.sum(w * E_bins)
    e2_mean = np.sum(w * E_bins**2)
    var_E = e2_mean - e_mean ** 2
    cv = beta**2 * var_E / N

    return cv

sim_name = input("Simulation name: ") 
alg = input("Algorithm (metropolis/wolff): ")

filename = f"data/{sim_name}/metadata.csv"

metadata = pd.read_csv(filename)
wham_results = pd.read_csv(f"results/{sim_name}/wham_results_n.csv")

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
colors = plt.cm.Blues(np.linspace(.2,1,metadata.shape[0]))

beta_max = []
CV_max = [] 

LL = metadata["L"]

for i in tqdm(range(0,metadata.shape[0]), desc="Calcolo C_V per L", position=0, dynamic_ncols=True) :
    L = LL[i]
    beta_i = metadata["beta_i"][i]
    beta_f = metadata["beta_f"][i]

    N = L**2
    bins = wham_results["E"]
    log_gE = wham_results[f"{L}"]

    data = pd.read_csv(f"results/{sim_name}/L{L}.csv")
    ax.errorbar(data["beta"], data["C"], yerr=data["err_C"], fmt="o", capsize=3, color=colors[i], markerfacecolor='none', label=rf"$L={L}$")

    x = np.linspace(beta_i, beta_f, 1000)
    y = [specific_heat(beta, bins, log_gE, N) for beta in x]
    line = ax.plot(x,y, color=colors[i])[0]

    res = minimize_scalar(lambda beta: -specific_heat(beta, bins, log_gE, N), bounds=(beta_i, beta_f), method='bounded')
    beta_max.append(res.x)
    CV_max.append(-res.fun)

    ax.plot(beta_max, CV_max, marker="o", linestyle="None", color="black", ms=5)

beta_max = np.asarray(beta_max)
CV_max = np.asarray(CV_max)




# aggiusto i grafici
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$C_V$")

ax.legend(loc="best")
fig.tight_layout()

fig.savefig(f"results/figure/{sim_name}_C_wham.pdf", bbox_inches="tight")
plt.show()
