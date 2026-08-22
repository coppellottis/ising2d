import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sim_name = input("Simulation name: ") 
alg = input("Algorithm (metropolis/wolff): ")
L = int(input("Select the system size L for which you want to generate the histograms: "))

filename = f"data/{sim_name}/metadata.csv"

metadata = pd.read_csv(filename)
row = metadata.loc[metadata["L"] == L].iloc[0]

beta_i = row["beta_i"]
beta_f = row["beta_f"]
n_beta = row["n_beta"]
n_measures = row["n_measures"]

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
fig2, ax2 = plt.subplots(figsize=(12, 8))

cmap = plt.colormaps["plasma"]
norm = plt.Normalize(vmin=beta_i, vmax=beta_f)

beta_c = 0.5*np.log(1+np.sqrt(2))
beta_values = np.linspace(beta_i, beta_f, n_beta)

if n_beta == 1:
    betas = [beta_i]

elif n_beta == 2:
    betas = [beta_i, beta_f]

else:
    beta_middle = beta_values[np.argmin(np.abs(beta_values - beta_c))]
    betas = [beta_i, beta_middle, beta_f]

for beta in betas :
    
    file_name = f"{sim_name}/{alg}_L{L}_beta{beta:.4f}.csv"
    data = pd.read_csv("data/" + file_name)

    color = cmap(norm(beta))
    ax.hist(data["m"], bins=30, histtype="step", linewidth=2, color=color, label=rf"$\beta={beta:.4f}$",density=True)
    ax2.hist(data["E_per_site"], bins=30, histtype="step", linewidth=2, color=color, label=rf"$\beta={beta:.4f}$", density=True)

ax.set_xlabel(r"$m$")
ax.set_ylabel(r"$P_\beta(m)$")

ax2.set_xlabel(r"$E$")
ax2.set_ylabel(r"$P_\beta(E)$")

ax.text(0.03, 0.97, rf"$L={L}$", transform=ax.transAxes, ha="left", va="top", fontsize=18)
ax2.text(0.03, 0.97, rf"$L={L}$", transform=ax2.transAxes, ha="left", va="top", fontsize=18)

fig.legend(loc="upper center")
fig2.legend()

fig.tight_layout()
fig2.tight_layout()

fig.savefig(f"results/figure/{sim_name}_L{L}_m_distr.pdf")
fig2.savefig(f"results/figure/{sim_name}_L{L}_E_distr.pdf")

plt.show()