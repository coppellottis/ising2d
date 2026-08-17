import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sim_name = input("Simulation name: ") 
alg = input("Algorithm (metropolis/wolff): ")
z = float(input("Dynamic exponent z': "))
nu = 1
beta_c = 0.5*np.log(1+np.sqrt(2))

filename = f"data/{sim_name}/metadata.csv"

metadata = pd.read_csv(filename)

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

for i in range(0,metadata.shape[0]):
    L = metadata["L"][i]
    data = pd.read_csv(f"results/{sim_name}/L{L}_tau.csv")
    tau = data["tau_abs_m"]
    err_tau = data["err_tau_abs_m"]
    beta = data["beta"]

    x = (beta-beta_c)*(L**(1/nu))
    y = tau/(L**z)
    yerr = err_tau/(L**z)

    ax.errorbar(
        x, y,
        yerr=yerr,
        fmt='o',
        markersize=5,
        capsize=3,
        label=f"{L}"
    )

ax.set_xlabel(r"$(\beta-\beta_c)L^{1/\nu}$")
ax.set_ylabel(r"$\tau_{int}^{|m|}/L^z$")
fig.legend(loc="upper left")

fig.tight_layout()

fig.savefig(f"results/figure/{sim_name}_fss_tau(z{z}).pdf",bbox_inches="tight")
plt.show()