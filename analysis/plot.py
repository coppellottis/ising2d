import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sim_name = input("Simulation name: ") 
alg = input("Algorithm (metropolis/wolff): ")

filename = f"data/{sim_name}/metadata.csv"

metadata = pd.read_csv(filename)

colors = plt.cm.viridis(np.linspace(0, 1, metadata.shape[0]))

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

for i in range(0,metadata.shape[0]) :
    L = metadata["L"][i]
    beta_i = metadata["beta_i"][i]
    beta_f = metadata["beta_f"][i]
    n_beta = metadata["n_beta"][i]
    n_measures = metadata["n_measures"][i]

    data = pd.read_csv(f"results/{sim_name}/L{L}_abs_m.csv")

    color = colors[i]
    ax.scatter(
        data["beta"],
        data["abs_m"], 
        marker="o", 
        color=color,
        s=30,
        linewidths=1.5,
        label=rf"$L={L}$"
    ) 
    ax.errorbar(
        data["beta"],
        data["abs_m"], 
        yerr=data["err_abs_m"],
        fmt="none",
        color=color,
        capsize=3
    )

beta_c = 0.5 * np.log(1 + np.sqrt(2))
x = np.linspace(beta_c,0.4584,1000)
y = (1 - np.sinh(2*x)**(-4))**(1/8)
ax.plot(x,y,ls="--",c="black")
ax.plot([0.4200, beta_c], [0, 0], ls="--",c="black",label=r"$L\to\infty$ (Onsager)")

ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$\langle |m| \rangle$")

ax.legend(
    frameon=False,
    loc="best"
)

fig.tight_layout()

fig.savefig(
    f"results/figure/{sim_name}_abs_m.pdf",
    bbox_inches="tight"
)

plt.show()