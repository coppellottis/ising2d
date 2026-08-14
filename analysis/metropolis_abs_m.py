import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

L_values = [8, 16, 32, 64, 128]

colors = plt.cm.viridis(np.linspace(0, 1, len(L_values)))

# Stile generale
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

data = pd.read_csv("results/metropolis_abs_m.csv")

fig, ax = plt.subplots(figsize=(12, 8))

for i, L in enumerate(L_values) :
    color = colors[i]
    ax.scatter(
        data.iloc[:, 0],
        data.iloc[:, (2*i+1)], 
        marker="o", 
        color=color,
        s=30,
        linewidths=1.5,
        label=rf"$L={L}$"
    ) 
    ax.errorbar(
        data.iloc[:, 0],
        data.iloc[:, (2*i+1)],
        yerr=data.iloc[:, 2*(i+1)],
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
    "results/figure/metropolis_abs_m.pdf",
    bbox_inches="tight"
)

plt.show()