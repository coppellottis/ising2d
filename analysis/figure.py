import pandas as pd
import matplotlib.pyplot as plt

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

ax.scatter(data.iloc[:, 0],data.iloc[:, 1], marker="s", s=30,facecolors="none",edgecolors="tab:blue",linewidths=1.5,label=r"$L=8$") 
ax.errorbar(
    data.iloc[:, 0],
    data.iloc[:, 1],
    yerr=data.iloc[:, 2],
    fmt="none",
    ecolor="tab:blue",
    capsize=3
)
#ax.scatter(data.iloc[:, 0],data.iloc[:, 3], marker="^", s=30,facecolors="none",edgecolors="tab:green",linewidths=1.5,label=r"$L=16$") 
#ax.scatter(data.iloc[:, 0],data.iloc[:, 5], marker="d", s=30,facecolors="none",edgecolors="tab:red",linewidths=1.5,label=r"$L=32$") 
ax.scatter(data.iloc[:, 0],data.iloc[:, 7], marker="o", s=30,facecolors="none",edgecolors="tab:purple",linewidths=1.5,label=r"$L=64$") 
ax.errorbar(
    data.iloc[:, 0],
    data.iloc[:, 7],
    yerr=data.iloc[:, 8],
    fmt="none",
    ecolor="tab:purple",
    capsize=3
)
ax.scatter(data.iloc[:, 0],data.iloc[:, 9], marker="v", s=30,facecolors="none",edgecolors="tab:orange",linewidths=1.5,label=r"$L=128$") 
ax.errorbar(
    data.iloc[:, 0],
    data.iloc[:, 9],
    yerr=data.iloc[:, 10],
    fmt="none",
    ecolor="tab:orange",
    capsize=3
)

ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$\langle |m| \rangle$")

#plt.xticks([.35,.40,.45,.50,.55])

ax.legend(
    frameon=False,
    loc="best"
)

fig.tight_layout()

fig.savefig(
    "results/figure/metropolis_energy.pdf",
    bbox_inches="tight"
)

plt.show()