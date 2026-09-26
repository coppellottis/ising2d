import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Stima della taglia media dei cluster Wolff in funzione dell'energia, c(E),
# a partire da due simulazioni alla stessa L (con e senza bias).
#
# Dalla teoria:  P_bias(E) = P(E) * c(E) / <c>
#   => c(E) = <c> * P_bias(E) / P(E)
# e <c> si ottiene dalla simulazione senza bias con l'improved estimator
#   <c> = N <m^2>.
#
# NB: dagli istogrammi dell'energia si ricava solo la taglia MEDIA dei cluster
# a energia fissata, c(E) = E[|C| | E], non l'intera distribuzione delle taglie.

os.makedirs("results/figure", exist_ok=True)

sim_bias = input("Simulation name (with bias): ")
sim_ok = input("Simulation name (without bias): ")
alg = input("Algorithm (metropolis/wolff): ")
L = int(input("L: "))
betas = [float(b) for b in input("beta (comma separated): ").split(",")]

N = L**2
min_counts = 20  # bin con meno conteggi esclusi

def load(sim, beta):
    data = pd.read_csv(f"data/{sim}/{alg}_L{L}_beta{beta:.4f}.csv")
    E = np.rint(data["E_per_site"].to_numpy() * N).astype(int)
    return E, data["m"].to_numpy()

def histogram(E):
    vals, counts = np.unique(E, return_counts=True)
    return vals, counts, counts / counts.sum()

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

colors = plt.cm.Blues(np.linspace(.4, 1, len(betas)))
fig, ax = plt.subplots(figsize=(12, 8))

rows = []
print(f"\nL = {L}")
print(f"{'beta':>6} | {'<c>/N = <m^2>':>14}")

for i, beta in enumerate(betas):
    Eb, _ = load(sim_bias, beta)
    Eo, mo = load(sim_ok, beta)
    vb, nb, pb = histogram(Eb)
    vo, no, po = histogram(Eo)

    c_mean = N * np.mean(mo**2)   # improved estimator

    common, ib, io = np.intersect1d(vb, vo, return_indices=True)
    ok = (nb[ib] >= min_counts) & (no[io] >= min_counts)
    ratio = pb[ib][ok] / po[io][ok]
    # errore poissoniano naive (ignora l'autocorrelazione: sottostimato)
    err_ratio = ratio * np.sqrt(1/nb[ib][ok] + 1/no[io][ok])

    c_E = c_mean * ratio
    err_c = c_mean * err_ratio
    eps = common[ok] / N

    ax.errorbar(eps, c_E/N, yerr=err_c/N, fmt="o", ms=4, capsize=3,
                color=colors[i], markerfacecolor="none", label=rf"$\beta={beta:.4f}$")

    print(f"{beta:6.4f} | {c_mean/N:14.4f}")
    rows.append(pd.DataFrame({"beta": beta, "e": eps, "c": c_E, "err_c": err_c}))

ax.set_xlabel(r"$\epsilon$")
ax.set_ylabel(r"$c(\epsilon)/N$")
ax.set_title(rf"$L={L}$")
ax.legend(loc="best")
fig.tight_layout()

os.makedirs(f"results/{sim_ok}", exist_ok=True)
pd.concat(rows).to_csv(f"results/{sim_ok}/cluster_size_vs_E_L{L}.csv", index=False)
fig.savefig(f"results/figure/{sim_bias}_vs_{sim_ok}_L{L}_cluster.pdf", bbox_inches="tight")

plt.show()
