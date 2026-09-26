import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Confronto tra le distribuzioni dell'energia di due simulazioni alla stessa L:
# una con misura a sweep equivalente (bias) e una con n_cl fisso (senza bias).
#   fig  -> P(E) delle due simulazioni per ogni beta
#   fig2 -> rapporto P_bias(E)/P_ok(E): per P_meas = pi*c/<c> stima W(E) = E[c|E]/<c>

os.makedirs("results/figure", exist_ok=True)

sim_bias = input("Simulation name (with bias): ")
sim_ok = input("Simulation name (without bias): ")
alg = input("Algorithm (metropolis/wolff): ")
L = int(input("L: "))
betas = [float(b) for b in input("beta (comma separated): ").split(",")]

N = L**2
min_counts = 20  # bin con meno conteggi esclusi dal rapporto

def load_energy(sim, beta):
    # energia totale, intera con passo 4
    data = pd.read_csv(f"data/{sim}/{alg}_L{L}_beta{beta:.4f}.csv")
    return np.rint(data["E_per_site"].to_numpy() * N).astype(int)

def histogram(E):
    # bin = livelli esatti dell'energia
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
fig2, ax2 = plt.subplots(figsize=(12, 8))

print(f"\nL = {L}")
print(f"{'beta':>6} | {'e bias':>9} {'e ok':>9} {'diff':>9} | {'C bias':>7} {'C ok':>7} {'ratio':>6}")

for i, beta in enumerate(betas):
    Eb = load_energy(sim_bias, beta)
    Eo = load_energy(sim_ok, beta)
    vb, nb, pb = histogram(Eb)
    vo, no, po = histogram(Eo)

    # distribuzioni: linea = senza bias, cerchi vuoti = con bias
    ax.plot(vo/N, po, ls="-", color=colors[i], label=rf"$\beta={beta:.4f}$")
    ax.plot(vb/N, pb, "o", ms=4, color=colors[i], markerfacecolor="none")

    # rapporto sui bin comuni con statistica sufficiente
    common, ib, io = np.intersect1d(vb, vo, return_indices=True)
    ok = (nb[ib] >= min_counts) & (no[io] >= min_counts)
    ratio = pb[ib][ok] / po[io][ok]
    # errore poissoniano naive (ignora l'autocorrelazione: sottostimato)
    err = ratio * np.sqrt(1/nb[ib][ok] + 1/no[io][ok])
    ax2.errorbar(common[ok]/N, ratio, yerr=err, fmt="o", ms=4, capsize=3,
                 color=colors[i], markerfacecolor="none", label=rf"$\beta={beta:.4f}$")

    eb, eo = Eb.mean()/N, Eo.mean()/N
    Cb, Co = beta**2*Eb.var()/N, beta**2*Eo.var()/N
    print(f"{beta:6.4f} | {eb:9.4f} {eo:9.4f} {eb-eo:+9.4f} | {Cb:7.3f} {Co:7.3f} {Cb/Co:6.3f}")

ax2.axhline(1, ls="--", c="black")

# legenda: colori = beta, stile = simulazione
handles, labels = ax.get_legend_handles_labels()
handles += [Line2D([], [], ls="-", c="gray"),
            Line2D([], [], ls="none", marker="o", c="gray", markerfacecolor="none")]
labels += ["senza bias", "con bias"]

ax.set_xlabel(r"$\epsilon$")
ax.set_ylabel(r"$P(\epsilon)$")
ax.set_title(rf"$L={L}$")
ax.legend(handles, labels, loc="best")

ax2.set_xlabel(r"$\epsilon$")
ax2.set_ylabel(r"$P_{\mathrm{bias}}(\epsilon)/P(\epsilon)$")
ax2.set_title(rf"$L={L}$")
ax2.legend(loc="best")

fig.tight_layout()
fig2.tight_layout()

fig.savefig(f"results/figure/{sim_bias}_vs_{sim_ok}_L{L}_hist.pdf", bbox_inches="tight")
fig2.savefig(f"results/figure/{sim_bias}_vs_{sim_ok}_L{L}_ratio.pdf", bbox_inches="tight")

plt.show()
