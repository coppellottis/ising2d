import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Same purpose as your loop-over-all-beta version of hist.py -- show how
# P(m) and P(E) evolve across the whole beta scan for one L -- but as a
# 2D density map (beta on the y-axis, m or E on the x-axis, color =
# probability density) instead of 20-30 overlaid step histograms.
#
# Overlaying that many outlines on one axes is inherently hard to read:
# every curve competes for the same space, near-identical colors from a
# continuous colormap are hard to tell apart in a legend with 20+ entries,
# and curves that cross each other hide one another. A heatmap sidesteps
# all three problems by giving every beta its own row -- nothing overlaps
# by construction, no legend is needed, and the same "peak shifts and
# narrows/widens near beta_c" story from the overlaid version is even more
# visible here as a continuous shape (a ridge that narrows and bends
# through the critical region).
#
# NOTE (bugfix, same as elsewhere in analysis/): results/figure/ is
# gitignored and doesn't exist on a fresh clone.
os.makedirs("results/figure", exist_ok=True)

sim_name = input("Simulation name: ")
alg = input("Algorithm (metropolis/wolff): ")
L = int(input("Select the system size L for which you want to generate the histograms: "))

filename = f"data/{sim_name}/metadata.csv"
metadata = pd.read_csv(filename)
row = metadata.loc[metadata["L"] == L].iloc[0]

beta_i = row["beta_i"]
beta_f = row["beta_f"]
n_beta = int(row["n_beta"])
n_measures = row["n_measures"]

# style (same as the rest of analysis/, for visual consistency)
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

    "legend.fontsize": 16,
})

beta_c = 0.5 * np.log(1 + np.sqrt(2))
betas = np.linspace(beta_i, beta_f, n_beta) if n_beta > 1 else np.array([beta_i])

# Number of bins along the observable axis (m or E). This is independent
# of n_beta (which sets the resolution along the other axis, one row per
# simulated beta) -- bump it up for a smoother-looking map.
N_BINS = 100

# First pass: load every beta's data and figure out ONE shared set of bin
# edges per observable. This is the key difference from the overlaid-step
# version: there, each histogram could silently use its own auto-ranged
# bins because only the *shape* mattered visually. Here every beta becomes
# one ROW of a 2D array, so all rows must be histogrammed on the exact
# same bins, or the columns wouldn't correspond to the same m/E value
# across rows.
m_data, E_data = [], []
for beta in betas:
    file_name = f"{sim_name}/{alg}_L{L}_beta{beta:.4f}.csv"
    data = pd.read_csv("data/" + file_name)
    m_data.append(data["m"].to_numpy())
    E_data.append(data["E_per_site"].to_numpy())

m_all = np.concatenate(m_data)
E_all = np.concatenate(E_data)
m_bins = np.linspace(m_all.min(), m_all.max(), N_BINS + 1)
E_bins = np.linspace(E_all.min(), E_all.max(), N_BINS + 1)


def build_density_grid(data_list, bins):
    # Z[i, :] = normalized histogram (density=True, same convention as
    # your ax.hist(..., density=True)) of beta index i's data on the
    # shared bins -- one row per beta.
    Z = np.zeros((len(betas), len(bins) - 1))
    for i, x in enumerate(data_list):
        Z[i], _ = np.histogram(x, bins=bins, density=True)
    return Z


def plot_density_map(Z, bins, xlabel, out_path):
    fig, ax = plt.subplots(figsize=(12, 8))

    # pcolormesh needs its X/Y coordinate arrays to be consistent with Z's
    # shape: either both given as cell EDGES (one more element than Z in
    # that dimension) or both as cell CENTERS (same length as Z).
    # "bins" from np.histogram are edges (N_BINS+1 of them); "betas" are
    # centers (n_beta of them, one per simulated point) -- mixing the two
    # is what shading="auto" was rejecting. Using bin centers for the
    # observable axis lets both axes be centers, matching Z's shape
    # directly with shading="nearest".
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    pcm = ax.pcolormesh(bin_centers, betas, Z, shading="nearest", cmap="Blues")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(r"$P_\beta$")

    if betas.min() <= beta_c <= betas.max():
        ax.axhline(beta_c, color="black", ls="--", lw=1.8, label=r"$\beta_c$ (Onsager)")
        ax.legend(loc="upper right")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\beta$")
    ax.text(0.03, 0.97, rf"$L={L}$", transform=ax.transAxes, ha="left", va="top",
            fontsize=18, color="white",
            bbox=dict(facecolor="white", alpha=0.35, edgecolor="none", pad=3))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Salvato: {out_path}")


Zm = build_density_grid(m_data, m_bins)
ZE = build_density_grid(E_data, E_bins)

plot_density_map(Zm, m_bins, r"$m$", f"results/figure/{sim_name}_L{L}_m_distr_all.pdf")
plot_density_map(ZE, E_bins, r"$E$", f"results/figure/{sim_name}_L{L}_E_distr_all.pdf")

plt.show()
