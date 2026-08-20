import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk, ellipe

def energy_theory(beta):
    k = 2 * np.sinh(2 * beta) / np.cosh(2 * beta)**2
    m = k**2

    return -1 / np.tanh(2 * beta) * (
        1
        + (2 / np.pi)
        * (2 * np.tanh(2 * beta)**2 - 1)
        * ellipk(m)
    )

def specific_heat_theory(beta):
    t = np.tanh(2 * beta)

    k = 2 * np.sinh(2 * beta) / np.cosh(2 * beta)**2
    m = k**2

    K = ellipk(m)
    E = ellipe(m)

    C = (2 / np.pi) * (beta / np.tanh(2 * beta))**2 * (
        2 * (K - E)
        - (1 - t**2) * (
            np.pi / 2
            + (2 * t**2 - 1) * K
        )
    )

    return C

sim_name = input("Simulation name: ") 
alg = input("Algorithm (metropolis/wolff): ")

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
fig2, ax2 = plt.subplots(figsize=(12, 8))
fig3, ax3 = plt.subplots(figsize=(12, 8))
fig4, ax4 = plt.subplots(figsize=(12, 8))
fig5, ax5 = plt.subplots(figsize=(12, 8))
fig6, ax6 = plt.subplots(figsize=(12, 8))
beta_min = 1
beta_max = 0

for i in range(0,metadata.shape[0]) :
    L = metadata["L"][i]
    beta_i = metadata["beta_i"][i]
    beta_f = metadata["beta_f"][i]
    n_beta = metadata["n_beta"][i]
    n_measures = metadata["n_measures"][i]

    data = pd.read_csv(f"results/{sim_name}/L{L}.csv")

    ax.errorbar(data["beta"], data["abs_m"], yerr=data["err_abs_m"], fmt="o", capsize=3, markerfacecolor='none', label=rf"$L={L}$")
    ax2.errorbar(data["beta"], data["e"], yerr=data["err_e"], fmt="o", capsize=3, markerfacecolor='none', label=rf"$L={L}$")
    ax3.errorbar(data["beta"], data["C"], yerr=data["err_C"], fmt="o", capsize=3, markerfacecolor='none', label=rf"$L={L}$")
    ax4.errorbar(data["beta"], data["chi"], yerr=data["err_chi"], fmt="o", capsize=3, markerfacecolor='none', label=rf"$L={L}$")
    ax5.errorbar(data["beta"], data["chi1"]/(1e2), yerr=data["err_chi1"]/(1e2), fmt="o", capsize=3, markerfacecolor='none', label=rf"$L={L}$")
    ax6.errorbar(data["beta"], data["U"], yerr=data["err_U"], fmt="o", capsize=3, markerfacecolor='none', label=rf"$L={L}$")


    beta_min = data["beta"].min() if data["beta"].min() < beta_min else beta_min
    beta_max = data["beta"].max() if data["beta"].max() > beta_max else beta_max

### theory vs experimental
beta_c = 0.5 * np.log(1 + np.sqrt(2))
x = np.linspace(beta_c,beta_max,1000)
y = (1 - np.sinh(2*x)**(-4))**(1/8)
ax.plot(x,y,ls="--",c="black")
ax.plot([beta_min, beta_c], [0, 0], ls="--",c="black",label=r"$L\to\infty$ (Onsager)")

x2 = np.linspace(beta_min,beta_max,1000)
y2 = energy_theory(x2)
ax2.plot(x2,y2,ls="--",c="black",label=r"$L\to\infty$ (Onsager)")

#y3 = specific_heat_theory(x2)
#ax3.plot(x2,y3,ls="--",c="black",label=r"$L\to\infty$ (Onsager)")

##### 
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$\langle |m| \rangle$")

ax2.set_xlabel(r"$\beta$")
ax2.set_ylabel(r"$\epsilon$")

ax3.set_xlabel(r"$\beta$")
ax3.set_ylabel(r"$C$")

ax4.set_xlabel(r"$\beta$")
ax4.set_ylabel(r"$\chi$")

ax5.set_xlabel(r"$\beta$")
ax5.set_ylabel(r"$\chi^\prime/10^2$")

ax6.set_xlabel(r"$\beta$")
ax6.set_ylabel(r"$U$")

ax.legend(loc="best")
ax2.legend(loc="best")
ax3.legend(loc="best")
ax4.legend(loc="best")
ax5.legend(loc="best")
ax6.legend(loc="best")

fig.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()
fig6.tight_layout()

fig.savefig(f"results/figure/{sim_name}_abs_m.pdf", bbox_inches="tight")
fig2.savefig(f"results/figure/{sim_name}_e.pdf", bbox_inches="tight")
fig3.savefig(f"results/figure/{sim_name}_C.pdf", bbox_inches="tight")
fig4.savefig(f"results/figure/{sim_name}_chi.pdf", bbox_inches="tight")
fig5.savefig(f"results/figure/{sim_name}_chi1.pdf", bbox_inches="tight")
fig6.savefig(f"results/figure/{sim_name}_U.pdf", bbox_inches="tight")

plt.show()