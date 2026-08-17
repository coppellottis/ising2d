import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sim_name = input("Simulation name: ") 
alg = input("Algorithm (metropolis/wolff): ")

filename = f"data/{sim_name}/metadata.csv"
metadata = pd.read_csv(filename)

beta_c = 0.5*np.log(1+np.sqrt(2))
tau = np.zeros(metadata.shape[0])
err_tau = np.zeros(metadata.shape[0])

for i in range(0,metadata.shape[0]) :
    L = metadata["L"][i]

    data = pd.read_csv(f"results/{sim_name}/L{L}_tau.csv")

    idx = (data["beta"]-beta_c).abs().idxmin() # finds the beta closest to beta_c

    tau[i] = data["tau_abs_m"][idx]
    err_tau[i] = data["err_tau_abs_m"][idx]

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
x = np.log(metadata["L"])
y = np.log(tau)
yerr = (err_tau/tau)

ax.errorbar(
    x, y,
    yerr=yerr,
    fmt='o',
    markersize=5,
    capsize=3,
    label='Data',
    color='black'
)

ax.set_xlabel(r"$\log L$")
ax.set_ylabel(r"$\log \tau_{int}^{|m|}$")

def linear(x, a, z):
    return a + z*x

popt, pcov = curve_fit(
    linear,
    x,
    y,
    sigma=yerr,
    absolute_sigma=True
)

a, z = popt
a_err, z_err = np.sqrt(np.diag(pcov)) ## see doc

xfit = np.linspace(x.min(), x.max(), 200)
yfit = linear(xfit,a,z)

var_fit = (
    pcov[0, 0]
    + xfit**2 * pcov[1, 1]
    + 2*xfit * pcov[0, 1]
)
sigma_fit = np.sqrt(var_fit)

y2 = linear(x,a,z)
chi2_value = np.sum(((y - y2) / yerr)**2)
dof = len(y)-2
chi2_red = chi2_value / dof

ax.plot(
    xfit,
    yfit,
    label=fr'Fit: $z^\prime={z:.2f}\pm{z_err:.2f}$, $\chi_r^2={chi2_red:.2f}$ (dof=$3$)'
)

ax.fill_between(
    xfit,
    yfit - sigma_fit,
    yfit + sigma_fit,
    alpha=0.2
)

## subplot
axins = inset_axes(
    ax,
    width="38%",
    height="38%",
    loc="lower right",
    borderpad=2
)

axins.errorbar(
    metadata["L"], tau,
    yerr=err_tau,
    fmt='o',
    markersize=5,
    capsize=3,
    color='black'
)

# eventualmente tick più piccoli
axins.tick_params(labelsize=12)
axins.set_xlabel('L',size=16)
axins.set_ylabel(r"$\tau_{int}^{|m|}$",size=16)

ax.legend()
fig.tight_layout()

fig.savefig(
    f"results/figure/{sim_name}_dynamic_z.pdf",
    bbox_inches="tight"
)

plt.show()

