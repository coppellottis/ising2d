import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq, curve_fit
from concurrent.futures import ProcessPoolExecutor
from functions import format_error

### I've assumed FSS for beta_x to be beta_x(L,sL)=beta_c+A(s)L^(-omega-nu), with omega = 2, nu = 21

def find_crossing(args) :
    sim_name, L1, L2 = args

    df1 = pd.read_csv(f"results/{sim_name}/L{L1}.csv").sort_values("beta")
    df2 = pd.read_csv(f"results/{sim_name}/L{L2}.csv").sort_values("beta")

    beta_cross = []

    for i in range(0,1000): 
        U1_star = df1["U"] + np.random.normal(0, df1["err_U"])
        U2_star = df2["U"] + np.random.normal(0, df2["err_U"])

        f1 = interp1d(df1["beta"], U1_star, kind="cubic")
        f2 = interp1d(df2["beta"], U2_star, kind="cubic")

        lo = max(df1["beta"].min(), df2["beta"].min())
        hi = min(df1["beta"].max(), df2["beta"].max())

        beta_cross.append(brentq(lambda x: f1(x)-f2(x), lo, hi))

    return L1, np.mean(beta_cross), np.std(beta_cross, ddof=1)

if __name__ == "__main__" :

    sim_name = input("Simulation: ").strip()
    alg = input("Algorithm (metropolis/wolff): ").strip()
    filename = f"data/{sim_name}/metadata.csv"

    metadata = pd.read_csv(filename)
    Ls = sorted(metadata["L"])

    jobs = [(sim_name, L, 2*L) for L in Ls if 2*L in Ls]

    n_processes = 4
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        crossings = list(executor.map(find_crossing, jobs))

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

    x, y, yerr = np.array(crossings).T # traspose
    x = x[1:]
    y = y[1:]
    yerr = yerr[1:]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.errorbar(x**(-3), y, yerr=yerr, fmt='o', markersize=5, capsize=3, label='Data', color='black')

    ax.set_xlabel(r"$L^{-3}$")
    ax.set_ylabel(r"$\beta_\times$")

    # fit

    def linear(x, bc, A):
        return bc + A*x

    popt, pcov = curve_fit(linear, x**(-3), y, sigma=yerr, absolute_sigma=True)

    bc, A = popt
    bc_err, A_err = np.sqrt(np.diag(pcov)) ## see doc

    formatted_bc = format_error(bc, bc_err, 1)

    xfit = np.linspace((x**(-3)).min(), (x**(-3)).max(), 200)
    yfit = linear(xfit,bc,A)

    var_fit = (pcov[0, 0] + xfit**2*pcov[1, 1] + 2*(xfit)*pcov[0, 1])
    sigma_fit = np.sqrt(var_fit)

    y2 = linear(x,bc,A)
    chi2_value = np.sum(((y - y2) / yerr)**2)
    dof = len(y)-2
    chi2_red = chi2_value / dof

    ax.plot(xfit, yfit, label=fr'Fit: $\beta_c={formatted_bc}$')
    ax.fill_between(xfit, yfit - sigma_fit, yfit + sigma_fit, alpha=0.2)

    fig.legend()
    fig.savefig(f"results/figure/{sim_name}_beta_c.pdf")

    plt.show()