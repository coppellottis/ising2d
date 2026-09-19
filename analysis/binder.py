import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import brentq, curve_fit
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from functions import format_error

### I've assumed FSS for beta_x to be beta_x(L,sL)=beta_c+A(s)L^(-omega-nu), with omega = 2, nu = 21

def find_crossing(args) :
    sim_name, L1, L2, position = args

    df1 = pd.read_csv(f"results/{sim_name}/L{L1}.csv").sort_values("beta")
    df2 = pd.read_csv(f"results/{sim_name}/L{L2}.csv").sort_values("beta")

    # Doesn't depend on the bootstrap resample -- computing it once instead
    # of on every one of the 1000 iterations below.
    lo = max(df1["beta"].min(), df2["beta"].min())
    hi = min(df1["beta"].max(), df2["beta"].max())

    beta_cross = []
    n_failed = 0

    # position > 0: one fixed terminal row per concurrent worker, same
    # scheme as analyze.py (row 0 is the overall bar in the main process).
    for i in tqdm(range(0,10000), desc=f"L={L1}-{L2} bootstrap", position=position, leave=False, dynamic_ncols=True):
        U1_star = df1["U"] + np.random.normal(0, df1["err_U"])
        U2_star = df2["U"] + np.random.normal(0, df2["err_U"])

        f1 = interp1d(df1["beta"], U1_star, kind="cubic")
        f2 = interp1d(df2["beta"], U2_star, kind="cubic")

        # NOTE (bugfix/robustness): brentq requires f(lo) and f(hi) to have
        # opposite signs (it's a bisection method -- it needs the root
        # bracketed to guarantee one exists in [lo, hi]). A noisy bootstrap
        # resample can occasionally fail this even when the *mean* U1/U2
        # curves genuinely cross inside [lo, hi] -- most likely near the
        # edges of the beta window, or when U1 and U2 are close together
        # relative to err_U. A single such resample used to raise all the
        # way up and kill this whole job (and, once it hit fut.result() in
        # __main__, silently discarded every other L-pair's results too).
        # Skip just that resample instead and keep going.
        try:
            beta_cross.append(brentq(lambda x: f1(x)-f2(x), lo, hi))
        except ValueError:
            n_failed += 1

    if n_failed > 0:
        print(f"Warning: L={L1}-{L2}: {n_failed}/10000 bootstrap resamples did not bracket "
              f"a crossing in beta in [{lo:.4f}, {hi:.4f}] and were skipped.")

    if len(beta_cross) < 2:
        # Not a transient noise issue anymore -- the U(L1) and U(L2) curves
        # essentially don't cross in the simulated beta range. Raising here
        # (instead of computing std on 0-1 points) gives a specific,
        # actionable message instead of a bare "f(a) and f(b) must have
        # different signs" with no indication of which L pair or why.
        raise RuntimeError(
            f"L={L1}-{L2}: only {len(beta_cross)}/10000 bootstrap resamples found a Binder "
            f"crossing in beta in [{lo:.4f}, {hi:.4f}]. This usually means beta_i/beta_f for "
            f"these two L don't actually straddle the crossing (beta_c ~ 0.4407 for the 2D "
            f"Ising model), or that err_U is too large (too few measurements/too short a "
            f"chain) to resolve it. Check results/{sim_name}/L{L1}.csv and L{L2}.csv's "
            f"'beta'/'U'/'err_U' columns."
        )

    return L1, np.mean(beta_cross), np.std(beta_cross, ddof=1)

if __name__ == "__main__" :

    # NOTE (bugfix): results/figure/ is gitignored and doesn't exist on a
    # fresh clone; fig.savefig() below used to fail with FileNotFoundError
    # the first time this script ran.
    os.makedirs("results/figure", exist_ok=True)

    sim_name = input("Simulation: ").strip()
    alg = input("Algorithm (metropolis/wolff): ").strip()
    filename = f"data/{sim_name}/metadata.csv"

    metadata = pd.read_csv(filename)
    Ls = sorted(metadata["L"])

    n_processes = 4
    jobs = [(sim_name, L, 2*L) for L in Ls if 2*L in Ls]
    jobs = [(*job, 1 + i % n_processes) for i, job in enumerate(jobs)]

    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = {executor.submit(find_crossing, job): job for job in jobs}
        crossings = []
        with tqdm(total=len(jobs), desc="Crossing completati", position=0, dynamic_ncols=True) as outer:
            for fut in as_completed(futures):
                _sim_name, L1, L2, _position = futures[fut]
                try:
                    crossings.append(fut.result())
                except Exception as e:
                    # NOTE (bugfix/robustness): one L-pair whose Binder
                    # curves genuinely don't cross in range (see
                    # find_crossing) used to raise here and discard every
                    # other pair's results too -- a single bad pair meant
                    # zero output for the whole run. Report it and keep
                    # whatever pairs did succeed.
                    print(f"Warning: L={L1}-{L2} crossing failed and was skipped: {e}")
                outer.update(1)

    if len(crossings) < 2:
        raise RuntimeError(
            f"Only {len(crossings)}/{len(jobs)} L-pair crossing(s) succeeded -- need at least "
            f"2 for the beta_c(L) fit below. See the warnings above for why each pair failed."
        )

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

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.errorbar(np.log10(x**(-3)), y, yerr=yerr, fmt='o', markersize=5, capsize=3, label='Data', color='black')

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

    ax.plot(np.log10(xfit), yfit, label=fr'Fit: $\beta_c={formatted_bc}$')
    ax.fill_between(np.log10(xfit), yfit - sigma_fit, yfit + sigma_fit, alpha=0.2)

    fig.legend()
    fig.savefig(f"results/figure/{sim_name}_beta_c.pdf")

    plt.show()