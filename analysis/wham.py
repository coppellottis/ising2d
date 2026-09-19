import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
from scipy.optimize import curve_fit
from tqdm import tqdm
from functions import build_histogram

# NOTE (bugfix): results/figure/ is gitignored and doesn't exist on a fresh
# clone; fig.savefig() below used to fail with FileNotFoundError the first
# time this script ran.
os.makedirs("results/figure", exist_ok=True)

def solve_multiple_histogram(betas, n_eff, E_bins, H, tol=1e-10, max_iter=20000, position=1) :
    # K = # of betas, M = # of bins
    K, M = H.shape

    logn = np.log(n_eff)
    H_tot = H.sum(axis=0)
    mask = H_tot > 0
    logH_tot = np.full(M, -np.inf)
    logH_tot[mask] = np.log(H_tot[mask])

    # initial guess
    f = np.zeros(K)
    betaE = np.outer(betas, E_bins)

    n_iter = 0
    # max_iter is a safety cap, not the expected iteration count (this
    # usually converges in a few hundred iterations), so the bar's
    # percentage is mostly meaningless -- what matters is the live delta
    # in the postfix, showing it's actually shrinking towards tol.
    pbar = tqdm(range(1, max_iter+1), desc="WHAM: iterazioni", position=position, leave=False, dynamic_ncols=True)
    for n_iter in pbar :
        exponent = (logn+f)[:,None]-betaE
        log_denom = logsumexp(exponent, axis=0)
        log_gE = np.full(M, -np.inf)
        log_gE[mask] = logH_tot[mask] - log_denom[mask]

        exponent2 = log_gE[None,:] - betaE
        f_new = -logsumexp(exponent2, axis=1)

        #gauge
        f_new = f_new - f_new[0]
        delta = np.max(np.abs(f_new-f))
        f = f_new

        if n_iter % 20 == 0 or delta < tol :
            pbar.set_postfix(delta=f"{delta:.2e}", tol=f"{tol:.0e}")

        if delta < tol :
            break
    pbar.close()

    return log_gE, f, n_iter

def specific_heat(beta, E_bins, log_gE, N):

    log_w = log_gE - beta * E_bins
    log_Z = logsumexp(log_w)
    w = np.exp(log_w - log_Z) 
 
    e_mean = np.sum(w * E_bins)
    e2_mean = np.sum(w * E_bins**2)
    var_E = e2_mean - e_mean ** 2
    cv = beta**2 * var_E / N

    return cv

def chi(beta, M, E_bins, log_gE, N):

    return

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

sim_name = input("Simulation name: ") 
alg = input("Algorithm (metropolis/wolff): ")

filename = f"data/{sim_name}/metadata.csv"

metadata = pd.read_csv(filename)

beta_max = []
cv_max = []

for i in tqdm(range(0,metadata.shape[0]), desc="WHAM per L", position=0, dynamic_ncols=True) :
    L = metadata["L"][i]
    N = L**2

    beta_i = metadata["beta_i"][i]
    beta_f = metadata["beta_f"][i]
    n_beta = metadata["n_beta"][i]
    n_measures = metadata["n_measures"][i]

    H_all = []

    betas = np.linspace(beta_i, beta_f, n_beta)

    for beta in tqdm(betas, desc=f"L={L:<4} istogrammi", position=1, leave=False, dynamic_ncols=True) :
        data = pd.read_csv(f"data/{sim_name}/{alg}_L{L}_beta{beta:.4f}.csv")
        energies = (L**2)*data["E_per_site"]
        e_max = 2*N
        E_bins, H = build_histogram(energies, -e_max, e_max)
        H_all.append(H)

    H_all = np.asarray(H_all)

    data3 = pd.read_csv(f"results/{sim_name}/L{L}_tau.csv")
    tau_E = data3["tau_E"]
    err_tau_E = data3["err_tau_E"]

    # NOTE (robustness): tau_E/err_tau_E can now contain NaN for a beta
    # where tau_int_fft didn't converge (get_tau() reports this and moves
    # on instead of crashing tau.py -- see functions.py). UnivariateSpline
    # can't fit through NaNs, so this used to blow up here with an opaque
    # scipy error for the *next* script in the pipeline, far from the
    # actual cause. Interpolate across the gap (it's a smoothing spline
    # for n_eff anyway, not a physical result in itself) and say so.
    if tau_E.isna().any():
        n_missing = int(tau_E.isna().sum())
        print(f"Warning: L={L}: {n_missing}/{len(tau_E)} tau_E values did not converge "
              f"(see functions.py warnings above); interpolating across them for the WHAM n_eff spline.")
        tau_E = tau_E.interpolate(limit_direction="both")
        err_tau_E = err_tau_E.interpolate(limit_direction="both")

    spline = UnivariateSpline(betas, np.log(tau_E), w=1/(err_tau_E/tau_E), k=3)
    tau_E_smooth = np.exp(spline(betas))

    n_eff = np.asarray(n_measures / (2*tau_E_smooth), dtype=float)

    log_gE, f, n_iter =  solve_multiple_histogram(betas, np.mean(n_eff), E_bins, H_all)
    print(f"n_iter: {n_iter}")

    x = np.linspace(beta_i, beta_f, 1000)
    y = [specific_heat(beta, E_bins, log_gE, N) for beta in x]

    line = ax.plot(x,y, color="black")[0]

    data2 = pd.read_csv(f"results/{sim_name}/L{L}.csv")
    ax.errorbar(data2["beta"], data2["C"], yerr=data2["err_C"], fmt="o", capsize=3, markerfacecolor='none', label=rf"$L={L}$")

    result = minimize_scalar(lambda beta: -specific_heat(beta, E_bins, log_gE, N),bounds=(beta_i, beta_f),method='bounded')
    beta_max.append(result.x)
    cv_max.append(-result.fun)

line.set_label('WHAM')

beta_max = np.asarray(beta_max)
cv_max = np.asarray(cv_max)

L = np.asarray(metadata["L"])

def beta_max_fit(L, beta_c, x0, nu):
    return beta_c + x0 * L**(-1/nu)

# Fit
popt, pcov = curve_fit(beta_max_fit,L,beta_max,p0=[0.4407, 1.0, 1.0])

beta_c, x0, nu = popt
err_beta_c, err_x0, err_nu = np.sqrt(np.diag(pcov))

df = pd.DataFrame()
df.loc[0, "beta_c"] = beta_c
df.loc[0, "err_beta_c"] = err_beta_c
df.loc[0,"nu"] = nu 
df.loc[0,"err_nu"] = err_nu
df.loc[0,"x0"] = x0
df.loc[0,"err_x0"] = err_x0 

df.to_csv(f"results/{sim_name}/fit_cv_fss.csv", index=False)

fig.tight_layout()
fig.legend()

ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$C_V(\beta)/N$")

fig.savefig(f"results/figure/{sim_name}_wham.pdf", bbox_inches="tight")

plt.show()