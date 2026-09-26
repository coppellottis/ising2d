import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from tqdm import tqdm
from scipy.special import logsumexp
from scipy.optimize import curve_fit
from functions import format_error

def specific_heat(beta, E_bins, log_gE, N):

    log_w = log_gE - beta * E_bins
    f = logsumexp(log_w)

    w = np.exp(log_w - f)

    e_mean = np.sum(w * E_bins)
    e2_mean = np.sum(w * E_bins**2)
    var_E = e2_mean - e_mean ** 2
    cv = beta**2 * var_E / N

    return cv

# beta_pc(L) = beta_c + x0 * L^(-1/nu): forma standard della finite-size
# scaling per la posizione del massimo di una grandezza pseudo-critica
# (qui C_V). p0 = [0.4407, -0.2, 1.0] parte da beta_c teorico noto
# (0.5*ln(1+sqrt2)) e da un esponente nu=1 (valore Onsager per il 2D Ising).
def beta_max_fit(L, beta_c, x0, nu):
    return beta_c + x0 * np.asarray(L, dtype=float)**(-1.0/nu)

# Stampa il fit beta_pc(L)=beta_c+x0*L^-1/nu su un dato sottoinsieme di
# taglie, col chi^2 ridotto. NOTA: qui non abbiamo una stima dell'errore
# su beta_max(L) (beta_max viene da un argmax deterministico sulla curva
# WHAM, non da un resampling) -- curve_fit viene quindi chiamato senza
# sigma, e il residuo sum(residuals**2)/dof NON è un chi^2 calibrato
# (non ha le unità giuste, non tende a 1 per un buon fit): è solo un
# indicatore relativo, utile per confrontare i vari tagli L_min tra loro,
# non per un test di bontà del fit in senso stretto. Per un chi^2/dof
# vero serve err_beta_max(L) dal bootstrap (wham_bootstrap.py o lo script
# di bootstrap con multiprocessing).
def fit_beta_pc(L_fit, beta_fit, p0=(0.4407, -0.2, 1.0)):
    popt, pcov = curve_fit(beta_max_fit, L_fit, beta_fit, p0=p0, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))

    dof = len(L_fit) - len(popt)
    residuals = np.asarray(beta_fit) - beta_max_fit(L_fit, *popt)
    chi2 = np.sum(residuals**2)
    chi2_red = chi2 / dof if dof > 0 else np.nan

    return popt, perr, dof, chi2_red

# Come format_error, ma non esplode se l'errore non è finito: con dof=0
# (fit a 3 punti su 3 parametri, curva passa esattamente per i dati) o con
# una covarianza singolare, curve_fit restituisce pcov=inf/nan invece di
# sollevare un'eccezione (solo un OptimizeWarning) -- format_error fa
# np.log10(errore), che con inf/nan crasha con OverflowError. In quel
# caso non c'è un vero errore da riportare: si stampa solo la stima.
def fmt(value, err):
    if not np.isfinite(err):
        return f"{value:.4g}(n/d)"
    return format_error(value, err)

def print_fit_result(label, L_fit, popt, perr, dof, chi2_red):
    beta_c, x0, nu = popt
    err_beta_c, err_x0, err_nu = perr

    print(f"{label} (usati {len(L_fit)} punti, dof={dof}):")
    print(fr"  $\beta_{{pc}}(L) = {fmt(beta_c, err_beta_c)}  {fmt(x0, err_x0)} \cdot L^{{-1/{fmt(nu, err_nu)}}}$")
    if not np.all(np.isfinite(perr)):
        print("  (errore sui parametri non stimabile: covarianza singolare -- con dof=0 il fit passa esattamente per i punti)")
    if dof > 0:
        print(f"  chi^2/dof = {chi2_red:.3g}  (fit non pesato: vedi nota in fit_beta_pc)")
    else:
        print("  chi^2/dof = n/d (dof=0, il fit passa esattamente per i punti)")
    print("-" * 70)

def stability_analysis(L_all, beta_all, p0=(0.4407, -0.2, 1.0)):
    print()
    print("=" * 70)
    print(" ANALISI DI STABILITÀ DEL FIT (rimozione progressiva L piccoli)")
    print("=" * 70)

    L_all = np.asarray(L_all, dtype=float)
    beta_all = np.asarray(beta_all, dtype=float)

    for L_min in np.sort(np.unique(L_all)):
        mask = L_all >= L_min
        curr_L = L_all[mask]
        curr_beta = beta_all[mask]

        if len(curr_L) < 3:
            print(f"Fit con L >= {L_min:<4}: saltato (servono almeno 3 taglie, ne restano {len(curr_L)})")
            print("-" * 70)
            continue

        try:
            popt, perr, dof, chi2_red = fit_beta_pc(curr_L, curr_beta, p0=p0)
        except RuntimeError as e:
            print(f"Fit con L >= {L_min:<4}: non convergente ({e})")
            print("-" * 70)
            continue

        print_fit_result(f"Fit con L >= {L_min:<4}", curr_L, popt, perr, dof, chi2_red)

sim_name = input("Simulation name: ")
alg = input("Algorithm (metropolis/wolff): ")

filename = f"data/{sim_name}/metadata.csv"

metadata = pd.read_csv(filename)
wham_results = pd.read_csv(f"results/{sim_name}/wham_results_n.csv")

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
colors = plt.cm.Blues(np.linspace(.2,1,metadata.shape[0]))

beta_max = []
CV_max = []

LL = metadata["L"]

for i in tqdm(range(0,metadata.shape[0]), desc="Calcolo C_V per L", position=0, dynamic_ncols=True) :
    L = LL[i]
    beta_i = metadata["beta_i"][i]
    beta_f = metadata["beta_f"][i]

    N = L**2
    bins = wham_results["E"]
    log_gE = wham_results[f"{L}"]

    data = pd.read_csv(f"results/{sim_name}/L{L}.csv")
    ax.errorbar(data["beta"], data["C"], yerr=data["err_C"], fmt="o", capsize=3, color=colors[i], markerfacecolor='none', label=rf"$L={L}$")

    x = np.linspace(beta_i, beta_f, 1000)
    y = [specific_heat(beta, bins, log_gE, N) for beta in x]
    line = ax.plot(x,y, color=colors[i])[0]

    res = minimize_scalar(lambda beta: -specific_heat(beta, bins, log_gE, N), bounds=(beta_i, beta_f), method='bounded')
    beta_max.append(res.x)
    CV_max.append(-res.fun)

    ax.plot(beta_max, CV_max, marker="o", linestyle="None", color="black", ms=5)

beta_max = np.asarray(beta_max)
CV_max = np.asarray(CV_max)


# --- fit FSS: beta_pc(L) = beta_c + x0 * L^(-1/nu), tutte le taglie ---
print()
print("=" * 70)
print(" FIT FSS: beta_pc(L) = beta_c + x0 * L^(-1/nu)  (tutte le taglie)")
print("=" * 70)
popt_all, perr_all, dof_all, chi2_red_all = fit_beta_pc(np.asarray(LL, dtype=float), beta_max)
print_fit_result("Fit su tutte le taglie", LL, popt_all, perr_all, dof_all, chi2_red_all)

# --- test di consistenza: rimozione progressiva delle taglie piccole ---
stability_analysis(LL, beta_max)


# aggiusto i grafici
ax.set_xlabel(r"$\beta$")
ax.set_ylabel(r"$C_V$")

ax.legend(loc="best")
fig.tight_layout()

fig.savefig(f"results/figure/{sim_name}_C_wham.pdf", bbox_inches="tight")
plt.show()
