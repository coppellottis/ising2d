import os
import pandas as pd
import numpy as np
from scipy.special import logsumexp
from functions import format_error
from scipy.optimize import minimize_scalar
from scipy.optimize import curve_fit
from tqdm import tqdm
from functions import tau_int_fft

sim_name = input("Simulation name: ") 
alg = input("Algorithm (metropolis/wolff): ")

filename = f"data/{sim_name}/metadata.csv"

metadata = pd.read_csv(filename)

# output
os.makedirs("results/figure", exist_ok=True)
results = {}

## Parametri bootstrap
N_IT = 20
BLOCK_SIZE_MULT = 100 # c*tau

def block_bootstrap_sampler(data, block_size):
    length = len(data)

    # Solo blocchi completi
    n_blocks = length // block_size

    blocks = [
        data[i:i + block_size]
        for i in range(0, n_blocks * block_size, block_size)
    ]

    # Numero di blocchi necessario per superare length
    n_sample_blocks = int(np.ceil(length / block_size))

    sampled_blocks = np.random.choice(
        len(blocks),
        size=n_sample_blocks,
        replace=True
    )

    bootstrap_sample = np.concatenate(
        [blocks[i] for i in sampled_blocks]
    )

    return bootstrap_sample[:length]

def specific_heat(beta, E_bins, log_gE, N):

    log_w = log_gE - beta * E_bins
    f = logsumexp(log_w)

    w = np.exp(log_w - f) 
 
    e_mean = np.sum(w * E_bins)
    e2_mean = np.sum(w * E_bins**2)
    var_E = e2_mean - e_mean ** 2
    cv = beta**2 * var_E / N

    return cv

beta_max = {}
CV_max = {}

beta_max_mean = []
beta_max_err = []

for i in tqdm(range(0,metadata.shape[0]), desc="WHAM per L", position=0, dynamic_ncols=True) :
    L = metadata["L"][i]
    N = L**2

    beta_i = metadata["beta_i"][i]
    beta_f = metadata["beta_f"][i]
    n_beta = metadata["n_beta"][i]
    n_measures = metadata["n_measures"][i]

    beta_max[L] = []
    CV_max[L] = []

    betas = np.linspace(beta_i, beta_f, n_beta)

    for j in tqdm(range(0, N_IT), desc="Iterazione block-bootstrap", position=1, dynamic_ncols=True):

        I_all = []
        tau = []
        N_eff = []

        for beta in tqdm(betas, desc=f"L={L:<4} istogrammi", position=2, leave=False, dynamic_ncols=True) :
            data = pd.read_csv(f"data/{sim_name}/{alg}_L{L}_beta{beta:.4f}.csv")
            tau_E, _ = tau_int_fft(data["E_per_site"])

            data_boot = block_bootstrap_sampler(data["E_per_site"].to_numpy(), int(np.ceil(BLOCK_SIZE_MULT * tau_E)))

            energies = (L**2)*data_boot
            E_max = 2*N
            bins = np.arange(-E_max, E_max+4, 4) # arange per 0, 1000 si ferma a 999 (motivo del +4)...
            I_beta = np.bincount(np.round((energies-(-E_max))/4).astype(int), minlength=len(bins))[:len(bins)]
            I_beta = I_beta.astype(float)

            # correzioni correlazione
            I_beta /= 2*tau_E
            N_eff.append(n_measures/(2*tau_E))

            I_all.append(I_beta)

        # converto la lista in array numpy
        I_all = np.asarray(I_all)
        N_eff = np.asarray(N_eff)

        # soluzione WHAM
        # n_beta = R, # simulazione;
        n_beta, n_bins = I_all.shape
        I_sum = I_all.sum(axis=0)
        mask = I_sum > 0
        # anzichè calcolare log(0), metto -inf dove I_sum = 0
        logI_sum = np.full(n_bins, -np.inf)
        logI_sum[mask] = np.log(I_sum[mask])

        # anzichè tenere N_j^eff * e^{...}, lo converto in un log, così e^{log N_eff+...}
        logN_eff = np.log(N_eff)

        # initial guess: array degli f_k = 0
        f = np.zeros(n_beta)
        # calcolala matrice beta_j E, con tutti i valori di E tra -E_max e +E_max
        betaE = np.outer(betas, bins)

        # logN_eff, f hanno stesso dimensione n_betax1. betaE ha dimensione n_betaxn_bins

        n_iter = 0
        max_iter = 20000
        tol = 1e-10
        # max_iter is a safety cap, not the expected iteration count (this
        # usually converges in a few hundred iterations), so the bar's
        # percentage is mostly meaningless -- what matters is the live delta
        # in the postfix, showing it's actually shrinking towards tol.
        pbar = tqdm(range(1, max_iter+1), desc="WHAM: iterazioni", position=2, leave=False, dynamic_ncols=True)
        for n_iter in pbar :
            log_denom = logsumexp((logN_eff+f)[:,None]-betaE, axis=0)
            log_gE = np.full(n_bins, -np.inf)
            log_gE[mask] = logI_sum[mask] - log_denom[mask]
            # log_gE è un array di dimensione 1xn_bins

            f_new = -logsumexp(log_gE[None,:]-betaE, axis=1)
            # sommo lungo le righe e ottengo un vettore lungo n_beta = R
        
            #gauge
            f_new = f_new - f_new[0]
            delta = np.max(np.abs(f_new-f))
            f = f_new
        
            if n_iter % 20 == 0 or delta < tol :
                pbar.set_postfix(delta=f"{delta:.2e}", tol=f"{tol:.0e}")
            if delta < tol :
                break
        
        pbar.close()

        res = minimize_scalar(lambda beta: -specific_heat(beta, bins, log_gE, N), bounds=(beta_i, beta_f), method='bounded')
        beta_max[L].append(res.x)
        CV_max[L].append(-res.fun)
        
    beta_max[L] = np.asarray(beta_max[L])
    beta_max_mean.append(np.mean(beta_max[L]))
    beta_max_err.append(np.std(beta_max[L], ddof=1))

beta_max_mean = np.asarray(beta_max_mean)
beta_max_err = np.asarray(beta_max_err)

# faccio fit beta_c, nu
def beta_max_fit(L, beta_c, x0, nu):
    return beta_c + x0 * L**(-1/nu)

# Fit
popt, pcov = curve_fit(beta_max_fit,metadata["L"],beta_max_mean,p0=[0.4407, -0.2, 1.0],sigma=beta_max_err,absolute_sigma=True)

beta_c, x0, nu = popt
err_beta_c, err_x0, err_nu = np.sqrt(np.diag(pcov))

print(fr"$\beta_{{pc}}(L) = {format_error(beta_c, err_beta_c)}{format_error(x0, err_x0)}*L^{{1/{format_error(nu, err_nu)}}}$")
