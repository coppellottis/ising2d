import os
import pandas as pd
import numpy as np
import multiprocessing
from scipy.special import logsumexp
from scipy.optimize import minimize_scalar
from scipy.optimize import curve_fit
from tqdm import tqdm
from functools import partial

# Import personalizzati
from functions import format_error, tau_int_fft

## Parametri bootstrap
N_IT = 200
BLOCK_SIZE_MULT = 100 # c*tau

def block_bootstrap_sampler(data, block_size):
    length = len(data)
    block_size = min(block_size, length) # Salvaguardia
    n_blocks = length // block_size

    blocks = [
        data[i:i + block_size]
        for i in range(0, n_blocks * block_size, block_size)
    ]

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

# =============================================================================
# FUNZIONE WORKER PER IL MULTIPROCESSING
# =============================================================================
def run_single_bootstrap(it_index, L, N, betas, original_data, tau_dict, bins, E_max):
    # Inizializza un seed random diverso per ogni processo/iterazione
    # altrimenti tutti i core generano lo stesso campione di bootstrap!
    np.random.seed((os.getpid() * int((it_index+1) * 10000)) % 123456789)

    I_all = []
    N_eff = []

    # Genera istogrammi
    for beta in betas:
        data_arr = original_data[beta]
        tau_E = tau_dict[beta]

        block_size = max(1, int(np.ceil(BLOCK_SIZE_MULT * tau_E)))
        data_boot = block_bootstrap_sampler(data_arr, block_size)

        energies = (L**2) * data_boot
        I_beta = np.bincount(np.round((energies - (-E_max))/4).astype(int), minlength=len(bins))[:len(bins)]
        I_beta = I_beta.astype(float)

        I_beta /= (2 * tau_E)
        N_eff.append(len(data_boot) / (2 * tau_E))

        I_all.append(I_beta)

    I_all = np.asarray(I_all)
    N_eff = np.asarray(N_eff)

    n_beta, n_bins = I_all.shape
    I_sum = I_all.sum(axis=0)
    mask = I_sum > 0
    
    logI_sum = np.full(n_bins, -np.inf)
    logI_sum[mask] = np.log(I_sum[mask])

    logN_eff = np.log(N_eff)
    f = np.zeros(n_beta)
    betaE = np.outer(betas, bins)

    n_iter = 0
    max_iter = 20000
    tol = 1e-10
    
    # WHAM Loop (senza barra spaziale interna per non intasare il terminale)
    for n_iter in range(1, max_iter + 1):
        log_denom = logsumexp((logN_eff + f)[:, None] - betaE, axis=0)
        log_gE = np.full(n_bins, -np.inf)
        log_gE[mask] = logI_sum[mask] - log_denom[mask]

        f_new = -logsumexp(log_gE[None, :] - betaE, axis=1)
        f_new = f_new - f_new[0]
        
        delta = np.max(np.abs(f_new - f))
        f = f_new
        
        if delta < tol:
            break

    # Ricerca del massimo
    res = minimize_scalar(
        lambda b: -specific_heat(b, bins, log_gE, N), 
        bounds=(betas[0], betas[-1]), 
        method='bounded'
    )
    
    return res.x, -res.fun


# =============================================================================
# MAIN SCRIPT
# =============================================================================
def main():
    sim_name = input("Simulation name: ") 
    alg = input("Algorithm (metropolis/wolff): ")

    filename = f"data/{sim_name}/metadata.csv"
    metadata = pd.read_csv(filename)

    os.makedirs("results/figure", exist_ok=True)
    
    beta_max_mean = []
    beta_max_err = []

    # Determiniamo i core da usare (lasciamone uno libero per il sistema operativo)
    num_cores = 4
    print(f"Avvio elaborazione in parallelo utilizzando {num_cores} cores.")

    for i in range(metadata.shape[0]):
        L = metadata["L"][i]
        N = L**2

        beta_i = metadata["beta_i"][i]
        beta_f = metadata["beta_f"][i]
        n_beta = metadata["n_beta"][i]
        n_measures = metadata["n_measures"][i]

        betas = np.linspace(beta_i, beta_f, n_beta)
        E_max = 2 * N
        bins = np.arange(-E_max, E_max + 4, 4)

        # ---------------------------------------------------------
        # FASE 1: I/O e calcolo tau una tantum per questa taglia L
        # ---------------------------------------------------------
        original_data = {}
        tau_dict = {}
        
        print(f"\nCaricamento dati per L={L}...")
        for beta in tqdm(betas, desc="Lettura CSV & FFT", leave=False):
            data = pd.read_csv(f"data/{sim_name}/{alg}_L{L}_beta{beta:.4f}.csv")
            data_arr = data["E_per_site"].to_numpy()
            tau_E, _ = tau_int_fft(data_arr)
            
            original_data[beta] = data_arr
            tau_dict[beta] = max(tau_E, 0.5)

        # ---------------------------------------------------------
        # FASE 2: Bootstrap Parallelo
        # ---------------------------------------------------------
        # Uso 'partial' per "fissare" gli argomenti costanti della funzione
        worker_func = partial(
            run_single_bootstrap,
            L=L, N=N, betas=betas, 
            original_data=original_data, tau_dict=tau_dict,
            bins=bins, E_max=E_max
        )

        beta_max_L = []
        CV_max_L = []

        # Avvio il pool di processi
        with multiprocessing.Pool(processes=num_cores) as pool:
            # imap permette a tqdm di aggiornarsi man mano che i processi finiscono
            iterator = pool.imap_unordered(worker_func, range(N_IT))
            
            for result_b, result_cv in tqdm(iterator, total=N_IT, desc=f"WHAM Bootstrap L={L}"):
                beta_max_L.append(result_b)
                CV_max_L.append(result_cv)

        beta_max_mean.append(np.mean(beta_max_L))
        beta_max_err.append(np.std(beta_max_L, ddof=1))

    # =========================================================================
    # FIT E OUTPUT FINALE - ANALISI DI STABILITA'
    # =========================================================================
    beta_max_mean_arr = np.asarray(beta_max_mean)
    beta_max_err_arr = np.asarray(beta_max_err)
    L_array = metadata["L"].to_numpy()

    def beta_max_fit(L, beta_c, x0, nu):
        return beta_c + x0 * L**(-1/nu)

    print("\n" + "="*70)
    print(" ANALISI DI STABILITÀ DEL FIT (Rimozione progressiva L piccoli)")
    print("="*70)

    # Per fittare 3 parametri (beta_c, x0, nu) servono come minimo assoluto 3 punti.
    # Quindi possiamo rimuovere taglie finché ci restano almeno 3 elementi.
    max_removals = len(L_array) - 3

    for i in range(max_removals + 1):
        # Slicing degli array: da 'i' in poi (ignora i primi 'i' elementi)
        curr_L = L_array[i:]
        curr_beta = beta_max_mean_arr[i:]
        curr_err = beta_max_err_arr[i:]
        
        L_min = curr_L[0]
        
        try:
            popt, pcov = curve_fit(
                beta_max_fit, 
                curr_L, 
                curr_beta, 
                p0=[0.4407, 1, 1], 
                sigma=curr_err, 
                absolute_sigma=True,
                maxfev=10000 # Aumentiamo le iterazioni per fit con pochi punti
            )

            beta_c, x0, nu = popt
            err_beta_c, err_x0, err_nu = np.sqrt(np.diag(pcov))

            # chi^2 ridotto: residui pesati per l'errore bootstrap, dof = punti - 3 parametri
            residuals = curr_beta - beta_max_fit(curr_L, *popt)
            chi2 = np.sum((residuals / curr_err) ** 2)
            dof = len(curr_L) - len(popt)
            chi2_red = chi2 / dof if dof > 0 else np.nan

            sign = "+" if x0 >= 0 else ""
            
            # Output formattato
            print(f"Fit con L >= {L_min:<4} (usati {len(curr_L)} punti, dof={dof}):")
            print(fr"  $\beta_{{pc}}(L) = {format_error(beta_c, err_beta_c)} {sign} {format_error(x0, err_x0)} \cdot L^{{-1/{format_error(nu, err_nu)}}}$")
            if dof > 0:
                print(f"  chi^2/dof = {chi2_red:.3f}")
            else:
                print("  chi^2/dof = n/d (dof=0, il fit passa esattamente per i punti)")
            print("-" * 70)
            
        except Exception as e:
            print(f"Fit con L >= {L_min:<4} fallito. Errore: {e}")
            print("-" * 70)

if __name__ == '__main__':
    main()
