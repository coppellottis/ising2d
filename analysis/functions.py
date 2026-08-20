import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


rng = np.random.default_rng()

# Binder's cumulant
def binder_cumulant(x) :
    return 1-(x**4).mean()/(3*((x**2).mean())**2)

# block bootstrap error
# use this for secondary variables such as U4 Binder's cumulant
# returns error for different block dimension k
def blckbstr_error(x, f, k = 20):
    k = max(1, int(k))

    N = x.shape[0]
    n = N // k

    # pre-computation of block
    blocks = np.array([x[i*k:(i+1)*k] for i in range(n)])

    R = 1000
    means = np.zeros(R)

    for r in range(R):
        indices = rng.integers(0, n, size=n)
        means[r] = f(blocks[indices].reshape(-1))

    error = means.std(ddof=1)

    return error

# original definition of the integrated autocorrelation time
def tau_int(x) :
    tau = 0.5
    N = x.shape[0]

    mean = x.mean()
    var = x.var()

    for k in range(1,N//2) :
        d = var*(N-k)
        rho_k = 0
        for i in range(N-k):
            rho_k += (x[i]-mean)*(x[i+k]-mean)

        tau += 1/d*rho_k

        if(k>10*tau) :
            return tau
        
    print("Warning: Window did not converge; returned None.")
    return None

# (faster) computation of the int. autocorr. time through fast fourier transform
def tau_int_fft(x, c = 5, min_window = 4) :
    
    x = np.asarray(x)
    N = len(x)

    x = x - np.mean(x)

    nfft = 2**int(np.ceil(np.log2(2*N)))

    f = np.fft.fft(x, n=nfft)
    power = f * np.conjugate(f)

    acov = np.fft.ifft(power).real[:N]
    acov /= N # keeps only the first 0,... N-1 elements (on nnft)
    # biased convention used; acov/= N works better than un-biased acov/= N-lag... (biased+more stable for large k)

    rho = acov / acov[0]

    taus = np.concatenate(([0.5], 0.5+np.cumsum(rho[1:])))
    idx = np.arange(0, len(taus)) # lag k

    valid = (idx >= min_window) & (idx < N // 2) & (idx > c * taus)

    if not np.any(valid):
        print("Warning: Window did not converge; returned None.")
        return None
    else :
        return max(taus[np.argmax(valid)], 0.5)

def error(x) :
    x = np.asarray(x)
    N = len(x)

    tau = tau_int_fft(x)
    if tau is None:
        return np.nan, tau
    else:
        error = np.sqrt(np.var(x)*2*tau/N)
        return error, tau

# tau value + block jackknife error
def get_tau(x, tau_func, n_blocks=20):
    x = np.asarray(x)
    N = len(x)

    tau = tau_func(x)

    block_size = N // n_blocks
    x = x[:block_size * n_blocks]   # discard the tail
    blocks = np.array_split(x, n_blocks)

    tau_k = np.empty(n_blocks)
    for k in range(n_blocks):
        reduced = np.concatenate([blocks[i] for i in range(n_blocks) if i != k])
        tau_k[k] = tau_func(reduced)

    tau_mean = tau_k.mean()
    tau_err = np.sqrt((n_blocks - 1) / n_blocks * np.sum((tau_k - tau_mean) ** 2))
    return tau, tau_err

# output format
def format_error(value, error, sig=1):
    """
    Format as value(error), e.g.
    2.1734 ± 0.0512 -> 2.17(5)
    2.1734 ± 0.0051 -> 2.173(5)
    """
    if error == 0:
        return f"{value}"

    exponent = int(np.floor(np.log10(abs(error))))
    decimals = max(0, -exponent + (sig - 1))

    value_r = round(value, decimals)
    error_r = round(error, decimals)

    # errore espresso come intero nelle ultime cifre
    error_digits = int(round(error_r * 10**decimals))

    return f"{value_r:.{decimals}f}({error_digits})"