import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng()

# use this for secondary variables such as U4 Binder's cumulant
# returns error for different block dimension k
def block_bootstrap(x):
    error = np.zeros(15) # k=1,2,4,8...,2**15 
    N = x.shape[0]

    for j in range(15):
        k = 2**j
        n = N // k

        # pre-computation of block means
        block_means = np.array([
            x[i*k:(i+1)*k].mean()
            for i in range(n)
        ])

        R = 1000
        means = np.zeros(R)

        for r in range(R):
            indices = rng.integers(0, n, size=n)
            means[r] = block_means[indices].mean()

        error[j] = means.std(ddof=1)
        print(f"sigma_k = {error[j]} per k = {k}")

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

        if(k>5*tau) :
            return tau
        
    print("Warning: Window did not converge; using tau at k=N//2 (likely underestimated).")
    return tau

# (faster) computation of the int. autocorr. time through fast fourier transform
def tau_int_fft(x) :
    tau = 0.5

    x = np.asarray(x)
    N = len(x)

    x = x - np.mean(x)

    nfft = 2**int(np.ceil(np.log2(2*N)))

    f = np.fft.fft(x, n=nfft)
    power = f * np.conjugate(f)

    acov = np.fft.ifft(power).real[:N]
    acov /= np.arange(N, 0, -1) # keeps only the first 0,... N-1 elements (on nnft)

    rho = acov / acov[0]

    for k in range(1,N//2):
        tau += rho[k]
        if (k>5*tau) :
            return tau

    print("Warning: Window did not converge; using tau at k=N//2 (likely underestimated).")
    return tau

def error(x) :
    x = np.asarray(x)
    N = len(x)

    tau = tau_int_fft(x)
    if tau is None:
        return 0
    else:
        error = np.sqrt(np.var(x)*2*tau/N)
        return error

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