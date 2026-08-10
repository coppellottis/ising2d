import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/metropolis_L32_beta0.4200.csv")

rng = np.random.default_rng();

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

# original definition of integrated autocorrelation time
def tau_int(x) :
    tau = 0
    N = x.shape[0]

    mean = x.mean()
    var = x.var()

    for k in range(1,N//2) :
        d = var*(N-k)
        rho_k = 0
        for i in range(N-k):
            rho_k += (x[i]-mean)*(x[i+k]-mean)

        tau += 1/d*rho_k
        print(f"tau_int (k={k}): {tau}")

        if(k>5*tau) :
            return tau
    return tau

# (faster) computation of int. autocorr. time through fast fourier transform
def tau_int_fft(x) :
    tau = 0

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
            print(f"tau:{tau}")
            return tau

    print(f"No tau")

def error(x) :
    x = np.asarray(x)
    N = len(x)

    tau = tau_int_fft(x)
    if tau is None:
        return 0
    else:
        error = np.sqrt(np.var(x)*(1+2*tau)/N)
        return error