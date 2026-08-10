import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/metropolis_L32_beta0.4200.csv")

rng = np.random.default_rng();

# use this for secondary variables such as U4 Binder's cumulant
# returns error for different block dimension k
def block_bootstrap(sample):
    error = np.zeros(15) # k=1,2,4,8...,2**15 
    N = sample.shape[0]

    for j in range(15):
        k = 2**j
        n = N // k

        # pre-computation of block means
        block_means = np.array([
            sample[i*k:(i+1)*k].mean()
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
def tau_int(sample) :
    tau = 0
    N = sample.shape[0]

    mean = sample.mean()
    var = sample.var()

    for k in range(1,N//2) :
        d = var*(N-k)
        rho_k = 0
        for i in range(N-k):
            rho_k += (sample[i]-mean)*(sample[i+k]-mean)

        tau += 1/d*rho_k
        print(f"tau_int (k={k}): {tau}")

        if(k>5*tau) :
            return tau
    return tau

# (faster) computation of int. autocorr. time through fast fourier transform
def tau_int_fft(sample) :
    return


tau = tau_int(data["m"])
print(f"tau_int: {tau}")