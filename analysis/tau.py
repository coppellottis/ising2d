import numpy as np
import pandas as pd
from functions import tau_int_fft

simulation = "metropolis" # "wolff"

beta_i = 0.4200
beta_f = 0.4600
delta_beta = beta_f-beta_i

n_beta = 25 # number of beta for each size L
n_size = 5 # L=8,16,32,64,128

betas = np.zeros(n_beta)
m = np.zeros((n_beta,n_size))

df = pd.DataFrame()

for i in range(0,n_beta) :

    beta = beta_i+i*(delta_beta/n_beta)
    betas[i] = beta

    for j in range(0,n_size) :
        L = 8*pow(2,j) 

        file_name = f"{simulation}_L{L}_beta{beta:.4f}.csv"
        data = pd.read_csv("data/" + file_name)

        df.loc[i, f"L{L}"] = tau_int_fft(data["m"])

df.insert(0,"beta",betas)

df.to_csv(f"results/{simulation}_tau.csv",index=False)