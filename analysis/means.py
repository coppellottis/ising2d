import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


beta_i = 0.4200
beta_f = 0.4600
delta_beta = beta_f-beta_i

n_beta = 25 # number of beta for each size L
n_size = 5 # L=8,16,32,64,128

betas = np.zeros((n_beta,n_size))
abs_m = np.zeros((n_beta,n_size))
m = np.zeros((n_beta,n_size))

for i in range(0,n_beta) :

    beta = beta_i+i*(delta_beta/n_beta)

    for j in range(0,n_size) :

        L = 8*pow(2,j) 

        file_name = f"metropolis_L{L}_beta{beta:.4f}.csv"
        data = pd.read_csv("data/" + file_name)

        abs_m[i,j] = data["m"].abs().mean()
        m[i,j] = data["m"].mean()
        betas[i,j] = beta


df = pd.DataFrame({
    "beta" : betas[:,0],
    "L8" : abs_m[:,0],
    "L16" : abs_m[:,1],
    "L32" : abs_m[:,2],
    "L64" : abs_m[:,3],
    "L164" : abs_m[:,4]
})

df.to_csv("results/metropolis_abs_m.csv", index=False);

df2 = pd.DataFrame({
    "beta" : betas[:,0],
    "L8" : m[:,0],
    "L16" : m[:,1],
    "L32" : m[:,2],
    "L64" : m[:,3],
    "L164" : m[:,4]
})

df2.to_csv("results/metropolis_m.csv", index=False);