import pandas as pd 
import numpy as np
from functions import error

sim_name = input("Nome simulazione: ").strip()
filename = f"data/{sim_name}/metadata.csv"


beta_i = 0.4200
beta_f = 0.4600
delta_beta = beta_f-beta_i

n_beta = 25 # number of beta for each size L
n_size = 5 # L=8,16,32,64,128

betas = np.zeros(n_beta)
abs_m = np.zeros((n_beta,n_size))
e_abs_m = np.zeros((n_beta,n_size))
m = np.zeros((n_beta,n_size))
e_m = np.zeros((n_beta,n_size))
energy = np.zeros((n_beta, n_size))

df1 = pd.DataFrame() # m
df2 = pd.DataFrame() # abs_m
df3 = pd.DataFrame() # energy

for i in range(0,n_beta) :

    beta = beta_i+i*(delta_beta/n_beta)

    for j in range(0,n_size) :

        L = 8*pow(2,j) 

        file_name = f"{simulation}_L{L}_beta{beta:.4f}.csv"
        data = pd.read_csv("data/" + file_name)

        df2.loc[i, f"L{L}"] = data["m"].abs().mean()
        df2.loc[i, f"e_L{L}"] = error(data["m"].abs())
        df1.loc[i, f"L{L}"] = data["m"].mean()
        df1.loc[i, f"e_L{L}"] = error(data["m"])
        df3.loc[i, f"L{L}"] = data["E_per_site"].mean()
        betas[i] = beta

df1.insert(0,"beta",betas)
df2.insert(0,"beta",betas)
df3.insert(0,"beta",betas)

df1.to_csv(f"results/{simulation}_m.csv",index=False)
df2.to_csv(f"results/{simulation}_abs_m.csv",index=False)
df3.to_csv(f"results/{simulation}_energy.csv",index=False)