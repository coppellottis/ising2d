import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


beta0 = 0.3500
betas = np.zeros(30)
abs_m = np.zeros(30)

for i in range(0,30) :
    beta = beta0+(i*0.2/30.0);
    file_name = f"L30_beta{beta:.4f}.csv"
    data = pd.read_csv("results/" + file_name)

    abs_m[i] = data["m"].abs().mean()
    betas[i] = beta

plt.scatter(betas, abs_m, marker="s")
plt.xlabel(r"$\beta$")
plt.ylabel(r"$\langle |m|\rangle$")
plt.show()