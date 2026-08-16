import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

simulation = "metropolis" # "wolff"
beta_c = 0.5*np.log(1+np.sqrt(2))

data = pd.read_csv(f"results/{simulation}_tau.csv")

idx = (data["beta"]-beta_c).abs().idxmin() # finds the beta closest to beta_c

tau = data.loc[idx][1:]
fig, ax = plt.subplots()

ax.plot(np.linspace(3,7,5),tau)
ax.set_yscale("log")

plt.show()