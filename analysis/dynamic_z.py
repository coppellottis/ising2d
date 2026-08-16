import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sim_name = input("Simulation name: ") 
alg = input("Algorithm (metropolis/wolff): ")

filename = f"data/{sim_name}/metadata.csv"
metadata = pd.read_csv(filename)

beta_c = 0.5*np.log(1+np.sqrt(2))
tau = np.zeros(metadata.shape[0])

for i in range(0,metadata.shape[0]) :
    L = metadata["L"][i]

    data = pd.read_csv(f"results/{sim_name}/L{L}_tau.csv")

    idx = (data["beta"]-beta_c).abs().idxmin() # finds the beta closest to beta_c

    tau[i] = data["tau_abs_m"][idx]

# style
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 18,

    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,

    "axes.linewidth": 1.0,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": True,
    "ytick.right": True,

    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,

    "legend.fontsize": 18,
})


fig, ax = plt.subplots(figsize=(12, 8))
x = np.log(metadata["L"])
y = np.log(tau)

ax.scatter(x,y)

ax.set_xlabel(r"$\log L$")
ax.set_ylabel(r"$\log \tau_{|m|}$")

p = np.polyfit(x, y, 1)

z = p[0]
logA = p[1]

print("z =", z)

plt.show()