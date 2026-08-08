import pandas as pd 

data = pd.read_csv("results/results_beta_0.4000.csv")

mean_E = data["E_per_site"].mean()
mean_m = data["m"].mean()

print(f"<E>/N = {mean_E:.6f}")
print(f"<m> = {mean_m:.6f}")