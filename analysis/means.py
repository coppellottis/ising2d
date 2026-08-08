import pandas as pd 

data = pd.read_csv("results/results_beta_0.4500.csv")

mean_E = data["E_per_site"].mean()
mean_m = data["m"].mean()
mean_abs_m = data["m"].abs().mean()

print(f"<E>/N = {mean_E:.6f}")
print(f"<m> = {mean_m:.6f}")
print(f"<|m|> = {mean_abs_m:.6f}")
