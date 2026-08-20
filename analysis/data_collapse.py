import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sim_name = input("Simulation name: ") 
alg = input("Algorithm (metropolis/wolff): ")

filename = f"data/{sim_name}/metadata.csv"

metadata = pd.read_csv(filename)
