import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# 1. Wczytanie danych
data = pd.read_csv('Modul1\\data01.csv', header=None)
x_val = data.iloc[:, 0].values
y_val = data.iloc[:, 1].values

# --- METODA LP (Norma L1) ---
a_lp = cp.Variable()
b_lp = cp.Variable()
obj_lp = cp.Minimize(cp.norm1(a_lp * x_val + b_lp - y_val))
cp.Problem(obj_lp).solve()

# --- METODA LS (Norma L2) ---
a_ls = cp.Variable()
b_ls = cp.Variable()
obj_ls = cp.Minimize(cp.sum_squares(a_ls * x_val + b_ls - y_val))
cp.Problem(obj_ls).solve()

# 2. Wyświetlenie wyników
print(f"LP: a = {a_lp.value:.4f}, b = {b_lp.value:.4f}")
print(f"LS: a = {a_ls.value:.4f}, b = {b_ls.value:.4f}") 

# 3. Wykres
plt.scatter(x_val, y_val, color='red', s=10, label='Dane')
plt.plot(x_val, a_lp.value * x_val + b_lp.value, 'blue', label='LP')
plt.plot(x_val, a_ls.value * x_val + b_ls.value, 'black', label='LS')
plt.legend()
plt.show()