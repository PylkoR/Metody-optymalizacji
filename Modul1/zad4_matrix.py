import numpy as np
import cvxpy as cp
import pandas as pd

# 1. Wczytanie danych
data = pd.read_csv('Modul1\\data01.csv')
x_val = data.iloc[:, 0].values
y_val = data.iloc[:, 1].values
N = len(x_val)

# 1. Budowa macierzy Phi i wektora y
Phi = np.column_stack((x_val, np.ones(N))) # Macierz z x i jedynkami 
y = y_val.reshape(-1, 1)

# 2. Zmienne: theta (a, b) oraz tau (wektor błędów)
theta = cp.Variable((2, 1))
tau = cp.Variable((N, 1))

# 3. Funkcja celu: suma tau [cite: 504, 505]
objective = cp.Minimize(cp.sum(tau))

# 4. Ograniczenia macierzowe z równania (43) w PDF
constraints = [
    Phi @ theta - y <= tau,    # [cite: 530]
    -Phi @ theta + y <= tau    # [cite: 531]
]

prob = cp.Problem(objective, constraints)
prob.solve()

print(f"a = {theta.value[0][0]:.4f}")
print(f"b = {theta.value[1][0]:.4f}")