import numpy as np
import cvxpy as cp
import scipy.io
import matplotlib.pyplot as plt
import os

# Dynamiczna ścieżka do pliku
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'isoPerimData.mat')

# Wczytanie danych
data = scipy.io.loadmat(file_path)
a, L, N = data['a'].item(), data['L'].item(), data['N'].item()
y_fixed_all = data['y_fixed'].flatten()
# Korekta indeksowania Matlab -> Python
fixed_idx = data['F'].flatten() - 1

h = a / N
y = cp.Variable(N + 1)

# Cel: Maksymalizacja pola
objective = cp.Maximize(h * cp.sum(y))

constraints = [
    y[0] == 0, y[N] == 0,
    y[fixed_idx] == y_fixed_all[fixed_idx],
    # Brak ograniczenia na krzywiznę (warunek 16c usunięty)
    cp.sum([cp.norm(cp.vstack([h, y[i+1] - y[i]])) for i in range(N)]) <= L
]

prob = cp.Problem(objective, constraints)
prob.solve()

print(f"Maksymalne pole bez ograniczenia C (1c): A = {prob.value:.4f}") # Odp: 0.6016

# Wykres krzywej optymalnej i punktów stałych
plt.figure(figsize=(10, 6))

# Rysowanie krzywej i punktów stałych
plt.plot(np.linspace(0, a, N + 1) / a, y.value, 'b-', linewidth=2, label='Krzywa optymalna')
plt.scatter(np.linspace(0, a, N + 1)[fixed_idx] / a, y_fixed_all[fixed_idx], color='red', zorder=5, label='Punkty stałe F')

plt.ylim(-0.25, 1.25)
plt.xlim(0, 1)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.grid(True, linestyle=':', alpha=0.6)
plt.xlabel('x / a')
plt.ylabel('y(x)')
plt.title(f"Zadanie Izoperymetryczne - Wynik: A = {prob.value:.4f}")
plt.legend()
plt.show()