import cvxpy as cp

# Definicja zmiennych
x_LekI = cp.Variable(nonneg=True)
x_LekII = cp.Variable(nonneg=True)
x_SurI = cp.Variable(nonneg=True)
x_SurII = cp.Variable(nonneg=True)

# Koszty i przychody
f_costs = 100.00 * x_SurI + 199.90 * x_SurII + 700.00 * x_LekI + 800.00 * x_LekII
f_income = 6500.00 * x_LekI + 7100.00 * x_LekII

# Funkcja celu
objective = cp.Minimize(f_costs - f_income)

constraints = [
    # 1. Bilans czynnika aktywnego A
    0.01 * x_SurI + 0.02 * x_SurII - 0.50 * x_LekI - 0.60 * x_LekII >= 0,
    # 2. Ograniczenia magazynowe
    x_SurI + x_SurII <= 1000,
    # 3. Zasoby ludzkie
    90 * x_LekI + 100 * x_LekII <= 2000,
    # 4. Zasoby sprzętowe
    40 * x_LekI + 50 * x_LekII <= 800,
    # 5. Budżet
    f_costs <= 100000,
    # 6. Ograniczenia zakresu
    x_LekI >= 0,
    x_LekII >= 0,
    x_SurI >= 0,
    x_SurII >= 0
]

prob = cp.Problem(objective, constraints)
prob.solve()

print(f"Status: {prob.status}")
print(f"Lek I: {x_LekI.value:.3f} tys. opakowań")
print(f"Lek II: {x_LekII.value:.3f} tys. opakowań")
print(f"Surowiec I: {x_SurI.value:.3f} kg")
print(f"Surowiec II: {x_SurII.value:.3f} kg")