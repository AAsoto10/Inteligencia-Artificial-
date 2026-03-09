import numpy as np

np.random.seed(42)

# ===============================
# Generar 10 000 datos
# ===============================
n = 10000

incrementos = np.random.uniform(0.5, 1.5, n)
x = np.cumsum(incrementos)   # X creciente

# Modelo real
a_real = 20      # término independiente
b_real = 0.9     # pendiente (coeficiente de x)

ruido = np.random.normal(0, 50, n)
y = a_real + b_real * x + ruido

# ===============================
# Rangos para buscar coeficientes
# ===============================
rango_a = np.arange(-100, 100, 5)
rango_b = np.arange(-2, 2, 0.1)

print(rango_a)
print("#")
print(rango_b)