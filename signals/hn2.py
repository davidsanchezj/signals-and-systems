import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dark_background")

# Espacio que ocupa la señal X
pi_x = -3
pf_x = 2
Lx = pf_x - pi_x + 1
vector_x = np.arange(pi_x, pf_x + 1, 1)

# Espacio que ocupa la señal Y
pi_h = 0
pf_h = 6
Lh = pf_h - pi_h + 1
vector_h = np.arange(pi_h, pf_h + 1, 1)

# Definir las funciones que representan las señales
X = np.ones_like(vector_x)
H = np.power(7 / 8, vector_h) * np.ones_like(vector_h)

# Graficar las señales
plt.figure(1)
plt.stem(vector_x, X)
plt.title("x[n]")
plt.tight_layout()
plt.show()

plt.figure(2)
plt.stem(vector_h, H)
plt.title("h[n]")
plt.tight_layout()
plt.show()

# Eje de las señales
Eje_n = np.arange(pi_x - Lh, pf_x + Lh + 1, 1)

# Señales
x_n = np.concatenate((np.zeros(Lh, dtype=int), X, np.zeros(Lh, dtype=int)), axis=0)
h_n = np.concatenate((np.zeros(Lh, dtype=int), H, np.zeros(Lx, dtype=int)), axis=0)

# -----------------------------------------------------------------------
y_n = np.zeros(Lx + Lh - 1, dtype=float)
y = []
for n in range(0, Lx + Lh - 2 + 1):
    # Graficar x[k]
    x_k = x_n
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].stem(Eje_n, x_k)
    ax[0, 0].set_title("x[k]")
    ax[0, 0].set_xlabel("k")

    # Graficar h[n-k]
    h_n_menos_k = np.concatenate(
        (np.zeros(n + 1, dtype=int), H[::-1], np.zeros(Lx + Lh - (n + 1), dtype=int)),
        axis=0,
    )
    ax[1, 0].stem(Eje_n, h_n_menos_k)
    ax[1, 0].set_title("h[n-k]")
    ax[1, 0].set_xlabel("k")

    # Graficar v[k]
    v_k = x_k * h_n_menos_k
    ax[0, 1].stem(Eje_n, v_k)
    ax[0, 1].set_title("v[k]")
    ax[0, 1].set_xlabel("k")

    # Graficar y[n]
    y_n[n] = float(np.sum(v_k))
    n_y = np.arange(pi_x + pi_h, pf_x + pf_h + 1, 1)
    ax[1, 1].stem(n_y, y_n)
    ax[1, 1].set_title("y[n]")
    ax[1, 1].set_xlabel("k")
    if n < Lx + Lh - 2:
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
