import numpy as np
import matplotlib.pyplot as plt

Delta = 0.01

first_signal = input("Selecciona la primera señal: ")
second_signal = input("Selecciona la segunda señal: ")

if first_signal == "a":
    tx = np.arange(-1.01, 1.01 + Delta, Delta)
    Ltx = len(tx)
    x_t = np.zeros(Ltx)
    x_ti = np.where(np.isclose(tx, -1))[0][0]
    x_tf = np.where(np.isclose(tx, 1))[0][0]
    x_t[x_ti : x_tf + 1] = 1

elif first_signal == "b":
    tx1 = np.arange(0, 1, Delta)
    tx2 = np.arange(1, 3, Delta)
    tx3 = np.arange(3, 4, Delta)
    tx4 = np.arange(4, 5 + Delta, Delta)

    x1_t = np.zeros_like(tx1)
    x2_t = np.ones_like(tx2)
    x3_t = 2 * np.ones_like(tx3)
    x4_t = np.zeros_like(tx4)

    x_t = np.concatenate((x1_t, x2_t, x3_t, x4_t))
    tx = np.concatenate((tx1, tx2, tx3, tx4))

elif first_signal == "c":
    tx1 = np.arange(-3, -2, Delta)
    tx2 = np.arange(-2, 0, Delta)
    tx3 = np.arange(0, 2, Delta)
    tx4 = np.arange(2, 3 + Delta, Delta)

    x1_t = np.zeros_like(tx1)
    x2_t = -1 * np.ones_like(tx2)
    x3_t = np.ones_like(tx3)
    x4_t = np.zeros_like(tx4)

    x_t = np.concatenate((x1_t, x2_t, x3_t, x4_t))
    tx = np.concatenate((tx1, tx2, tx3, tx4))


elif first_signal == "d":
    tx1 = np.arange(-2, -1, Delta)
    tx2 = np.arange(-1, 1, Delta)
    tx3 = np.arange(1, 2 + Delta, Delta)

    x1_t = np.zeros_like(tx1)
    x2_t = tx2
    x3_t = np.zeros_like(tx3)

    x_t = np.concatenate((x1_t, x2_t, x3_t))
    tx = np.concatenate((tx1, tx2, tx3))

elif first_signal == "e":
    tx1 = np.arange(-5, -4, Delta)
    tx2 = np.arange(-4, -2, Delta)
    tx3 = np.arange(-2, 0, Delta)
    tx4 = np.arange(0, 1, Delta)
    tx5 = np.arange(1, 2, Delta)  # Se le quito el +Delta

    x1_t = np.zeros_like(tx1)
    x2_t = 0.5 * tx2 + 2
    x3_t = np.ones_like(tx3)
    x4_t = -1 * tx4 + 1
    x5_t = np.zeros_like(tx5)

    x_t = np.concatenate((x1_t, x2_t, x3_t, x4_t, x5_t))
    tx = np.concatenate((tx1, tx2, tx3, tx4, tx5))

elif first_signal == "f":
    tx = np.arange(0, 1 + Delta, Delta)
    x_t = np.exp(-tx)

# Segunda Señal
if second_signal == "a":
    th = np.arange(-1.01, 1.01 + Delta, Delta)
    Lth = len(th)
    h_t = np.zeros(Lth)
    h_ti = np.where(np.isclose(th, -1))[0][0]
    h_tf = np.where(np.isclose(th, 1))[0][0]
    h_t[h_ti : h_tf + 1] = 1

elif second_signal == "b":
    th1 = np.arange(0, 1, Delta)
    th2 = np.arange(1, 3, Delta)
    th3 = np.arange(3, 4, Delta)
    th4 = np.arange(4, 5 + Delta, Delta)

    h1_t = np.zeros_like(th1)
    h2_t = np.ones_like(th2)
    h3_t = 2 * np.ones_like(th3)
    h4_t = np.zeros_like(th4)

    h_t = np.concatenate((h1_t, h2_t, h3_t, h4_t))
    th = np.concatenate((th1, th2, th3, th4))

elif second_signal == "c":
    th1 = np.arange(-3, -2, Delta)
    th2 = np.arange(-2, 0, Delta)
    th3 = np.arange(0, 2, Delta)
    th4 = np.arange(2, 3 + Delta, Delta)

    h1_t = np.zeros_like(th1)
    h2_t = -1 * np.ones_like(th2)
    h3_t = np.ones_like(th3)
    h4_t = np.zeros_like(th4)

    h_t = np.concatenate((h1_t, h2_t, h3_t, h4_t))
    th = np.concatenate((th1, th2, th3, th4))


elif second_signal == "d":
    th1 = np.arange(-2, -1, Delta)
    th2 = np.arange(-1, 1, Delta)
    th3 = np.arange(1, 2 + Delta, Delta)

    h1_t = np.zeros_like(th1)
    h2_t = th2
    h3_t = np.zeros_like(th3)

    h_t = np.concatenate((h1_t, h2_t, h3_t))
    th = np.concatenate((th1, th2, th3))

elif second_signal == "e":
    th1 = np.arange(-5, -4, Delta)
    th2 = np.arange(-4, -2, Delta)
    th3 = np.arange(-2, 0, Delta)
    th4 = np.arange(0, 1, Delta)
    th5 = np.arange(1, 2, Delta)  # Se le quito el +Delta

    h1_t = np.zeros_like(th1)
    h2_t = 0.5 * th2 + 2
    h3_t = np.ones_like(th3)
    h4_t = -1 * th4 + 1
    h5_t = np.zeros_like(th5)

    h_t = np.concatenate((h1_t, h2_t, h3_t, h4_t, h5_t))
    th = np.concatenate((th1, th2, th3, th4, th5))

elif second_signal == "f":
    th = np.arange(0, 1 + Delta, Delta)
    h_t = np.exp(-th)


# Gráficas funciones
plt.figure(1)

plt.subplot(2, 1, 1)
plt.plot(tx, x_t)
plt.title("Primera Señal")

plt.subplot(2, 1, 2)
plt.plot(th, h_t)
plt.title("Segunda Señal")
plt.show()

# Convolución usando Numpy

if (first_signal == "a" and second_signal == "f") or (
    first_signal == "f" and second_signal == "a"
):
    ty = np.arange(tx[0] + th[0], tx[-1] + th[-1], Delta)
    y_t = np.convolve(x_t, h_t, "full") * Delta
elif (first_signal == "a" and second_signal == "d") or (
    first_signal == "d" and second_signal == "a"
):
    ty = np.arange(tx[0] + th[0], tx[-1] + th[-1], Delta)
    y_t = np.convolve(x_t, h_t, "full") * Delta
elif (first_signal == "d" and second_signal == "f") or (
    first_signal == "f" and second_signal == "d"
):
    ty = np.arange(tx[0] + th[0], tx[-1] + th[-1], Delta)
    y_t = np.convolve(x_t, h_t, "full") * Delta
elif (first_signal == "e" and second_signal == "a") or (
    first_signal == "a" and second_signal == "e"
):
    ty = np.arange(tx[0] + th[0], tx[-1] + th[-1], Delta)
    y_t = np.convolve(x_t, h_t, "full") * Delta
elif (first_signal == "e" and second_signal == "d") or (
    first_signal == "d" and second_signal == "e"
):
    ty = np.arange(tx[0] + th[0], tx[-1] + th[-1], Delta)
    y_t = np.convolve(x_t, h_t, "full") * Delta
elif (first_signal == "a" and second_signal == "a") or (
    first_signal == "a" and second_signal == "a"
):
    ty = np.arange(tx[0] + th[0], tx[-1] + th[-1], Delta)
    y_t = np.convolve(x_t, h_t, "full") * Delta
elif (first_signal == "d" and second_signal == "d") or (
    first_signal == "d" and second_signal == "d"
):
    ty = np.arange(tx[0] + th[0], tx[-1] + th[-1], Delta)
    y_t = np.convolve(x_t, h_t, "full") * Delta
elif (first_signal == "e" and second_signal == "e") or (
    first_signal == "e" and second_signal == "e"
):
    ty = np.arange(tx[0] + th[0], tx[-1] + th[-1], Delta)
    y_t = np.convolve(x_t, h_t, "full") * Delta
else:
    ty = np.arange(tx[0] + th[0], tx[-1] + th[-1] + Delta, Delta)
    y_t = np.convolve(x_t, h_t, "full") * Delta

# Convolución Manual

if (first_signal == "a" and second_signal == "e") or (
    first_signal == "e" and second_signal == "a"
):
    ty1_manual = np.arange(-6, -5, Delta)
    ty2_manual = np.arange(-5, -3, Delta)
    ty3_manual = np.arange(-3, -1, Delta)
    ty4_manual = np.arange(-1, 0, Delta)
    ty5_manual = np.arange(0, 1, Delta)
    ty6_manual = np.arange(1, 2, Delta)
    ty7_manual = np.arange(2, 3 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = (ty2_manual**2) / 4 + 5 * ty2_manual / 2 + 25 / 4
    y3_t_manual = -((ty3_manual) ** 2) / 4 - ty3_manual / 2 + 7 / 4
    y4_t_manual = -(ty4_manual**2) / 2 - ty4_manual + 3 / 2
    y5_t_manual = -ty5_manual + 3 / 2
    y6_t_manual = (ty6_manual**2) / 2 - 2 * ty6_manual + 2
    y7_t_manual = np.zeros_like(ty7_manual)

    y_t_manual = np.concatenate(
        (
            y1_t_manual,
            y2_t_manual,
            y3_t_manual,
            y4_t_manual,
            y5_t_manual,
            y6_t_manual,
            y7_t_manual,
        )
    )
    ty_manual = np.concatenate(
        (
            ty1_manual,
            ty2_manual,
            ty3_manual,
            ty4_manual,
            ty5_manual,
            ty6_manual,
            ty7_manual,
        )
    )

elif (first_signal == "a" and second_signal == "c") or (
    first_signal == "c" and second_signal == "a"
):
    ty1_manual = np.arange(-4, -3, Delta)
    ty2_manual = np.arange(-3, -1, Delta)
    ty3_manual = np.arange(-1, 1, Delta)
    ty4_manual = np.arange(1, 3, Delta)
    ty5_manual = np.arange(3, 4 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = -ty2_manual - 3
    y3_t_manual = 2 * ty3_manual
    y4_t_manual = -ty4_manual + 3
    y5_t_manual = np.zeros_like(ty5_manual)

    y_t_manual = np.concatenate(
        (y1_t_manual, y2_t_manual, y3_t_manual, y4_t_manual, y5_t_manual)
    )
    ty_manual = np.concatenate(
        (ty1_manual, ty2_manual, ty3_manual, ty4_manual, ty5_manual)
    )

elif (first_signal == "a" and second_signal == "d") or (
    first_signal == "d" and second_signal == "a"
):
    ty1_manual = np.arange(-3, -2, Delta)
    ty2_manual = np.arange(-2, 0, Delta)
    ty3_manual = np.arange(0, 2, Delta)
    ty4_manual = np.arange(2, 3 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = (ty2_manual**2) / 2 + ty2_manual
    y3_t_manual = -(ty3_manual**2) / 2 + ty3_manual
    y4_t_manual = np.zeros_like(ty4_manual)

    y_t_manual = np.concatenate((y1_t_manual, y2_t_manual, y3_t_manual, y4_t_manual))
    ty_manual = np.concatenate((ty1_manual, ty2_manual, ty3_manual, ty4_manual))

elif (first_signal == "d" and second_signal == "f") or (
    first_signal == "f" and second_signal == "d"
):
    ty1_manual = np.arange(-2, -1, Delta)
    ty2_manual = np.arange(-1, 0, Delta)
    ty3_manual = np.arange(0, 1, Delta)
    ty4_manual = np.arange(1, 2, Delta)
    ty5_manual = np.arange(2, 3 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = ty2_manual - 1 + 2 * np.exp(-(ty2_manual + 1))
    y3_t_manual = ty3_manual - 1 - (ty3_manual - 2) * np.exp(-1)
    y4_t_manual = -np.exp(-1) * (ty4_manual - 2)
    y5_t_manual = np.zeros_like(ty5_manual)

    y_t_manual = np.concatenate(
        (y1_t_manual, y2_t_manual, y3_t_manual, y4_t_manual, y5_t_manual)
    )
    ty_manual = np.concatenate(
        (ty1_manual, ty2_manual, ty3_manual, ty4_manual, ty5_manual)
    )

elif (first_signal == "a" and second_signal == "f") or (
    first_signal == "f" and second_signal == "a"
):
    ty1_manual = np.arange(-2, -1, Delta)
    ty2_manual = np.arange(-1, 0, Delta)
    ty3_manual = np.arange(0, 1, Delta)
    ty4_manual = np.arange(1, 2, Delta)
    ty5_manual = np.arange(2, 3 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = 1 - np.exp(-(ty2_manual + 1))
    y3_t_manual = 1 - np.exp(-1) + 0 * ty3_manual
    y4_t_manual = np.exp(1 - ty4_manual) - np.exp(-1)
    y5_t_manual = np.zeros_like(ty5_manual)

    y_t_manual = np.concatenate(
        (y1_t_manual, y2_t_manual, y3_t_manual, y4_t_manual, y5_t_manual)
    )
    ty_manual = np.concatenate(
        (ty1_manual, ty2_manual, ty3_manual, ty4_manual, ty5_manual)
    )

elif (first_signal == "b" and second_signal == "c") or (
    first_signal == "c" and second_signal == "b"
):
    ty1_manual = np.arange(-3, -1, Delta)
    ty2_manual = np.arange(-1, 1, Delta)
    ty3_manual = np.arange(1, 2, Delta)
    ty4_manual = np.arange(2, 3, Delta)
    ty5_manual = np.arange(3, 4, Delta)
    ty6_manual = np.arange(4, 5, Delta)
    ty7_manual = np.arange(5, 6, Delta)
    ty8_manual = np.arange(6, 8 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = -ty2_manual - 1
    y3_t_manual = -2 + 0 * ty3_manual
    y4_t_manual = 2 * ty4_manual - 6
    y5_t_manual = 3 * ty5_manual - 9
    y6_t_manual = -ty6_manual + 7
    y7_t_manual = -2 * ty7_manual + 12
    y8_t_manual = np.zeros_like(ty8_manual)

    y_t_manual = np.concatenate(
        (
            y1_t_manual,
            y2_t_manual,
            y3_t_manual,
            y4_t_manual,
            y5_t_manual,
            y6_t_manual,
            y7_t_manual,
            y8_t_manual,
        )
    )
    ty_manual = np.concatenate(
        (
            ty1_manual,
            ty2_manual,
            ty3_manual,
            ty4_manual,
            ty5_manual,
            ty6_manual,
            ty7_manual,
            ty8_manual,
        )
    )

elif (first_signal == "b" and second_signal == "d") or (
    first_signal == "d" and second_signal == "b"
):
    ty1_manual = np.arange(-2, 0, Delta)
    ty2_manual = np.arange(0, 2, Delta)
    ty3_manual = np.arange(2, 3, Delta)
    ty4_manual = np.arange(3, 4, Delta)
    ty5_manual = np.arange(4, 5, Delta)
    ty6_manual = np.arange(5, 7 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = (ty2_manual**2) / 2 - ty2_manual
    y3_t_manual = (ty3_manual**2) / 2 - 3 * ty3_manual + 4
    y4_t_manual = -(ty4_manual**2) / 2 + 5 * ty4_manual - 11
    y5_t_manual = -(ty5_manual**2) + 8 * ty5_manual - 15
    y6_t_manual = np.zeros_like(ty6_manual)

    y_t_manual = np.concatenate(
        (y1_t_manual, y2_t_manual, y3_t_manual, y4_t_manual, y5_t_manual, y6_t_manual)
    )
    ty_manual = np.concatenate(
        (ty1_manual, ty2_manual, ty3_manual, ty4_manual, ty5_manual, ty6_manual)
    )

elif (first_signal == "b" and second_signal == "e") or (
    first_signal == "e" and second_signal == "b"
):
    ty1_manual = np.arange(-5, -3, Delta)
    ty2_manual = np.arange(-3, -1, Delta)
    ty3_manual = np.arange(-1, 0, Delta)
    ty4_manual = np.arange(0, 1, Delta)
    ty5_manual = np.arange(1, 2, Delta)
    ty6_manual = np.arange(2, 3, Delta)
    ty7_manual = np.arange(3, 4, Delta)
    ty8_manual = np.arange(4, 5, Delta)
    ty9_manual = np.arange(5, 7 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = (ty2_manual**2) / 4 + 3 * ty2_manual / 2 + 9 / 4
    y3_t_manual = (ty3_manual**2) / 4 + 3 * ty3_manual / 2 + 9 / 4
    y4_t_manual = -(ty4_manual**2) / 4 + 3 * ty4_manual / 2 + 9 / 4
    y5_t_manual = -(ty5_manual**2) + 3 * ty5_manual + 3 / 2
    y6_t_manual = -ty6_manual + 11 / 2
    y7_t_manual = -(ty7_manual**2) / 2 + 2 * ty7_manual + 1
    y8_t_manual = (ty8_manual**2) - 10 * ty8_manual + 25
    y9_t_manual = np.zeros_like(ty9_manual)

    y_t_manual = np.concatenate(
        (
            y1_t_manual,
            y2_t_manual,
            y3_t_manual,
            y4_t_manual,
            y5_t_manual,
            y6_t_manual,
            y7_t_manual,
            y8_t_manual,
            y9_t_manual,
        )
    )
    ty_manual = np.concatenate(
        (
            ty1_manual,
            ty2_manual,
            ty3_manual,
            ty4_manual,
            ty5_manual,
            ty6_manual,
            ty7_manual,
            ty8_manual,
            ty9_manual,
        )
    )

elif (first_signal == "b" and second_signal == "f") or (
    first_signal == "f" and second_signal == "b"
):
    ty1_manual = np.arange(0, 1, Delta)
    ty2_manual = np.arange(1, 2, Delta)
    ty3_manual = np.arange(2, 3, Delta)
    ty4_manual = np.arange(3, 4, Delta)
    ty5_manual = np.arange(4, 5, Delta)
    ty6_manual = np.arange(5, 6 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = (np.exp(ty2_manual) - np.exp(1)) * np.exp(-ty2_manual)
    y3_t_manual = 1 - np.exp(-1) + 0 * ty3_manual
    y4_t_manual = ((2 * np.exp(1) - 1) * np.exp(ty4_manual) - np.exp(4)) * np.exp(
        -ty4_manual - 1
    )
    y5_t_manual = -2 * (np.exp(ty5_manual) - np.exp(5)) * np.exp(-ty5_manual - 1)
    y6_t_manual = np.zeros_like(ty6_manual)

    y_t_manual = np.concatenate(
        (y1_t_manual, y2_t_manual, y3_t_manual, y4_t_manual, y5_t_manual, y6_t_manual)
    )
    ty_manual = np.concatenate(
        (ty1_manual, ty2_manual, ty3_manual, ty4_manual, ty5_manual, ty6_manual)
    )

elif (first_signal == "c" and second_signal == "f") or (
    first_signal == "f" and second_signal == "c"
):
    ty1_manual = np.arange(-3, -2, Delta)
    ty2_manual = np.arange(-2, -1, Delta)
    ty3_manual = np.arange(-1, 0, Delta)
    ty4_manual = np.arange(0, 1, Delta)
    ty5_manual = np.arange(1, 2, Delta)
    ty6_manual = np.arange(2, 3, Delta)
    ty7_manual = np.arange(3, 4 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = -(np.exp(ty2_manual + 2) - 1) * np.exp(-ty2_manual - 2)
    y3_t_manual = -(np.exp(1) - 1) * np.exp(-1) + 0 * ty3_manual
    y4_t_manual = 1 + np.exp(-1) - 2 * np.exp(-ty4_manual)
    y5_t_manual = 1 - np.exp(-1) + 0 * ty5_manual
    y6_t_manual = -(np.exp(ty6_manual) - np.exp(3)) * np.exp(-ty6_manual - 1)
    y7_t_manual = np.zeros_like(ty7_manual)

    y_t_manual = np.concatenate(
        (
            y1_t_manual,
            y2_t_manual,
            y3_t_manual,
            y4_t_manual,
            y5_t_manual,
            y6_t_manual,
            y7_t_manual,
        )
    )
    ty_manual = np.concatenate(
        (
            ty1_manual,
            ty2_manual,
            ty3_manual,
            ty4_manual,
            ty5_manual,
            ty6_manual,
            ty7_manual,
        )
    )

elif (first_signal == "c" and second_signal == "d") or (
    first_signal == "d" and second_signal == "c"
):
    ty1_manual = np.arange(-5, -3, Delta)
    ty2_manual = np.arange(-3, -1, Delta)
    ty3_manual = np.arange(-1, 1, Delta)
    ty4_manual = np.arange(1, 3, Delta)
    ty5_manual = np.arange(3, 5 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = -(ty2_manual**2) / 2 - 2 * ty2_manual - 3 / 2
    y3_t_manual = (ty3_manual - 1) * (ty3_manual + 1)
    y4_t_manual = -(ty4_manual**2) / 2 + 2 * ty4_manual - 3 / 2
    y5_t_manual = np.zeros_like(ty5_manual)

    y_t_manual = np.concatenate(
        (y1_t_manual, y2_t_manual, y3_t_manual, y4_t_manual, y5_t_manual)
    )
    ty_manual = np.concatenate(
        (ty1_manual, ty2_manual, ty3_manual, ty4_manual, ty5_manual)
    )

elif (first_signal == "d" and second_signal == "e") or (
    first_signal == "e" and second_signal == "d"
):
    ty1_manual = np.arange(-7, -5, Delta)
    ty2_manual = np.arange(-5, -3, Delta)
    ty3_manual = np.arange(-3, -1, Delta)
    ty4_manual = np.arange(-1, 0, Delta)
    ty5_manual = np.arange(0, 1, Delta)
    ty6_manual = np.arange(1, 2, Delta)
    ty7_manual = np.arange(2, 4 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = (1 / 12) * (
        ty2_manual**3 + 12 * (ty2_manual**2) + 45 * ty2_manual + 50
    )
    y3_t_manual = (
        (-1 / 12) * (ty3_manual**3)
        - (1 / 2) * (ty3_manual**2)
        - 3 * ty3_manual / 4
        - 1 / 3
    )
    y4_t_manual = (-(ty4_manual**3) + 3 * ty4_manual + 2) / 6
    y5_t_manual = -(ty5_manual**2) / 2 + ty5_manual / 2 + 1 / 3
    y6_t_manual = (ty6_manual**3) / 6 - (ty6_manual**2) / 2 + 2 / 3
    y7_t_manual = np.zeros_like(ty7_manual)

    y_t_manual = np.concatenate(
        (
            y1_t_manual,
            y2_t_manual,
            y3_t_manual,
            y4_t_manual,
            y5_t_manual,
            y6_t_manual,
            y7_t_manual,
        )
    )
    ty_manual = np.concatenate(
        (
            ty1_manual,
            ty2_manual,
            ty3_manual,
            ty4_manual,
            ty5_manual,
            ty6_manual,
            ty7_manual,
        )
    )

elif (first_signal == "e" and second_signal == "f") or (
    first_signal == "f" and second_signal == "e"
):
    ty1_manual = np.arange(-5, -4, Delta)
    ty2_manual = np.arange(-4, -3, Delta)
    ty3_manual = np.arange(-3, -2, Delta)
    ty4_manual = np.arange(-2, -1, Delta)
    ty5_manual = np.arange(-1, 0, Delta)
    ty6_manual = np.arange(0, 1, Delta)
    ty7_manual = np.arange(1, 2, Delta)
    ty8_manual = np.arange(2, 3 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = (np.exp(-ty2_manual) * 0.5) * (
        np.exp(ty2_manual) * (ty2_manual + 3) + 0.018316
    )
    y3_t_manual = (ty3_manual + 3.58148) * (0.31606)
    y4_t_manual = (
        (-0.18394)
        * ((ty4_manual - 3.43656) * np.exp(ty4_manual) + 0.367879)
        * np.exp(-ty4_manual)
    )
    y5_t_manual = (1 - np.exp(-1)) * (ty5_manual / ty5_manual)
    y6_t_manual = -ty6_manual - np.exp(-ty6_manual) - np.exp(-1) + 2
    y7_t_manual = (np.exp(ty7_manual) * (ty7_manual - 3) + np.exp(2)) * (
        np.exp(-ty7_manual - 1)
    )
    y8_t_manual = np.zeros_like(ty8_manual)

    y_t_manual = np.concatenate(
        (
            y1_t_manual,
            y2_t_manual,
            y3_t_manual,
            y4_t_manual,
            y5_t_manual,
            y6_t_manual,
            y7_t_manual,
            y8_t_manual,
        )
    )
    ty_manual = np.concatenate(
        (
            ty1_manual,
            ty2_manual,
            ty3_manual,
            ty4_manual,
            ty5_manual,
            ty6_manual,
            ty7_manual,
            ty8_manual,
        )
    )

elif (first_signal == "a" and second_signal == "a") or (
    first_signal == "a" and second_signal == "a"
):
    ty1_manual = np.arange(-3, -2, Delta)
    ty2_manual = np.arange(-2, 0, Delta)
    ty3_manual = np.arange(0, 2, Delta)
    ty4_manual = np.arange(2, 3 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = ty2_manual + 2
    y3_t_manual = -ty3_manual + 2
    y4_t_manual = np.zeros_like(ty4_manual)

    y_t_manual = np.concatenate((y1_t_manual, y2_t_manual, y3_t_manual, y4_t_manual))
    ty_manual = np.concatenate((ty1_manual, ty2_manual, ty3_manual, ty4_manual))

elif (first_signal == "b" and second_signal == "b") or (
    first_signal == "b" and second_signal == "b"
):
    ty1_manual = np.arange(0, 2, Delta)
    ty2_manual = np.arange(2, 4, Delta)
    ty3_manual = np.arange(4, 5, Delta)
    ty4_manual = np.arange(5, 6, Delta)
    ty5_manual = np.arange(6, 7, Delta)
    ty6_manual = np.arange(7, 8, Delta)
    ty7_manual = np.arange(8, 10 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = ty2_manual - 2
    y3_t_manual = 3 * ty3_manual - 10
    y4_t_manual = -ty4_manual + 10
    y5_t_manual = 4 + 0 * ty5_manual
    y6_t_manual = -4 * ty6_manual + 32
    y7_t_manual = np.zeros_like(ty7_manual)

    y_t_manual = np.concatenate(
        (
            y1_t_manual,
            y2_t_manual,
            y3_t_manual,
            y4_t_manual,
            y5_t_manual,
            y6_t_manual,
            y7_t_manual,
        )
    )
    ty_manual = np.concatenate(
        (
            ty1_manual,
            ty2_manual,
            ty3_manual,
            ty4_manual,
            ty5_manual,
            ty6_manual,
            ty7_manual,
        )
    )

elif (first_signal == "c" and second_signal == "c") or (
    first_signal == "c" and second_signal == "c"
):
    ty1_manual = np.arange(-6, -4, Delta)
    ty2_manual = np.arange(-4, -2, Delta)
    ty3_manual = np.arange(-2, 0, Delta)
    ty4_manual = np.arange(0, 2, Delta)
    ty5_manual = np.arange(2, 4, Delta)
    ty6_manual = np.arange(4, 6 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = ty2_manual + 4
    y3_t_manual = -3 * ty3_manual - 4
    y4_t_manual = 3 * ty4_manual - 4
    y5_t_manual = -ty5_manual + 4
    y6_t_manual = np.zeros_like(ty6_manual)

    y_t_manual = np.concatenate(
        (y1_t_manual, y2_t_manual, y3_t_manual, y4_t_manual, y5_t_manual, y6_t_manual)
    )
    ty_manual = np.concatenate(
        (ty1_manual, ty2_manual, ty3_manual, ty4_manual, ty5_manual, ty6_manual)
    )

elif (first_signal == "c" and second_signal == "e") or (
    first_signal == "e" and second_signal == "c"
):
    ty1_manual = np.arange(-8, -6, Delta)
    ty2_manual = np.arange(-6, -4, Delta)
    ty3_manual = np.arange(-4, -2, Delta)
    ty4_manual = np.arange(-2, -1, Delta)
    ty5_manual = np.arange(-1, 0, Delta)
    ty6_manual = np.arange(0, 1, Delta)
    ty7_manual = np.arange(1, 2, Delta)
    ty8_manual = np.arange(2, 3, Delta)
    ty9_manual = np.arange(3, 5 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = -0.25 * ty2_manual**2 - 3 * ty2_manual - 9
    y3_t_manual = 0.5 * ty3_manual**2 + 3 * ty3_manual + 3
    y4_t_manual = 0.25 * ty4_manual**2 + 2 * ty4_manual + 2
    y5_t_manual = -0.25 * ty5_manual**2 + ty5_manual + 1.5
    y6_t_manual = ty6_manual + 1 + (-2 * (ty6_manual**2) + 1) / 2
    y7_t_manual = -1 * ty7_manual + 5 / 2
    y8_t_manual = -(-(ty8_manual**2) + 4 * ty8_manual - 3) / 2 - ty8_manual + 3
    y9_t_manual = np.zeros_like(ty9_manual)

    ty_manual = np.concatenate(
        (
            ty1_manual,
            ty2_manual,
            ty3_manual,
            ty4_manual,
            ty5_manual,
            ty6_manual,
            ty7_manual,
            ty8_manual,
            ty9_manual,
        )
    )
    y_t_manual = np.concatenate(
        (
            y1_t_manual,
            y2_t_manual,
            y3_t_manual,
            y4_t_manual,
            y5_t_manual,
            y6_t_manual,
            y7_t_manual,
            y8_t_manual,
            y9_t_manual,
        )
    )

elif (first_signal == "f" and second_signal == "f") or (
    first_signal == "f" and second_signal == "f"
):
    ty1_manual = np.arange(-2, 0, Delta)
    ty2_manual = np.arange(0, 1, Delta)
    ty3_manual = np.arange(1, 2, Delta)
    ty4_manual = np.arange(2, 4 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = np.exp(-1 * ty2_manual) * ty2_manual
    y3_t_manual = 2 * np.exp(-1 * ty3_manual) - np.exp(-1 * ty3_manual) * ty3_manual
    y4_t_manual = np.zeros_like(ty4_manual)

    ty_manual = np.concatenate((ty1_manual, ty2_manual, ty3_manual, ty4_manual))
    y_t_manual = np.concatenate((y1_t_manual, y2_t_manual, y3_t_manual, y4_t_manual))

elif (first_signal == "d" and second_signal == "d") or (
    first_signal == "d" and second_signal == "d"
):
    ty1_manual = np.arange(-4, -2, Delta)
    ty2_manual = np.arange(-2, 0, Delta)
    ty3_manual = np.arange(0, 2, Delta)
    ty4_manual = np.arange(2, 4 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = (
        (-1 / 3) * (ty2_manual**3 + 3 * ty2_manual**2 + 3 * ty2_manual + 2)
    ) + (0.5 * (ty2_manual**2 + 2 * ty2_manual) * ty2_manual)
    y3_t_manual = (
        (-1 / 3) * (-(ty3_manual**3) + 3 * (ty3_manual**2) - 3 * ty3_manual + 2)
    ) + (0.5 * (-(ty3_manual**2) + 2 * ty3_manual) * ty3_manual)
    y4_t_manual = np.zeros_like(ty4_manual)

    ty_manual = np.concatenate((ty1_manual, ty2_manual, ty3_manual, ty4_manual))
    y_t_manual = np.concatenate((y1_t_manual, y2_t_manual, y3_t_manual, y4_t_manual))

elif (first_signal == "e" and second_signal == "e") or (
    first_signal == "e" and second_signal == "e"
):
    ty1_manual = np.arange(-10, -8, Delta)
    ty2_manual = np.arange(-8, -6, Delta)
    ty3_manual = np.arange(-6, -4, Delta)
    ty4_manual = np.arange(-4, -3, Delta)
    ty5_manual = np.arange(-3, -2, Delta)
    ty6_manual = np.arange(-2, -1, Delta)
    ty7_manual = np.arange(-1, 0, Delta)
    ty8_manual = np.arange(0, 1, Delta)
    ty9_manual = np.arange(1, 2, Delta)
    ty10_manual = np.arange(2, 4 + Delta, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = (
        0.04166675 * ty2_manual**3
        + 1.00001 * ty2_manual**2
        + 8.00004 * ty2_manual
        + 21.33333875
    )
    y3_t_manual = (
        -0.04166 * ty3_manual**3 - 0.5 * ty3_manual**2 - 1 * ty3_manual + 3.3333
    )
    y4_t_manual = (
        -0.16667 * ty4_manual**3
        - 2.00002 * ty4_manual**2
        - 7.00008 * ty4_manual
        - 4.666775
    )
    y5_t_manual = -0.5 * ty5_manual**2 - 2.49995 * ty5_manual - 0.16667
    y5b_t_manual = (
        0.16667 * ty6_manual**3
        + 0.500005 * ty6_manual**2
        - 0.499975 * ty6_manual
        + 1.16668
    )
    y6_t_manual = -ty7_manual + 1
    y6b_t_manual = (ty8_manual**3) / 6 - ty8_manual + 1
    y7_t_manual = (
        (-1 / 3) * (-(ty9_manual**3) + 3 * ty9_manual**2 - 3 * ty9_manual + 2)
        + ty9_manual**2
        - 3 * ty9_manual
        + 2
        + 0.5 * (ty9_manual) * (-(ty9_manual**2) + 2 * ty9_manual)
    )
    y8_t_manual = np.zeros_like(ty10_manual)

    ty_manual = np.concatenate(
        (
            ty1_manual,
            ty2_manual,
            ty3_manual,
            ty4_manual,
            ty5_manual,
            ty6_manual,
            ty7_manual,
            ty8_manual,
            ty9_manual,
            ty10_manual,
        )
    )
    y_t_manual = np.concatenate(
        (
            y1_t_manual,
            y2_t_manual,
            y3_t_manual,
            y4_t_manual,
            y5_t_manual,
            y5b_t_manual,
            y6_t_manual,
            y6b_t_manual,
            y7_t_manual,
            y8_t_manual,
        )
    )


else:
    ty1_manual = np.arange(-1, 0, Delta)
    ty2_manual = np.arange(0, 2, Delta)
    ty3_manual = np.arange(2, 3, Delta)
    ty4_manual = np.arange(3, 4, Delta)
    ty5_manual = np.arange(4, 5, Delta)
    ty6_manual = np.arange(5, 6, Delta)

    y1_t_manual = np.zeros_like(ty1_manual)
    y2_t_manual = ty2_manual
    y3_t_manual = ty3_manual
    y4_t_manual = 6 - ty4_manual
    y5_t_manual = 10 - 2 * ty5_manual
    y6_t_manual = np.zeros_like(ty6_manual)

    y_t_manual = np.concatenate(
        (y1_t_manual, y2_t_manual, y3_t_manual, y4_t_manual, y5_t_manual, y6_t_manual)
    )
    ty_manual = np.concatenate(
        (ty1_manual, ty2_manual, ty3_manual, ty4_manual, ty5_manual, ty6_manual)
    )

# Gráfica Convolución
plt.figure(2)
plt.plot(ty, y_t, label="Convolución con Numpy")
plt.plot(ty_manual, y_t_manual, "r", label="Convolución Manual")
plt.title("Convolución Manual vs. Convolución con Numpy")
plt.legend()
plt.show()
