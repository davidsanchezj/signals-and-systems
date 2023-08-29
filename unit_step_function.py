import numpy as np
import matplotlib.pyplot as plt

def unit_step(il, sl, steps, shift):
    # independent variable and its length
    t = np.linspace(il,sl,steps)
    N = len(t)

    # amplitude
    A = 1

    u = np.zeros(N)

    for i in range (0,N):
        if t[i] >= shift:
            u[i] = 1

    return u, t
   
x, t_values = unit_step(-10,10,1000,2)

print("Unit Step Function Values:", x)
plt.plot(t_values, x)
plt.show()
