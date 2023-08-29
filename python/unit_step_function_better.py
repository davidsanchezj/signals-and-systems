import numpy as np
import matplotlib.pyplot as plt

def unit_step(il, sl, steps, shift):
    # independent variable and its length
    t = np.linspace(il,sl,steps)
    u = np.where(t >= shift, 1, 0)

    return u, t
   
x, t_values = unit_step(-10,10,1000,-5)

print("Unit Step Function Values:", x)
plt.plot(t_values, x)
plt.show()
