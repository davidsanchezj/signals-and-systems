import numpy as np
import matplotlib.pyplot as plt

# independent variable
t = np.arange(0,1,0.01)

# dependent variable
xt = np.sin(2*np.pi*3*t)

# sample the function
plt.subplot(1,2,1)
plt.stem(t,xt)
plt.title("Continuous")

# independent variable
T = [i for i in range(0,7)]

#dependent variable
xT = [-1, 4, -3, 0, 9, 8, 9]

# sample the function
plt.subplot(1,2,2)
plt.stem(T,xT)
plt.title("Discrete Time")

plt.show()
