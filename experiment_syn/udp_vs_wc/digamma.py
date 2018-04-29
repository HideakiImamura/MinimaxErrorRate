import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

n = 100
K = 2
x = np.concatenate(((np.arange(10) + 1) / 10, np.arange(n) + 1))
y = sp.digamma(x) - sp.digamma(K * x)
plt.plot(x, y)
print(y)
plt.show()