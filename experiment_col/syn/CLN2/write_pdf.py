import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append("/Users/forute/Documents/Academy/Resaech/Clustering_Worker")
now = "./experiment-6-3/syn/CLN2/"

data = pd.read_csv(now + "data_n1000m100K2N3c0.3.csv")
Ls = list(data['L'])
Rs = list(data['R'])
L = np.arange(1, len(Ls) + 1)

fig = plt.figure(figsize=(7, 3))
axL = fig.add_subplot(121)
axR = fig.add_subplot(122)
axL.plot(L, Ls, label="proposed", color="green", linewidth=3)
axR.plot(L, Rs, label="proposed", color="green", linewidth=3)

axL.set_title("$\mathscr{L}(\hat{G}, G)$", fontsize=20)
axR.set_title("$R($" + r"$\rho$" + ", $\pi)$", fontsize=20)

axL.set_xlabel("$L$", fontsize=20)
axR.set_xlabel("$L$", fontsize=20)

plt.tight_layout()
fig.savefig(now + "cln2.pdf")
