import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append("/Users/forute/Documents/Academy/Resaech/Clustering_Worker")
now = "./experiment_syn/worker_num/for_paper/"

data = pd.read_csv(now + "data_n1000m100K2N2c0.3L10.csv")
mv = list(data['MV'])[1:51]
ds = list(data['DS'])[1:51]
wc = list(data['proposed1(wcv)'])[1:51]
xs = np.arange(0, 50) / 100
print(len(mv))

plt.figure(figsize=(5, 4))
plt.scatter(xs, mv, label="MV", color="blue", marker='o')
plt.scatter(xs, wc, label="Proposed", color="green", marker='^')
plt.scatter(xs, ds, label="DS", color="red", marker='+')
ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
plt.legend([handles[0], handles[2], handles[1]], [labels[0], labels[2], labels[1]], loc="lower left", fontsize=20)

plt.ylabel("Accuracy", fontsize=20)
plt.xlabel("Proportion of adversaries", fontsize=20)

plt.ylim(0.0, 1.1)
plt.xlim(0.0, 0.5)

plt.tight_layout()
plt.savefig(now + "N2.pdf")
