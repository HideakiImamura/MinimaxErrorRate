import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append("/Users/forute/Documents/Academy/Resaech/Clustering_Worker")
now = "./experiment_syn/power_law/for_paper/"

data = pd.read_csv(now + "ForIRCNdata_n1000m100K2N100c0.3L10.csv")
mv = np.array(data['MV'])
ds = np.array(data['DS'])
wc = np.array(data['proposed1(wcv)'])
xs = np.arange(0, 100) / 100
# print(len(mv))

plt.figure(figsize=(7, 6))
plt.scatter(xs, 1 - mv, label="MV", color="blue", marker='o')
plt.scatter(xs, 1 - wc, label="Proposed", color="green", marker='^')
plt.scatter(xs, 1 - ds, label="DS", color="red", marker='+')

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
plt.legend([handles[0], handles[2], handles[1]], [labels[0], labels[2], labels[1]], loc="upper left", fontsize=20)
# plt.legend(loc="upper right", fontsize=20)

plt.ylabel("Error rate", fontsize=20)
plt.xlabel("Proportion of adversaries", fontsize=20)

plt.ylim(0.0, 0.5)
plt.xlim(0.0, 0.5)

plt.tight_layout()
plt.savefig(now + "sparse_for_ircn.pdf")
