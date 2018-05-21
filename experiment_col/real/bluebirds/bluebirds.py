import numpy as np
import yaml
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
sys.path.append("/Users/forute/Documents/Academy/Resaech/Clustering_Worker")
import model.Majority_Voting as mv
import model.Dawid_Skene as ds
import model.Worker_Clustering as wcv
import matplotlib.pyplot as plt



'''
f = open("./experiment-6-2/bluebirds/gt.yaml", "r")
task_label = yaml.load(f)
task = np.array([], dtype=int)
G = np.array([], dtype=int)
for i, k in task_label.items():
    task = np.append(task, [i])
    G = np.append(G, [int(k)])
f.close()
pd.DataFrame(G).to_csv("./experiment-6-2/bluebirds/gt.csv")


f = open("./experiment-6-2/bluebirds/labels.yaml", "r")
worker_task_label = yaml.load(f)
worker = np.array([], dtype=int)
task_worker_label = np.array([], dtype=int).reshape(0, 3)
jj = 0
for j, ik in worker_task_label.items():
    worker = np.append(worker, [j])
    for i, k in ik.items():
        ii = np.where(task == i)[0][0]
        task_worker_label = np.concatenate((task_worker_label, [[ii, jj, int(k)]]))
    jj += 1

pd.DataFrame(task_worker_label).to_csv("./experiment-6-2/bluebirds/twl.csv")
'''

n = 108
m = 39
K = 2
epsilon = 1e-2

task_worker_label = np.array(pd.read_csv("./experiment-6-2/bluebirds/twl.csv")[['0', '1', '2']].values.tolist())
g = np.array(pd.read_csv("./experiment-6-2/bluebirds/gt.csv")[['0']].values.tolist()).T[0]

g_mv = mv.MV(task_worker_label, n, m, K)
acc_mv = accuracy_score(g, g_mv)

g_ds, _ = ds.DS_elbo_debug(task_worker_label, n, m, K, epsilon=epsilon)
acc_ds = accuracy_score(g, g_ds)

L_list = np.arange(1, m + 1)
acc_evi_list = []
for L in L_list:
    g_evi, _, _ = wcv.wcv(task_worker_label, n, m, K, L, epsilon=epsilon)
    acc_evi = accuracy_score(g, g_evi)
    acc_evi_list.append(acc_evi)

data = pd.DataFrame({'MV': acc_mv,
                     'DS': acc_ds,
                     'proposed': acc_evi_list})
data.to_csv("./experiment-6-2/bluebirds/L1_m.csv")

plt.plot(L_list, acc_mv * np.ones(len(L_list)), label="MV", color="blue")
plt.plot(L_list, acc_ds * np.ones(len(L_list)), label="DS", color="red")
plt.plot(L_list, acc_evi_list, label="proposed", color="green")
plt.xlabel("L")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig("./experiment-6-2/bluebirds/bird.pdf")
