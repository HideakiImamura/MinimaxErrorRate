import numpy as np
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
sys.path.append("/Users/forute/Documents/Academy/Resaech/Clustering_Worker")
import model.Majority_Voting as mv
import model.Dawid_Skene as ds
import model.Worker_Clustering as wcv
import matplotlib.pyplot as plt


f = open("./experiment-6-2/dog/dog_truth.txt", "r")
lines = f.readlines()
f.close()
task = np.array([], dtype=int)
g = np.array([], dtype=int)
for line in lines[1:]:
    i, k = [int(x) for x in line.split("\t")]
    ii = np.where(task == i)[0]
    if len(ii) == 0:
        task = np.append(task, [i])
        g = np.append(g, [k - 1])

task_worker_label = np.array([], dtype=int).reshape((0, 3))
worker = np.array([], dtype=int)
f = open("./experiment-6-2/dog/dog_crowd.txt", "r")
lines = f.readlines()
f.close()
for line in lines[1:]:
    i, j_, k = [int(x) for x in line.split("\t")]
    jj = np.where(worker == j_)[0]
    if len(jj) == 0:
        j = len(jj)
        worker = np.append(worker, [j_])
    elif len(jj) == 1:
        j = jj[0]
    else:
        raise ValueError("Worker is not good")
    task_worker_label = np.concatenate((task_worker_label, [[i - 1, j - 1, k - 1]]))

print(task.shape)
print(g.shape)

n = 807
m = 109
K = 4
epsilon = 1e-2

g_mv = mv.MV(task_worker_label, n, m, K)
acc_mv = accuracy_score(g, g_mv)
print(acc_mv)

g_ds, _ = ds.DS_elbo_debug(task_worker_label, n, m, K, epsilon=epsilon)
acc_ds = accuracy_score(g, g_ds)
print(acc_ds)

L_list = np.arange(1, m)
acc_evi_list = []
for L in L_list:
    g_evi, _, _ = wcv.wcv(task_worker_label, n, m, K, L, epsilon=epsilon)
    acc_evi = accuracy_score(g, g_evi)
    print(L, " ", acc_evi)
    acc_evi_list.append(acc_evi)

    data = pd.DataFrame({'MV': acc_mv,
                     'DS': acc_ds,
                     'proposed': acc_evi_list})
    data.to_csv("./experiment-6-2/dog/L1_m.csv")

    plt.plot(np.arange(L) + 1, acc_mv * np.ones(L), label="MV", color="blue")
    plt.plot(np.arange(L) + 1, acc_ds * np.ones(L), label="DS", color="red")
    plt.plot(np.arange(L) + 1, acc_evi_list, label="proposed", color="green")
    plt.xlabel("L")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig("./experiment-6-2/dog/L1_m.pdf")
    plt.clf()