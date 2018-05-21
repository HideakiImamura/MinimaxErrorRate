import numpy as np
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
sys.path.append("/Users/forute/Documents/Academy/Resaech/Robustness_of_Dawid-Skene_Model/experiment-6-2")
import model.Majority_Voting as mv
import model.Dawid_Skene as ds
import model.Worker_Clusterring_with_Dirichlet_Prior as evi
import matplotlib.pyplot as plt


f = open("./trec_truth.txt", "r")
lines = f.readlines()
f.close()
task_known_g = np.array([], dtype=int)
g = np.array([], dtype=int)
for line in lines[1:]:
    i, k = [int(x) for x in line.split("\t")]
    ii = np.where(task_known_g == i)[0]
    if len(ii) == 0:
        task_known_g = np.append(task_known_g, [i])
        g = np.append(g, [k - 1])

task_worker_label = np.array([], dtype=int).reshape((0, 3))
task = np.array([], dtype=int)
worker = np.array([], dtype=int)
f = open("./trec_crowd.txt", "r")
lines = f.readlines()
f.close()
for line in lines[1:]:
    i_, j_, k = [int(x) for x in line.split("\t")]
    ii = np.where(task == i_)[0]
    jj = np.where(worker == j_)[0]
    if len(ii) == 0:
        i = len(ii)
        task = np.append(task, [i_])
    elif len(ii) == 1:
        i = ii[0]
    else:
        raise ValueError("Task is not good")
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
print(task)
print(g)

n = task.shape[0]
m = worker.shape[0]
K = len(set(g))
epsilon = 1e-2

g_mv = mv.MV(task_worker_label, n, m, K)
acc_mv = accuracy_score(g, g_mv[task_known_g])
print(acc_mv)

g_ds, _ = ds.DS_elbo_debug(task_worker_label, n, m, K, epsilon=epsilon)
acc_ds = accuracy_score(g, g_ds[task_known_g])
print(acc_ds)

L_list = np.arange(1, m)
acc_evi_list = []
for L in L_list:
    g_evi, _, _ = evi.EVI(task_worker_label, n, m, K, L, epsilon=epsilon)
    acc_evi = accuracy_score(g, g_evi[task_known_g])
    print(L, " ", acc_evi)
    acc_evi_list.append(acc_evi)

    data = pd.DataFrame({'MV': acc_mv,
                     'DS': acc_ds,
                     'proposed': acc_evi_list})
    data.to_csv("./L1_m.csv")

    plt.plot(np.arange(L) + 1, acc_mv * np.ones(L), label="MV")
    plt.plot(np.arange(L) + 1, acc_ds * np.ones(L), label="DS")
    plt.plot(np.arange(L) + 1, acc_evi_list, label="proposed")
    plt.xlabel("L")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig("./L1_m.png")
    plt.clf()