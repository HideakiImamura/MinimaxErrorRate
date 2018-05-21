import numpy as np
import pandas as pd
import sys
sys.path.append("/Users/forute/Documents/Academy/Resaech/Clustering_Worker")
import model.Majority_Voting as mv
import model.Dawid_Skene as ds
import model.Worker_Clustering as wcv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


data = pd.read_csv("./experiment-6-2/rte/rte.standardized.tsv", delimiter="\t")
task = np.array([], dtype=int)
worker = np.array([], dtype=int)
g = np.array([], dtype=int)
task_worker_label = np.array([], dtype=int).reshape((0, 3))
for _, row in data.iterrows():
    ii = np.where(task == row["orig_id"])[0]
    jj = np.where(worker == row["!amt_worker_ids"])[0]
    g_k = int(row["gold"])
    if len(ii) == 0:
        i = len(task)
        g = np.append(g, [g_k])
        task = np.append(task, [row["orig_id"]])
    elif len(ii) == 1:
        i = ii[0]
    else:
        raise ValueError("Task is not good")
    if len(jj) == 0:
        j = len(worker)
        worker = np.append(worker, [row["!amt_worker_ids"]])
    elif len(jj) == 1:
        j = jj[0]
    else:
        raise ValueError("Worker is not good")
    k = int(row["response"])
    task_worker_label = np.concatenate((task_worker_label, [[i, j, k]]))

n = 800
m = 164
K = 2
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
    data.to_csv("./experiment-6-2/rte/rte.csv")
    plt.plot(np.arange(L), acc_mv * np.ones(L), label="MV", color="blue")
    plt.plot(np.arange(L), acc_ds * np.ones(L), label="DS", color="red")
    plt.plot(np.arange(L), acc_evi_list, label="proposed", color="green")
    plt.xlabel("L")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig("./experiment-6-2/rte/rte.pdf")
    plt.clf()
