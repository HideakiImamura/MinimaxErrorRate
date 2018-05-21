import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import sys
import os

sys.path.append("/Users/forute/Documents/Academy/Resaech/Clustering_Worker")
import model.Dawid_Skene as ds
import model.Worker_Clustering as wcv


def R(rho, pi):
    delta = 1e-100
    m = pi.shape[0]
    K = pi.shape[1]
    kl = entropy(pk=rho)
    for j in range(m):
        for g in range(K):
            for g_ in range(K):
                kl -= rho[g] * rho[g_] * entropy(pk=(pi[j, g] + delta), qk=(pi[j, g_] + delta))
    return kl


n = 800
m = 164
K = 2
epsilon = 1e-2
Ls = np.arange(1, 165)

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

name = "./experiment-6-2/rte/data_" + \
       "n" + str(n) + \
       "m" + str(m) + \
       "K" + str(K)
if os.path.exists(name):
    data = pd.read_csv(name + "csv")[["WC"]]
else:
    data = pd.DataFrame(columns=["WC"])

g_ds, [pi_ds, rho_ds] = ds.DS_elbo_debug(task_worker_label, n, m, K, epsilon=epsilon)
R_ds = R(rho_ds, pi_ds)
print("R_ds      = {}".format(R_ds))

R_wc_list = []
for L in Ls:
    g_wcv, [pi_wcv, rho_wcv], _ = wcv.wcv(task_worker_label, n, m, K, L, epsilon=epsilon)
    R_wc = R(rho_wcv, pi_wcv)
    R_wc_list.append(R_wc)
    print("R_wc      = {}".format(R_wc))

    acc = pd.DataFrame([{"WC": R_wc}])
    data = data.append(acc)
    data.to_csv(name + "csv")

    plt.plot(Ls[0:L], R_ds * np.ones(L), label="DS", color="red")
    plt.plot(Ls[0:L], R_wc_list, label="proposed", color="green")
    plt.xlabel("the number of cluster L")
    plt.ylabel("R")
    plt.legend(loc="upper right")
    plt.savefig(name + ".pdf")
    plt.clf()
