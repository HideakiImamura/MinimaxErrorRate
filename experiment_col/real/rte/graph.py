import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import sys

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
ms = [164]
K = 2
epsilon = 1e-2
Ls = np.arange(1, 165)

for m in ms:
    data = pd.DataFrame(columns=["L", "R"])

    g_ds, [pi_ds, rho_ds] = ds.DS_elbo_debug(task_worker_label, n, m, K, epsilon=epsilon)
    L_ds = 1.0 - accuracy_score(g, g_ds)
    R_ds = R(rho_ds, pi_ds)
    print("L_ds, R_ds   = {0}, {1}".format(L_ds, R_ds))
    data = data.append(pd.Series([L_ds, R_ds], index=data.columns), ignore_index=True)

    L_wc_s = []
    R_wc_s = []
    for L in Ls:
        g_wcv, [pi_wcv, rho_wcv], _ = wcv.wcv(task_worker_label, n, m, K, L, epsilon=epsilon)
        L_wcv = 1.0 - accuracy_score(g, g_wcv)
        R_wcv = R(rho_wcv, pi_wcv)
        print("L = {2}: L_wc, R_wc   = {0}, {1}".format(L_wcv, R_wcv, L))
        data = data.append(pd.Series([L_wcv, R_wcv], index=data.columns), ignore_index=True)
        data.to_csv("./experiment-6-3/real/rte/data_" +
                    "n" + str(n) +
                    "m" + str(m) +
                    "K" + str(K) + ".csv")

        L_wc_s.append(L_wcv)
        R_wc_s.append(R_wcv)

        fig = plt.figure(figsize=(16, 6))
        axL = fig.add_subplot(121)
        axR = fig.add_subplot(122)
        axL.plot(Ls[0:L], L_wc_s, label="proposed", color="green")
        axL.plot(Ls[0:L], L_ds * np.ones(L), label="DS", color="red")
        axR.plot(Ls[0:L], R_wc_s, label="proposed", color="green")
        axR.plot(Ls[0:L], R_ds * np.ones(L), label="DS", color="red")

        axL.set_title("$L(\hat{G}, G)$")
        axR.set_title("$R($" + r"$\rho$" + ", $\pi)$")

        axL.set_xlabel("$L$")
        axR.set_xlabel("$L$")

        axL.legend(loc="upper right")
        axR.legend(loc="upper right")
        fig.savefig("./experiment-6-3/real/rte/graph_" +
                    "n" + str(n) +
                    "m" + str(m) +
                    "K" + str(K) + ".png")
        fig.clf()
        plt.clf()
