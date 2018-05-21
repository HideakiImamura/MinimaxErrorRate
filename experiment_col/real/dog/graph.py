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

f = open("./experiment-6-3/real/dog/dog_truth.txt", "r")
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
f = open("./experiment-6-3/real/dog/dog_crowd.txt", "r")
lines = f.readlines()
f.close()
s = set()
for line in lines[1:]:
    i, j_, k = [int(x) for x in line.split("\t")]
    s.add(j_)
    jj = np.where(worker == j_)[0]
    if len(jj) == 0:
        j = len(jj)
        worker = np.append(worker, [j_])
    elif len(jj) == 1:
        j = jj[0]
    else:
        raise ValueError("Worker is not good")
    task_worker_label = np.concatenate((task_worker_label, [[i - 1, j - 1, k - 1]]))

print(np.sort(list(s)))

n = 807
ms = [109]
K = 4
epsilon = 1e-2
Ls = np.arange(1, 53)

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
        data.to_csv("./experiment-6-3/real/dog/data_" +
                    "n" + str(n) +
                    "m" + str(m) +
                    "K" + str(K) + ".csv")

        L_wc_s.append(L_wcv)
        R_wc_s.append(R_wcv)

        fig = plt.figure(figsize=(8, 3))
        axL = fig.add_subplot(121)
        axR = fig.add_subplot(122)
        axL.plot(Ls[0:L], L_wc_s, label="proposed", color="green")
        axR.plot(Ls[0:L], R_wc_s, label="proposed", color="green")

        axL.set_title("$L(\hat{G}, G)$")
        axR.set_title("$R($" + r"$\rho$" + ", $\pi)$")

        axL.set_xlabel("$L$")
        axR.set_xlabel("$L$")

        fig.savefig("./experiment-6-3/real/dog/graph_" +
                    "n" + str(n) +
                    "m" + str(m) +
                    "K" + str(K) + ".pdf")
        fig.clf()
        plt.clf()
