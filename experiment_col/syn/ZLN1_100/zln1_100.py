import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import sys

sys.path.append("/Users/forute/Documents/Academy/Resaech/Clustering_Worker")
import experiment_syn.power_law.create_power_law_dataset as csd
import model.Dawid_Skene as ds
import model.proposed_wcv as wcv


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

n = 1000
ms = [100]
K = 2
Ns = [100]
cs = [0.3]
Ls = np.arange(1, 101)
x = 0.3
epsilon = 1e-2


for m in ms:
    for N in Ns:
        for c in cs:
            task_class = csd.create_data(n, K, [c, 1 - c])
            task = np.hsplit(task_class, [1])[0].ravel()
            true_class = np.hsplit(task_class, [1])[1].ravel()
            g = np.array(sorted(task_class, key=lambda pair: pair[0]))[:, 1]

            data = pd.DataFrame(columns=["L", "R"])

            task_worker_class = csd.task_worker_label(m, K, task, true_class, N, [1 - x, 0.0, x])

            g_ds, [pi_ds, rho_ds] = ds.DS_elbo_debug(task_worker_class, n, m, K, epsilon=epsilon)
            L_ds = 1.0 - accuracy_score(g, g_ds)
            R_ds = R(rho_ds, pi_ds)
            print("L_ds, R_ds   = {0}, {1}".format(L_ds, R_ds))
            data = data.append(pd.Series([L_ds, R_ds], index=data.columns), ignore_index=True)

            L_wc_s = []
            R_wc_s = []
            for L in Ls:
                g_wcv, [pi_wcv, rho_wcv], _ = wcv.wcv(task_worker_class, n, m, K, L, epsilon=epsilon)
                L_wcv = 1.0 - accuracy_score(g, g_wcv)
                R_wcv = R(rho_wcv, pi_wcv)
                print("L = {2}: L_wc, R_wc   = {0}, {1}".format(L_wcv, R_wcv, L))
                data = data.append(pd.Series([L_wcv, R_wcv], index=data.columns), ignore_index=True)
                data.to_csv("./experiment_col/syn/ZLN1_100/data_" +
                            "n" + str(n) +
                            "m" + str(m) +
                            "K" + str(K) +
                            "N" + str(N) +
                            "c" + str(c) + ".csv")

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
                fig.savefig("./experiment_col/syn/ZLN1_100/graph_" +
                            "n" + str(n) +
                            "m" + str(m) +
                            "K" + str(K) +
                            "N" + str(N) +
                            "c" + str(c) + ".png")
                fig.clf()
