import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
import sys
import os

sys.path.append("/Users/forute/Documents/Academy/Resaech/Clustering_Worker")
import experiment_syn.worker_num.create_worker_labeling_number_dataset as csd
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

n = 1000
ms = [100]
K = 2
Ns = [3]
cs = [0.3]
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
            name = "./experiment-6-3/syn/cln2/data_" + \
                            "n" + str(n) + \
                            "m" + str(m) + \
                            "K" + str(K) + \
                            "N" + str(N) + \
                            "c" + str(c) + ".csv"
            if os.path.exists(name):
                data = pd.read_csv(name)[["L", "R"]]
            else:
                data = pd.DataFrame(columns=['L', 'R'])

            task_worker_class = csd.task_worker_label(m, K, task, true_class, N, [1 - x, 0.0, x])

            L_wc_s = list(data['L'])
            R_wc_s = list(data['R'])
            print(L_wc_s)
            print(len(L_wc_s))
            Ls = np.arange(len(L_wc_s) + 1, 101)
            for L in Ls:
                g_wcv, [pi_wcv, rho_wcv], _ = wcv.wcv(task_worker_class, n, m, K, L, epsilon=epsilon)
                L_wcv = 1.0 - accuracy_score(g, g_wcv)
                R_wcv = R(rho_wcv, pi_wcv)
                print("L = {2}: L_wc, R_wc   = {0}, {1}".format(L_wcv, R_wcv, L))
                data = data.append(pd.Series([L_wcv, R_wcv], index=data.columns), ignore_index=True)
                data.to_csv(name)

                L_wc_s.append(L_wcv)
                R_wc_s.append(R_wcv)

                fig = plt.figure(figsize=(7, 3))
                axL = fig.add_subplot(121)
                axR = fig.add_subplot(122)
                axL.plot(Ls[0:L], L_wc_s, label="proposed", color="green", linewidth=3)
                axR.plot(Ls[0:L], R_wc_s, label="proposed", color="green", linewidth=3)

                axL.set_title("$L(\hat{G}, G)$", fontsize=20)
                axR.set_title("$R($" + r"$\rho$" + ", $\pi)$", fontsize=20)

                axL.set_xlabel("$L$", fontsize=20)
                axR.set_xlabel("$L$", fontsize=20)

                plt.tight_layout()
                fig.savefig("./experiment-6-3/syn/CLN2/N2.pdf")
                fig.clf()
