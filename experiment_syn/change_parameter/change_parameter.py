import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import sys

sys.path.append("/Users/forute/Documents/Academy/Resaech/Robustness_of_Dawid-Skene_Model")
import experiment_syn.change_parameter.Create_Synthetic_Data as csd
import model.Dawid_Skene as ds
import model.Majority_Voting as mv
import model.Worker_Clustering as wcv
import model.Worker_Clusterring_with_Dirichlet_Prior as wcv_wp

n = 100
ms = [10, 50, 100]
K = 2
Ns = [1, 3, 5, 7, 10]
# L = np.arange(m) + 1
cs = np.arange(0.1, 0.6, 0.1)
xs = np.arange(0.0, 1.1, 0.1)
'''
ms = [50]
K = 2
Ns = [7]
cs = [0.4]
xs = [0.2]
'''
epsilon = 1e-2

for m in ms:
    for N in Ns:
        for c in cs:
            for x in xs:
                task_class = csd.create_dataset(n, K, [c, 1 - c])
                task = np.hsplit(task_class, [1])[0].ravel()
                true_class = np.hsplit(task_class, [1])[1].ravel()
                worker = csd.create_crowd(m, K, [1 - x, 0.0, x])
                task_worker_class = csd.labeling_by_crowd(m, K, task, true_class, worker, N)

                g = np.array(sorted(task_class, key=lambda pair: pair[0]))[:, 1]

                g_mv = mv.MV(task_worker_class, n, m, K)
                acc_mv = accuracy_score(g, g_mv)
                print("mv = ", acc_mv)

                g_ds, _ = ds.DS_elbo_debug(task_worker_class, n, m, K, epsilon=epsilon)
                acc_ds = accuracy_score(g, g_ds)
                print("ds = ", acc_ds)

                acc_wcv_list = []
                acc_wcv_wp_list = []
                Ls = (np.arange(m) + 1)[:100]
                for L in Ls:
                    print("L: ", L)
                    g_wcv, _, _ = wcv.wcv(task_worker_class, n, m, K, L, epsilon=epsilon)
                    acc_wcv = accuracy_score(g, g_wcv)
                    acc_wcv_list.append(acc_wcv)
                    print("     wcv    = ", acc_wcv)
                    g_wcv_wp, _, _ = wcv_wp.EVI(task_worker_class, n, m, K, L, epsilon=epsilon)
                    acc_wcv_wp = accuracy_score(g, g_wcv_wp)
                    acc_wcv_wp_list.append(acc_wcv_wp)
                    print("     wcv_wp = ", acc_wcv_wp)

                    data = pd.DataFrame({'MV': acc_mv,
                                         'DS': acc_ds,
                                         'proposed1(wcv)': acc_wcv_list,
                                         'proposed2(wcv_wp)': acc_wcv_wp_list})
                    data.to_csv("./experiment-6-1/change_parameter/data_" +
                                "n" + str(n) +
                                "m" + str(m) +
                                "K" + str(K) +
                                "N" + str(N) +
                                "c" + str(c) +
                                "x" + str(x) + ".csv")

                    plt.plot(np.arange(L) + 1, acc_mv * np.ones(L), label="MV", color="blue")
                    plt.plot(np.arange(L) + 1, acc_ds * np.ones(L), label="DS", color="red")
                    plt.plot(np.arange(L) + 1, acc_wcv_list, label="proposed1", color="green")
                    plt.plot(np.arange(L) + 1, acc_wcv_wp_list, label="proposed2", color="yellow")
                    plt.xlabel("L")
                    plt.ylabel("Accuracy")
                    plt.legend(loc="lower right")
                    plt.ylim(min(acc_mv, acc_ds, min(acc_wcv_list), min(acc_wcv_wp_list)) - 0.05,
                             max(acc_mv, acc_ds, max(acc_wcv_list), max(acc_wcv_wp_list)) + 0.05)
                    plt.savefig("./experiment-6-1/change_parameter/graph_" +
                                "n" + str(n) +
                                "m" + str(m) +
                                "K" + str(K) +
                                "N" + str(N) +
                                "c" + str(c) +
                                "x" + str(x) + ".png")
                    plt.clf()
