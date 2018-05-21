import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import sys

sys.path.append("/Users/forute/Documents/Academy/Resaech/Clustering_Worker")
import experiment_syn.change_parameter.Create_Synthetic_Data as csd
import model.Dawid_Skene as ds
import model.Majority_Voting as mv
import model.Worker_Clustering as wcv
import model.Worker_Clusterring_with_Dirichlet_Prior as wcv_wp

n = 1000
ms = [100]
K = 2
Ns = [2, 5, 10, 100]
# L = np.arange(m) + 1
cs = [0.3]
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
            Ls = [20]
            for L in Ls:
                task_class = csd.create_dataset(n, K, [c, 1 - c])
                task = np.hsplit(task_class, [1])[0].ravel()
                true_class = np.hsplit(task_class, [1])[1].ravel()
                g = np.array(sorted(task_class, key=lambda pair: pair[0]))[:, 1]

                data = pd.DataFrame(columns=["MV", "DS", "proposed1(wcv)"])
                acc_mv_list = []
                acc_ds_list = []
                acc_wcv_list = []
                xs = np.arange(m // 2 + 1) / m
                for i, x in enumerate(xs):
                    worker = csd.create_crowd(m, K, [1 - x, 0.0, x])
                    task_worker_class = csd.labeling_by_crowd(m, K, task, true_class, worker, N)

                    g_mv = mv.MV(task_worker_class, n, m, K)
                    acc_mv = accuracy_score(g, g_mv)
                    print("mv = ", acc_mv)
                    acc_mv_list.append(acc_mv)

                    g_ds, _ = ds.DS_elbo_debug(task_worker_class, n, m, K, epsilon=epsilon)
                    acc_ds = accuracy_score(g, g_ds)
                    print("ds = ", acc_ds)
                    acc_ds_list.append(acc_ds)

                    g_wcv, _, _ = wcv.wcv(task_worker_class, n, m, K, L, epsilon=epsilon)
                    acc_wcv = accuracy_score(g, g_wcv)
                    print("wcv    = ", acc_wcv)
                    acc_wcv_list.append(acc_wcv)

                    # g_wcv_wp, _, _ = wcv_wp.EVI(task_worker_class, n, m, K, L, epsilon=epsilon)
                    # acc_wcv_wp = accuracy_score(g, g_wcv_wp)
                    # print("     wcv_wp = ", acc_wcv_wp)

                    acc = pd.DataFrame([{'MV': acc_mv,
                                         'DS': acc_ds,
                                         'proposed1(wcv)': acc_wcv}])
                                         # 'proposed2(wcv_wp)': acc_wcv_wp})
                    data = data.append(acc)
                    data.to_csv("./experiment-6-1/adversarial/for_paper/data_" +
                                "n" + str(n) +
                                "m" + str(m) +
                                "K" + str(K) +
                                "N" + str(N) +
                                "c" + str(c) +
                                "L" + str(L) + ".csv")

                    plt.scatter(xs[0:i + 1], acc_mv_list, label="MV", color="blue")
                    plt.scatter(xs[0:i + 1], acc_ds_list, label="DS", color="red")
                    plt.scatter(xs[0:i + 1], acc_wcv_list, label="proposed1", color="green")
                    # plt.scatter(xs[0:i + 1], acc_wcv_wp_list, label="proposed2", color="yellow")
                    plt.xlabel("proportion of adversary")
                    plt.ylabel("Accuracy")
                    plt.legend(loc="upper right")
                    plt.savefig("./experiment-6-1/adversarial/for_paper/graph_" +
                                "n" + str(n) +
                                "m" + str(m) +
                                "K" + str(K) +
                                "N" + str(N) +
                                "c" + str(c) +
                                "L" + str(L) + ".png")
                    plt.clf()
