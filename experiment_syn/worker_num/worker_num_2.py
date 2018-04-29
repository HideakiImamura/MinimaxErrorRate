import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import sys
import os

sys.path.append("/Users/forute/Documents/Academy/Resaech/Clustering_Worker")
import experiment_syn.worker_num.create_worker_labeling_number_dataset as csd
import model.Dawid_Skene as ds
import model.Majority_Voting as mv
import model.proposed_wcv as wcv

n = 1000
ms = [100]
K = 2
Ns = [2]
cs = [0.3]
L = 10
epsilon = 1e-2


for m in ms:
    for N in Ns:
        for c in cs:
            xs = np.arange(m // 2 + 1) / m
            task_class = csd.create_data(n, K, [c, 1 - c])
            task = np.hsplit(task_class, [1])[0].ravel()
            true_class = np.hsplit(task_class, [1])[1].ravel()
            g = np.array(sorted(task_class, key=lambda pair: pair[0]))[:, 1]

            name = "./experiment_syn/worker_num/for_paper/data_" + \
                            "n" + str(n) + \
                            "m" + str(m) + \
                            "K" + str(K) + \
                            "N" + str(N) + \
                            "c" + str(c) + \
                            "L" + str(L) + ".csv"
            if os.path.exists(name):
                data = pd.read_csv(name)[["MV", "DS", "proposed1(wcv)"]]
            else:
                data = pd.DataFrame(columns=["MV", "DS", "proposed1(wcv)"])
            acc_mv_list = list(data["MV"])
            acc_ds_list = list(data["DS"])
            acc_wcv_list = list(data["proposed1(wcv)"])
            xss = np.arange(len(data.index), m // 2 + 1) / m
            LLL = len(data.index)
            for i, x in enumerate(xss):
                task_worker_class = csd.task_worker_label(m, K, task, true_class, N, [1 - x, 0.0, x])

                g_mv = mv.MV(task_worker_class, n, m, K)
                acc_mv = accuracy_score(g, g_mv)
                print("mv       = {}".format(acc_mv))
                acc_mv_list.append(acc_mv)

                g_ds, [pi_ds, rho_ds] = ds.DS_elbo_debug(task_worker_class, n, m, K, epsilon=epsilon)
                acc_ds = accuracy_score(g, g_ds)
                print("acc_ds   = {}".format(acc_ds))
                acc_ds_list.append(acc_ds)

                g_wcv, [pi_wcv, rho_wcv], _ = wcv.wcv(task_worker_class, n, m, K, L, epsilon=epsilon)
                acc_wcv = accuracy_score(g, g_wcv)
                print("acc_wcv  = {}".format(acc_wcv))
                acc_wcv_list.append(acc_wcv)

                # g_wcv_wp, _, _ = wcv_wp.EVI(task_worker_class, n, m, K, L, epsilon=epsilon)
                # acc_wcv_wp = accuracy_score(g, g_wcv_wp)
                # print("     wcv_wp = ", acc_wcv_wp)

                acc = pd.DataFrame([{'MV': acc_mv,
                                     'DS': acc_ds,
                                     'proposed1(wcv)': acc_wcv}])
                # 'proposed2(wcv_wp)': acc_wcv_wp})
                data = data.append(acc, ignore_index=True)
                data.to_csv("./experiment_syn/worker_num/for_paper/data_" +
                            "n" + str(n) +
                            "m" + str(m) +
                            "K" + str(K) +
                            "N" + str(N) +
                            "c" + str(c) +
                            "L" + str(L) + ".csv")
                print(len(acc_mv_list))
                print(i + LLL)
                plt.scatter(xs[0:i + 1 + LLL], acc_mv_list, label="MV", color="blue")
                plt.scatter(xs[0:i + 1 + LLL], acc_ds_list, label="DS", color="red")
                plt.scatter(xs[0:i + 1 + LLL], acc_wcv_list, label="proposed", color="green")
                # plt.scatter(xs[0:i + 1], acc_wcv_wp_list, label="proposed2", color="yellow")
                plt.xlabel("proportion of adversary")
                plt.ylabel("Accuracy")
                plt.legend(loc="upper right")
                plt.savefig("./experiment_syn/worker_num/for_paper/graph_" +
                            "n" + str(n) +
                            "m" + str(m) +
                            "K" + str(K) +
                            "N" + str(N) +
                            "c" + str(c) +
                            "L" + str(L) + ".pdf")
                plt.clf()
