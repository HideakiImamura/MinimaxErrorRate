import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import sys
import os

sys.path.append("/Users/forute/Documents/Academy/Resaech/Clustering_Worker")
import experiment_syn.model_selection.create_worker_labeling_number_dataset as csd_c
import experiment_syn.model_selection.create_power_law_dataset as csd_p
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
m = 100
K = 2
N = 100
c = 0.3
epsilon = 1e-2
ad = 0.2
Ls = np.arange(1, 101)

task_class = csd_p.create_data(n, K, [c, 1 - c])
task = np.hsplit(task_class, [1])[0].ravel()
true_class = np.hsplit(task_class, [1])[1].ravel()
g = np.array(sorted(task_class, key=lambda pair: pair[0]))[:, 1]

name = "./experiment_syn/model_selection/data_" + \
       "n" + str(n) + \
       "m" + str(m) + \
       "K" + str(K) + \
       "N" + str(N) + \
       "c" + str(c) + \
       "ad" + str(ad)
if os.path.exists(name):
    data = pd.read_csv(name + "csv")[["WC"]]
else:
    data = pd.DataFrame(columns=["WC"])

task_worker_class = csd_p.task_worker_label(m, K, task, true_class, N, [1 - ad, 0.0, ad])

g_ds, [pi_ds, rho_ds] = ds.DS_elbo_debug(task_worker_class, n, m, K, epsilon=epsilon)
R_ds = R(rho_ds, pi_ds)
print("R_ds      = {}".format(R_ds))

R_wc_list = []
for L in Ls:
    g_wcv, [pi_wcv, rho_wcv], _ = wcv.wcv(task_worker_class, n, m, K, L, epsilon=epsilon)
    R_wc = R(rho_wcv, pi_wcv)
    R_wc_list.append(R_wc)
    print("R_wc      = {}".format(R_wc))

    acc = pd.DataFrame([{"WC": R_wc}])
    data = data.append(acc)
    data.to_csv(name + "csv")

    plt.plot(Ls[0:L], R_ds * np.ones(L), label="DS", color="red")
    plt.plot(Ls[0:L], R_wc_list, label="proposed", color="green")
    # plt.scatter(xs[0:i + 1], acc_wcv_wp_list, label="proposed2", color="yellow")
    plt.xlabel("the number of cluster L")
    plt.ylabel("R")
    plt.legend(loc="upper right")
    plt.savefig(name + ".pdf")
    plt.clf()
