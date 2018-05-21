import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
from scipy.optimize import newton
import sys

sys.path.append("/Users/forute/Documents/Academy/Resaech/Clustering_Worker")
import experiment_syn.change_parameter.Create_Synthetic_Data as csd
import model.Dawid_Skene as ds
import model.Majority_Voting as mv
import model.Worker_Clustering as wcv


def loss_lower_bound(n, pi, rho):
    delta = 1e-100
    m = pi.shape[0]
    K = pi.shape[1]
    kl = entropy(pk=rho)
    for j in range(m):
        for g in range(K):
            for g_ in range(K):
                kl -= rho[g] * rho[g_] * entropy(pk=(pi[j, g] + delta), qk=(pi[j, g_] + delta))
    # kl *= n
    x0 = (1 - 1 / (K ** n)) / 2
    print(kl)
    return kl
    '''
    if kl <= 0:
        lb = 0
    else:
        lb = newton(func=phi(n, K, kl), x0=x0, fprime=phi_prime(K), maxiter=int(1e+7), tol=1e-5) / (2 * n)
    return lb
    '''

def phi(n, K, kl):
    delta = 1e-100
    return lambda x: - x * np.log(x + delta) - (1 - x) * np.log(1 - x + delta) + n * x * np.log(K) - kl


def phi_prime(K):
    delta = 1e-100
    return lambda x: np.log((1 / (x + delta) - 1) * (K - 1))


n = 100
m = 50
K = 2
N = 3
c = 0.3
L = 10
epsilon = 1e-2

xs = np.arange(m + 1) / m

task_class = csd.create_dataset(n, K, [c, 1 - c])
task = np.hsplit(task_class, [1])[0].ravel()
true_class = np.hsplit(task_class, [1])[1].ravel()
g = np.array(sorted(task_class, key=lambda pair: pair[0]))[:, 1]

data = pd.DataFrame(columns=["MV", "DS", "proposed1(wcv)"])
# acc_mv_list = []
acc_ds_list = []
acc_wcv_list = []
lb_ds_list = []
lb_wcv_list = []
for i, x in enumerate(xs):
    worker = csd.create_crowd(m, K, [1 - x, 0.0, x])
    task_worker_class = csd.labeling_by_crowd(m, K, task, true_class, worker, N)

    #g_mv = mv.MV(task_worker_class, n, m, K)
    #acc_mv = accuracy_score(g, g_mv)
    #print("mv = ", acc_mv)
    #acc_mv_list.append(acc_mv)

    g_ds, [pi_ds, rho_ds] = ds.DS_elbo_debug(task_worker_class, n, m, K, epsilon=epsilon)
    acc_ds = accuracy_score(g, g_ds)
    lb_ds = loss_lower_bound(n, pi_ds, rho_ds)
    print("(acc_ds, lb_ds)   = ({}, {})".format(acc_ds, lb_ds))
    acc_ds_list.append(acc_ds)
    lb_ds_list.append(lb_ds)

    g_wcv, [pi_wcv, rho_wcv], _ = wcv.wcv(task_worker_class, n, m, K, L, epsilon=epsilon)
    acc_wcv = accuracy_score(g, g_wcv)
    lb_wcv = loss_lower_bound(n, pi_wcv, rho_wcv)
    print("(acc_wcv, lb_wcv) = ({}, {})".format(acc_wcv, lb_wcv))
    acc_wcv_list.append(acc_wcv)
    lb_wcv_list.append(lb_wcv)

    # g_wcv_wp, _, _ = wcv_wp.EVI(task_worker_class, n, m, K, L, epsilon=epsilon)
    # acc_wcv_wp = accuracy_score(g, g_wcv_wp)
    # print("     wcv_wp = ", acc_wcv_wp)

    acc = pd.DataFrame([{#'MV': acc_mv,
                         'DS': acc_ds,
                         'proposed1(wcv)': acc_wcv,
                         'DS_lb': lb_ds,
                         'proposed1(wcv)_lb': lb_wcv}])
    # 'proposed2(wcv_wp)': acc_wcv_wp})
    data = data.append(acc)
    data.to_csv("./experiment-6-1/bound/data_" +
                "n" + str(n) +
                "m" + str(m) +
                "K" + str(K) +
                "N" + str(N) +
                "c" + str(c) +
                "L" + str(L) + ".csv")

    #plt.plot(xs[0:i + 1], acc_mv_list, label="MV", color="blue")
    plt.scatter(xs[0:i + 1], lb_ds_list, label="DS", color="red")
    plt.scatter(xs[0:i + 1], lb_wcv_list, label="proposed1", color="green")
    # plt.scatter(xs[0:i + 1], acc_wcv_wp_list, label="proposed2", color="yellow")
    plt.xlabel("proportion of adversary")
    plt.ylabel("Error lower bound")
    plt.legend(loc="lower right")
    plt.savefig("./experiment-6-1/bound/graph_" +
                "n" + str(n) +
                "m" + str(m) +
                "K" + str(K) +
                "N" + str(N) +
                "c" + str(c) +
                "L" + str(L) + ".png")
    plt.clf()
