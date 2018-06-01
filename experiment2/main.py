import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
import sys

os.chdir("../")
sys.path.append(os.getcwd())
import data.synthetic_data.CLN_data_generator as cdg
import data.synthetic_data.ZLN_data_generator as zdg
import models.Dawid_Skene as ds
import models.Majority_Voting as mv
import models.Worker_Clustering as wc

n = 1000
m = 100
K = 2
L = 10
rho = 0.3
epsilon = 1e-2


def experiment_cln(N, csv_name, pdf_name):
    xs = np.arange(m // 2 + 1) / m
    task_class = cdg.create_data(n, K, [rho, 1 - rho])
    task = np.hsplit(task_class, [1])[0].ravel()
    true_class = np.hsplit(task_class, [1])[1].ravel()
    g = np.array(sorted(task_class, key=lambda pair: pair[0]))[:, 1]

    data = pd.DataFrame(columns=["MV", "DS", "WC"])
    accuracies = {"MV": [],
                  "DS": [],
                  "WC": []}
    for i, x in enumerate(xs):
        task_worker_class = cdg.task_worker_label(m, K, task, true_class, N, [1 - x, 0.0, x])

        g_mv = mv.inference(task_worker_class, n, m, K)
        acc_mv = accuracy_score(g, g_mv)
        print("mv       = {}".format(acc_mv))
        accuracies["MV"].append(acc_mv)

        g_ds, [pi_ds, rho_ds] = ds.inference(task_worker_class, n, m, K, epsilon=epsilon)
        acc_ds = accuracy_score(g, g_ds)
        print("acc_ds   = {}".format(acc_ds))
        accuracies["DS"].append(acc_ds)

        g_wc, [pi_wc, rho_wc] = wc.inference(task_worker_class, n, m, K, L, epsilon=epsilon)
        acc_wc = accuracy_score(g, g_wc)
        print("acc_wc   = {}".format(acc_wc))
        accuracies["WC"].append(acc_wc)

        acc = pd.DataFrame([{'MV': acc_mv,
                             'DS': acc_ds,
                             'WC': acc_wc}])
        data = data.append(acc, ignore_index=True)
        data.to_csv(csv_name)
        plt.scatter(xs[0:i + 1], accuracies["MV"], label="MV", color="blue")
        plt.scatter(xs[0:i + 1], accuracies["DS"], label="DS", color="red")
        plt.scatter(xs[0:i + 1], accuracies["WC"], label="proposed", color="green")
        plt.xlabel("proportion of adversary")
        plt.ylabel("Accuracy")
        plt.legend(loc="upper right")
        plt.savefig(pdf_name)
        plt.clf()


def experiment_zln(N, csv_name, pdf_name):
    xs = np.arange(m // 2 + 1) / m
    task_class = cdg.create_data(n, K, [rho, 1 - rho])
    task = np.hsplit(task_class, [1])[0].ravel()
    true_class = np.hsplit(task_class, [1])[1].ravel()
    g = np.array(sorted(task_class, key=lambda pair: pair[0]))[:, 1]

    data = pd.DataFrame(columns=["MV", "DS", "WC"])
    accuracies = {"MV": [],
                  "DS": [],
                  "WC": []}
    for i, x in enumerate(xs):
        task_worker_class = zdg.task_worker_label(m, K, task, true_class, N, [1 - x, 0.0, x])

        g_mv = mv.inference(task_worker_class, n, m, K)
        acc_mv = accuracy_score(g, g_mv)
        print("mv       = {}".format(acc_mv))
        accuracies["MV"].append(acc_mv)

        g_ds, [pi_ds, rho_ds] = ds.inference(task_worker_class, n, m, K, epsilon=epsilon)
        acc_ds = accuracy_score(g, g_ds)
        print("acc_ds   = {}".format(acc_ds))
        accuracies["DS"].append(acc_ds)

        g_wc, [pi_wc, rho_wc] = wc.inference(task_worker_class, n, m, K, L, epsilon=epsilon)
        acc_wc = accuracy_score(g, g_wc)
        print("acc_wc   = {}".format(acc_wc))
        accuracies["WC"].append(acc_wc)

        acc = pd.DataFrame([{'MV': acc_mv,
                             'DS': acc_ds,
                             'WC': acc_wc}])
        data = data.append(acc, ignore_index=True)
        data.to_csv(csv_name)
        plt.scatter(xs[0:i + 1], accuracies["MV"], label="MV", color="blue")
        plt.scatter(xs[0:i + 1], accuracies["DS"], label="DS", color="red")
        plt.scatter(xs[0:i + 1], accuracies["WC"], label="proposed", color="green")
        plt.xlabel("proportion of adversary")
        plt.ylabel("Accuracy")
        plt.legend(loc="upper right")
        plt.savefig(pdf_name)
        plt.clf()


########## CLN: N = 2 ##########
N = 2
csv_name = os.getcwd() + "/experiment1/result/CLN2.csv"
pdf_name = os.getcwd() + "/experiment1/result/CLN2.pdf"

experiment_cln(N, csv_name, pdf_name)

########## CLN: N = 10 ##########
N = 10
csv_name = os.getcwd() + "/experiment1/result/CLN10.csv"
pdf_name = os.getcwd() + "/experiment1/result/CLN10.pdf"

experiment_cln(N, csv_name, pdf_name)

########## CLN: N = 100 ##########
N = 100
csv_name = os.getcwd() + "/experiment1/result/CLN100.csv"
pdf_name = os.getcwd() + "/experiment1/result/CLN100.pdf"

experiment_cln(N, csv_name, pdf_name)

########## ZLN: N = 1...60 ##########
N = 100
csv_name = os.getcwd() + "/experiment1/result/ZLN1_60.csv"
pdf_name = os.getcwd() + "/experiment1/result/ZLN1_60.pdf"

experiment_zln(N, csv_name, pdf_name)



