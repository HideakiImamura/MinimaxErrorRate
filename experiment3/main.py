import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
import sys

os.chdir("../")
sys.path.append(os.getcwd())
import models.Dawid_Skene as ds
import models.Majority_Voting as mv
import models.Worker_Clustering as wc

epsilon = 1e-2


def experiment_real(crowd_label_name, ground_truth_name, csv_name, pdf_name):
    task_worker_class = np.array(
        pd.read_csv(crowd_label_name)[['task_id',
                                       'worker_id',
                                       'given_label']].values.tolist()
    )
    n, m, K = task_worker_class[0]
    task_worker_class = task_worker_class[1:]

    g = np.array(
        pd.read_csv(ground_truth_name)[['ground_truth']].values.tolist()
    ).T[0]

    rangeL = np.arange(1, m + 1)

    accuracies = {}

    g_mv = mv.inference(task_worker_class, n, m, K)
    acc_mv = accuracy_score(g, g_mv)
    print("mv       = {}".format(acc_mv))
    accuracies["MV"] = acc_mv

    g_ds, [pi_ds, rho_ds] = ds.inference(task_worker_class, n, m, K, epsilon=epsilon)
    acc_ds = accuracy_score(g, g_ds)
    print("ds       = {}".format(acc_ds))
    accuracies["DS"] = acc_ds

    acc_wcs = []
    for i, L in enumerate(rangeL):
        g_wc, [pi_wc, rho_wc] = wc.inference(task_worker_class, n, m, K, L, epsilon=epsilon, counter=10)
        acc_wc = accuracy_score(g, g_wc)
        print("wc L={}  = {}".format(L, acc_wc))
        accuracies["WC" + str(L)] = acc_wc
        acc_wcs.append(acc_wc)

        pd.DataFrame(accuracies, index=['accuracy']).to_csv(csv_name)

        plt.plot(rangeL[0:i + 1], accuracies["MV"] * np.ones(i + 1), label="MV", color="blue")
        plt.plot(rangeL[0:i + 1], accuracies["DS"] * np.ones(i + 1), label="DS", color="red")
        plt.plot(rangeL[0:i + 1], acc_wcs, label="proposed", color="green")
        plt.xlabel("Maximum number of clusters ($L$)")
        plt.ylabel("Accuracy")
        plt.legend(loc="upper right")
        plt.savefig(pdf_name)
        plt.clf()


########## bird ##########
crowd_label_name = os.getcwd() + "/data/real_data/bird_crowd_label.csv"
ground_truth_name = os.getcwd() + "/data/real_data/bird_ground_truth.csv"
csv_name = os.getcwd() + "/experiment2/result/bird.csv"
pdf_name = os.getcwd() + "/experiment2/result/bird.pdf"

experiment_real(crowd_label_name, ground_truth_name, csv_name, pdf_name)

########## dog ##########
crowd_label_name = os.getcwd() + "/data/real_data/dog_crowd_label.csv"
ground_truth_name = os.getcwd() + "/data/real_data/dog_ground_truth.csv"
csv_name = os.getcwd() + "/experiment2/result/dog.csv"
pdf_name = os.getcwd() + "/experiment2/result/dog.pdf"

experiment_real(crowd_label_name, ground_truth_name, csv_name, pdf_name)

########## rte ##########
crowd_label_name = os.getcwd() + "/data/real_data/rte_crowd_label.csv"
ground_truth_name = os.getcwd() + "/data/real_data/rte_ground_truth.csv"
csv_name = os.getcwd() + "/experiment2/result/rte.csv"
pdf_name = os.getcwd() + "/experiment2/result/rte.pdf"

experiment_real(crowd_label_name, ground_truth_name, csv_name, pdf_name)



