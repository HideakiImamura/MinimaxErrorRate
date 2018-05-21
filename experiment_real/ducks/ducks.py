import numpy as np
import yaml
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
sys.path.append("/Users/forute/Documents/Academy/Resaech/Robustness_of_Dawid-Skene_Model/experiment-6-2")
import model.Majority_Voting as mv
import model.Dawid_Skene as ds
import model.Worker_Clusterring_with_Dirichlet_Prior as evi
import matplotlib.pyplot as plt


f = open("./ducks/classes.yaml", "r")
task_label = yaml.load(f)
task = np.array([], dtype=int)
g = np.array([], dtype=int)
for i, k in task_label["labels"].items():
    task = np.append(task, [i])
    if int(k) == 0 or int(k) == 2:
        kk = 1
    else:
        kk = 0
    g = np.append(g, kk)
f.close()

task_worker_label = np.array([], dtype=int).reshape((0, 3))
f = open("./ducks/labels.txt", "r")
lines = f.readlines()
f.close()
for line in lines[1:]:
    i, j, k = [int(x) for x in line.split(" ")]
    task_worker_label = np.concatenate((task_worker_label, [[i, j, k]]))

n = 240
m = 53
K = 2
epsilon = 1e-2

g_mv = mv.MV(task_worker_label, n, m, K)
acc_mv = accuracy_score(g, g_mv)
print(acc_mv)

g_ds, _ = ds.DS_elbo_debug(task_worker_label, n, m, K, epsilon=epsilon)
acc_ds = accuracy_score(g, g_ds)
print(acc_ds)

L_list = np.arange(1, m)
acc_evi_list = []
for L in L_list:
    g_evi, _, _ = evi.EVI(task_worker_label, n, m, K, L, epsilon=epsilon)
    acc_evi = accuracy_score(g, g_evi)
    print(L, " ", acc_evi)
    acc_evi_list.append(acc_evi)

data = pd.DataFrame({'MV': acc_mv,
                     'DS': acc_ds,
                     'proposed': acc_evi_list})
data.to_csv("./ducks/L1_m.csv")

plt.plot(L_list, acc_mv * np.ones(len(L_list)), label="MV")
plt.plot(L_list, acc_ds * np.ones(len(L_list)), label="DS")
plt.plot(L_list, acc_evi_list, label="proposed")
plt.xlabel("L")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig("./ducks/L1_m.png")
