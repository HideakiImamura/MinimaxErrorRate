import numpy as np
import pandas as pd
import os


f = open(os.getcwd() + "/dog_truth.txt", 'r')
lines = f.readlines()
f.close()
task = np.array([], dtype=int)
g = np.array([], dtype=int)
for line in lines[0:]:
    i, k = [int(x) for x in line.split("\t")]
    ii = np.where(task == i)[0]
    if len(ii) == 0:
        task = np.append(task, [i])
        g = np.append(g, [k - 1])

task_worker_label = np.array([], dtype=int).reshape((0, 3))
worker = np.array([], dtype=int)
f = open(os.getcwd() + "/dog_crowd.txt", "r")
lines = f.readlines()
f.close()
for line in lines[0:]:
    i, j_, k = [int(x) for x in line.split("\t")]
    jj = np.where(worker == j_)[0]
    if len(jj) == 0:
        j = j_
        worker = np.append(worker, [j_])
    elif len(jj) == 1:
        j = jj[0]
    else:
        raise ValueError("Worker is not good")
    task_worker_label = np.concatenate((task_worker_label, [[i - 1, j - 1, k - 1]]))

name = "dog"
os.chdir("../../../data/real_data")
pd.DataFrame(g)\
    .rename(columns={0: "ground_truth"})\
    .to_csv(os.getcwd() + "/" + name + "_ground_truth.csv")
pd.DataFrame(task_worker_label)\
    .rename(columns={0: "task_id", 1: "worker_id", 2: "given_label"})\
    .to_csv(os.getcwd() + "/" + name + "_crowd_label.csv")