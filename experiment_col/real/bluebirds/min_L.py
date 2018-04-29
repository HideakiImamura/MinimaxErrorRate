import pandas as pd
import numpy as np
import sys

sys.path.append("/Users/forute/Documents/Academy/Resaech/Clustering_Worker")
now = "./experiment_col/real/dog/"

data = pd.read_csv(now + "data_n807m109K4.csv")
print(np.argmin(list(data['R'])[1:]))
print(list(data['L'])[1:][np.argmin(list(data['R'])[1:])])