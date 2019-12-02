from sklearn.metrics import adjusted_rand_score
import numpy as np
import pandas as pd
import sys

# Used to calculate the rand score from a result folder

if len(sys.argv) is not 2:
    print("usage: python defuzzyfy.py [dir name]")
    sys.exit()

df = pd.read_csv(sys.argv[1] + "/best_u.csv", header=None)

crisp = df.idxmax().values

ground_truth = np.genfromtxt('data/test_gt.csv', delimiter=',')

print('rand:    ' + str(adjusted_rand_score(ground_truth, crisp)))