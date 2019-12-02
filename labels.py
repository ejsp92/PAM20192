import pandas as pd
import numpy as np 

# This program was used to convert the labels in the data to a number index

data = pd.read_csv('data/seg.test')
label = {'FOLIAGE': 0, 'PATH': 1, 'CEMENT': 2, 'GRASS': 3, 'WINDOW': 4, 'SKY': 5, 'BRICKFACE': 6}
data.index = [label[item] for item in data.index] 
index = data.index

np.savetxt("data/test_gt.csv", index, delimiter=",")