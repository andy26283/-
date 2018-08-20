import numpy as np
import pandas as pd

def loadData(filename):
    ata = pd.read_csv(filename,sep='\\s+', header = None)
    data = data.as_matrix()
    col,row = data.shape
    X = np.c_[np.ones((col, 1)), data[:, 0: row-1]]
    return X, Y

# Q15-Q17導入資料
X, Y = loadData('hw1_15_train.dat')
col, row = X.shape
theta = np.zeros((row, 1))
print('X的前五项：\\n',X[0:5, :])
print('Y的前五项: \\n',Y[0:5,:].T)
