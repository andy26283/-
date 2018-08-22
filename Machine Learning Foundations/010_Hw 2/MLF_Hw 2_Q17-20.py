#匯入資料庫
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#定義 loadData 副程式
#  將 指定資料(filename) 載入程式
def loadData(filename):
    data = pd.read_csv(filename,sep='\\s+', header = None)
    data = data.as_matrix()
    col,row = data.shape
    X = np.c_[np.ones((col, 1)), data[:, 0: row-1]]
    Y = data[:, row-1:row]
    return X, Y

def generateData():
    x = np.random.uniform(-1, 1, 20)
    y = np.sign(x)
    y[y == 0] = -1
    prop = np.random.uniform(0, 1, 20)
    y[prop >= 0.8] *= -1
    return x,y

def decision_stump(X, Y):
    theta = np.sort(X)
    num = len(theta)
    Xtemp = np.tile(X, (num, 1))
    ttemp = np.tile(np.reshape(theta, (num, 1)), (1, num))
    ypred = np.sign(Xtemp - ttemp)
    ypred[ypred == 0] = -1
    err = np.sum(ypred != Y, axis=1)
    if np.min(err) <= num-np.max(err):
        return 1, theta[np.argmin(err)], np.min(err)/num
    else:
        return -1, theta[np.argmax(err)], (num-np.max(err))/num

# 多维度决策树桩算法
def decision_stump_multi(X, Y):
    row, col = X.shape
    err = np.zeros((col,)); s = np.zeros((col,)); theta = np.zeros((col,))
    for i in range(col):
        s[i], theta[i], err[i] = decision_stump(X[:, i], Y[:, 0])
    pos = np.argmin(err)
    return pos, s[pos], theta[pos], err[pos]

# Q17和Q18
totalin = 0; totalout = 0
for i in range(5000):
    X, Y = generateData()
    theta = np.sort(X)
    s, theta, errin = decision_stump(X, Y)
    errout = 0.5+0.3*s*(math.fabs(theta)-1)
    totalin += errin
    totalout += errout
print('训练集平均误差: ', totalin/5000)
print('测试集平均误差: ', totalout/5000)

# Q19和Q20
X, Y = loadData('MLF_Hw 2_train.dat')
Xtest, Ytest = loadData('MLF_Hw 2_test.dat')
pos, s, theta, err = decision_stump_multi(X, Y)
print('训练集误差: ', err)
ypred = s*np.sign(Xtest[:, pos]-theta)
ypred[ypred == 0] = -1
row, col = Ytest.shape
errout = np.sum(ypred != Ytest.reshape(row,))/len(ypred)
print('测试集误差: ', errout)
