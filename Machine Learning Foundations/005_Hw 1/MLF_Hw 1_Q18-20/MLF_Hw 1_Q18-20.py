#pocket algorithm

#匯入 資料庫
import numpy as np
import pandas as pd

#定義 loadData 副程式
#  將 指定資料(filename) 載入程式
def loadData(filename):
    data = pd.read_csv(filename,sep='\\s+', header = None)
    data = data.as_matrix()
    col,row = data.shape
    X = np.c_[np.ones((col, 1)), data[:, 0: row-1]]
    Y = data[:, row-1:row]
    return X, Y

#Q18-Q20導入資料
X, Y = loadData('MLF_Hw 1_Q18_train.dat')
X_test, Y_test = loadData('MLF_Hw 1_Q18_test.dat')
col, row = X.shape
theta = np.zeros((row, 1))

#在定義pocket演算法之前，要先定義用來測量pocket演算法的錯誤率函數
def mistake(yhat, y):
    row, col = y.shape
    return np.sum(yhat != y)/row

#pocket演算法
def pocket(X, Y, theta, iternum, eta = 1):
    y_hat = np.sign(X.dot(theta))
    y_hat[np.where(y_hat == 0)] = -1
    err_old = mistake(y_hat, Y)
    theta = np.zeros(theta.shape)
    for t in range(iternum):
        y_miss_index = np.where(y_hat != Y)[0]
        if not y_miss_index.any():
            break
        pos = y_miss_index[np.random.permutation(len(y_miss_index))[0]]
        theta += eta * Y[pos, 0] * X[pos:pos + 1, :].T
        y_hat = np.sign(X.dot(theta))
        y_hat[np.where(y_hat == 0)] = -1
        err_now = mistake(y_hat, Y)
        if err_now < err_old:
            theta_best = theta.copy()
            err_old = err_now
    return theta_best, theta

# Q18
total = 0
for i in range(2000):
    theta = np.zeros((row, 1))
    randpos = np.random.permutation(col)
    X_rnd = X[randpos, :]
    Y_rnd = Y[randpos, 0:1]
    theta, theta_bad = pocket(X_rnd, Y_rnd, theta, 50)
    y_hat = np.sign(X_test.dot(theta))
    y_hat[np.where(y_hat == 0)] = -1
    err = mistake(y_hat, Y_test)
    total += err
print('Q18：',total/2000)

# Q19
total = 0
for i in range(2000):
    theta = np.zeros((row, 1))
    randpos = np.random.permutation(col)
    X_rnd = X[randpos, :]
    Y_rnd = Y[randpos, 0:1]
    theta, theta_bad = pocket(X_rnd, Y_rnd, theta, 50)
    y_hat = np.sign(X_test.dot(theta_bad))
    y_hat[np.where(y_hat == 0)] = -1
    err = mistake(y_hat, Y_test)
    total += err
print('Q19：',total/2000)

# Q18
total = 0
for i in range(2000):
    theta = np.zeros((row, 1))
    randpos = np.random.permutation(col)
    X_rnd = X[randpos, :]
    Y_rnd = Y[randpos, 0:1]
    theta, theta_bad = pocket(X_rnd, Y_rnd, theta, 100)
    y_hat = np.sign(X_test.dot(theta))
    y_hat[np.where(y_hat == 0)] = -1
    err = mistake(y_hat, Y_test)
    total += err
print('Q20：',total/2000)
