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

#Q15-Q17導入資料
X, Y = loadData('MLF_Hw 1_Q15_train.dat')
col, row = X.shape
theta = np.zeros((row, 1))

#測試資料是否有進入程式
print('X的前五项：\n',X[0:5, :],'\n')
print('Y的前五项：\n',Y[0:5,:].T,'\n')
print('theta：\n',theta,'\n')

#二維PLA演算法
def perceptron(X, Y, theta, eta=1):
    remake_num = 0 #調整次數
    lastest_remake_point = 0 #上次調整時最後的錯誤點
    while(True): #無限遞迴
        y_hat = np.sign(X.dot(theta)) #將每個點標記出正確(1)或錯誤(0)
        y_hat[np.where(y_hat == 0)] = -1 #將錯誤代碼改成-1
        y_miss_index = np.where(y_hat != Y)[0] #將錯誤的第一個點紀錄下來，之後作為提高調整效率的基石
        if not y_miss_index.any(): #如果完全沒有錯誤點 → 找到最好表現的線，結束遞迴
            break
        if not y_miss_index[y_miss_index >= lastest_remake_point].any(): #若新一輪的錯誤點比前一輪的錯誤點還要的前面，重設前一輪的錯誤點
            lastest_remake_point = 0
        lastest_remake_point_orifinal = y_miss_index[y_miss_index >= lastest_remake_point][0] #紀錄之前錯誤點，之後作為調整的依據
        lastest_remake_point = lastest_remake_point_orifinal
        theta += eta*Y[lastest_remake_point_orifinal, 0]*X[lastest_remake_point_orifinal:lastest_remake_point_orifinal + 1, :].T #調整
        remake_num += 1 #調整次數+1
    return theta,remake_num #回傳參數

theta, num = perceptron(X,Y,theta)
print(num)
