# coding=utf-8
# 岭回归
import numpy as np
#加载岭回归方法
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
# 用于创建多项式特征
from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model

# 读取数据集
datasets_X = []
datasets_Y = []
fr = open('D:/workspace/FileForder/coursedata/regression/prices.txt', 'r')
lines = fr.readlines()
for line in lines:
    items = line.strip().split(',')
    datasets_X.append(int(items[0]))
    datasets_Y.append(int(items[1]))

length = len(datasets_X)
datasets_X = np.array(datasets_X).reshape([length, 1])
datasets_Y = np.array(datasets_Y)

minX = min(datasets_X)
maxX = max(datasets_X)
X = np.arange(minX, maxX).reshape([-1, 1])

poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(datasets_X)
lin_reg_2 = linear_model.LinearRegression()
lin_reg_2.fit(X_poly, datasets_Y)

# 图像中显示
plt.scatter(datasets_X, datasets_Y, color='red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()