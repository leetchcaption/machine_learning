# coding=utf-8
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 加载鸢尾花数据集
data = load_iris()
y = data.target
X = data.data
pca = PCA(n_components=2)
reduced_x = pca.fit_transform(X)

# 按照类别对降维后的数据进行保存
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_x)):
    if y[i] == 0 :
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])

plt.scatter(red_x,red_y,c='r',marker='.')
plt.scatter(blue_x,blue_y,c='b',marker='.')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()