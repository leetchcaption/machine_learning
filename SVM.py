# coding=utf-8
# svm 支持向量机 分类函数
import numpy as np
import pandas as pd
from sklearn import svm
# 交叉验证算法
from sklearn import cross_validation

data = pd.read_csv('D:\\workspace\\FileForder\\coursedata\\classification\\stock\\000777.csv',encoding='gbk',parse_dates=[0],index_col=0)
# 按照0列升序排序，并替换掉原有的值
data.sort_index(0,ascending=True,inplace=True)
# 选取150天的数据
dayfeature=150
# 每天选取5个特征数据
featurenum=5*dayfeature
x=np.zeros((data.shape[0]-dayfeature,featurenum+1))
y=np.zeros((data.shape[0]-dayfeature))

for i in range(0, data.shape[0] - dayfeature):
    x[i, 0:featurenum] = np.array(data[i:i + dayfeature] \
                                      [[u'收盘价', u'最高价', u'最低价', u'开盘价', u'成交量']]).reshape((1, featurenum))
    x[i, featurenum] = data.ix[i + dayfeature][u'开盘价']

for i in range(0, data.shape[0] - dayfeature):
    if data.ix[i + dayfeature][u'收盘价'] >= data.ix[i + dayfeature][u'开盘价']:
        y[i] = 1
    else:
        y[i] = 0

clf=svm.SVC(kernel='rbf')
result = []
for i in range(5):
    x_train, x_test, y_train, y_test = \
                cross_validation.train_test_split(x, y, test_size = 0.2)
    clf.fit(x_train, y_train)
    result.append(np.mean(y_test == clf.predict(x_test)))
print("svm classifier accuacy:")
print(result)