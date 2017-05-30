# coding=utf-8
import numpy as np
import pandas as pd

# 从sklearn导入预处理模块
from sklearn.preprocessing import Imputer
#导入自动生成训练集和测试集的模块
from sklearn.model_selection import train_test_split
#导入预测结果评估模块
from sklearn.metrics import classification_report
# 导入KNN分类器
from sklearn.neighbors import KNeighborsClassifier

# x = [[0],[1],[2],[3]]
# y = [0,0,1,1]
# neighbor = KNeighborsClassifier(n_neighbors=3)
# neighbor.fit(x,y)
# print(neighbor.predict([2.3]))


def load_dataset(feature_paths,label_paths):
    """
    读取特征数据和标签数据
    :param feature_paths:
    :param label_paths:
    :return:
    """
    feature = np.ndarray(shape=(0,41))
    for file in feature_paths:
        # 逗号分隔符，问号作为缺省值，文件中不包含表头
        df = pd.read_table(file,delimiter=',',na_values='?',header=None)
        # 使用平均值补全缺省值
        imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
        imp.fit(df)
        # 生成预处理结果
        df= imp.transform(df)
        # 将读入的数据合并到特征集合中
        feature = np.concatenate((feature,df))

    label = np.ndarray(shape=(0, 1))
    for file in label_paths:
        df = pd.read_table(file,header=None)
        label = np.concatenate((label,df))
    # 将标签规整为一维向量
    label = np.ravel(label)

    return feature,label

#设置特征数据路径
feature_paths = ['D:\\workspace\\FileForder\\coursedata\\classification\\A.feature',
                     'D:\\workspace\\FileForder\\coursedata\\classification\\B.feature',
                     'D:\\workspace\\FileForder\\coursedata\\classification\\C.feature',
                     'D:\\workspace\\FileForder\\coursedata\\classification\\D.feature',
                     'D:\\workspace\\FileForder\\coursedata\\classification\\E.feature']
label_paths = ['D:\\workspace\\FileForder\\coursedata\\classification\\A.label',
                   'D:\\workspace\\FileForder\\coursedata\\classification\\B.label',
                   'D:\\workspace\\FileForder\\coursedata\\classification\\C.label',
                   'D:\\workspace\\FileForder\\coursedata\\classification\\D.label',
                   'D:\\workspace\\FileForder\\coursedata\\classification\\E.label']
# 将前四个数据作为训练集导入
x_train,y_train = load_dataset(feature_paths[:4],label_paths[:4])
# 将最后一个数据集作为测试数据集
x_test,y_test = load_dataset(feature_paths[4:],label_paths[4:])
# 设置test_size=0将数据集随机打乱
x_train,x_,y_train,y_ = train_test_split(x_train,y_train,test_size=0.0)

# 创建K近邻分类器，并在测试集上进行预测
print("Start training knn\n")
knn = KNeighborsClassifier().fit(x_train,y_train)
print("Training done!")
answer_knn = knn.predict(x_test)
print("Predice done!")
print("\n\nThe classification report for knn")
print(classification_report(y_test,answer_knn))