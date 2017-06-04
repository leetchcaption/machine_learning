# coding=utf-8
import numpy as np
#使用listdir访问本地模块
from os import listdir
#导入神经网络
from sklearn.neural_network import MLPClassifier

#将32*32位图转换为1*1024矩阵
def img2vector(filName):
    retMat = np.zeros([1024],int) # 定义返回的矩阵，大小为1*1024
    fr = open(filName)
    lines = fr.readlines()
    for i in range(32):
        for j in range(32):
            retMat[i*32+j] = lines[i][j]
    return retMat

#加载数据集
def readDataSet(path):
    filelist = listdir(path) # 获取文件夹下的所有文件
    numfiles = len(filelist) #统计需要读取的文件数量
    dataSet = np.zeros([numfiles,1024],int)  #用于存放所有的数据文件
    hwlabels = np.zeros([numfiles,10],int)      #用于存放对应的one-hot标签
    for i in range(numfiles):
        filePath = filelist[i] #获取文件名称和路径
        dight = int(filePath.split("_")[0])
        hwlabels[i][dight] = 1.0
        dataSet[i] = img2vector(path+"/"+filePath)
    return dataSet,hwlabels


# read dataSet
train_dataSet, train_hwLabels = readDataSet('D:/workspace/FileForder/coursedata/handswrite/trainingDigits')

clf = MLPClassifier(hidden_layer_sizes=(100,),
                    activation='logistic', solver='adam',
                    learning_rate_init=0.0001, max_iter=2000)
print(clf)
clf.fit(train_dataSet, train_hwLabels)

# read  testing dataSet
dataSet, hwLabels = readDataSet('D:/workspace/FileForder/coursedata/handswrite/testDigits')
res = clf.predict(dataSet)  # 对测试集进行预测
error_num = 0  # 统计预测错误的数目
num = len(dataSet)  # 测试集的数目
for i in range(num):  # 遍历预测结果
    # 比较长度为10的数组，返回包含01的数组，0为不同，1为相同
    # 若预测结果与真实结果相同，则10个数字全为1，否则不全为1
    if np.sum(res[i] == hwLabels[i]) < 10:
        error_num += 1
print("Total num:", num, " Wrong num:", \
      error_num, "  WrongRate:", error_num / float(num))
