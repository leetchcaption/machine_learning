# coding=utf-8
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def loadData(path):
    fr = open(path,'r+')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1, len(items))])
    return retData,retCityName

if __name__ == '__main__':

    data,city = loadData('C:\\Users\\leetch\\Desktop\\课程数据\\聚类\\31procosData.txt')
    km = KMeans(n_clusters=4) #指定聚类中心的个数
    label = km.fit_predict(data)#计算簇中心以及为簇分配序号
    expenses = np.sum(km.cluster_centers_,axis=1)
    print(expenses,label)
    CityCluster = [[],[],[],[]]
    for i in range(len(city)):
        CityCluster[label[i]].append(city[i]) # 将城市按照label分设成不同的簇
    for i in range(len(CityCluster)):
        print("Expenses:%.2f" % expenses[i])
        print(CityCluster[i])

    print(type(CityCluster))




