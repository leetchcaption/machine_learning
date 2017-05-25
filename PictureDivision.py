# coding=utf-8
# 利用k-means聚类算法对图像像素点颜色进行聚类实现简单的图像分割

# 1、建立工程并引入sklearn包
# 2、加载图像并进行预处理
# 3、加载K-means聚类算法
# 4、对图像像素点进行聚类并输出

from sklearn.cluster import KMeans
import PIL.Image as image
import numpy as np

def loadData(filePath):
    f = open(filePath,'rb')
    data = []
    img = image.open(f) # 以像素点形式返回图像值
    m,n = img.size
    for i in range(m):
        for j in range(n):
            x,y,z = img.getpixel((i,j))
            data.append([x/256.0,y/256.0,z/256.0])
    f.close()
    return np.mat(data),m,n # 以矩阵形式返回图形

imgData,row,col = loadData('D:\\workspace\\FileForder\\基于聚类的整图分割\\bull.jpg')
km = KMeans(n_clusters=3)
#聚类获得每个像素所属的分类
label = km.fit_predict(imgData)
label = label.reshape([row,col])
pic_new = image.new("L",(row,col))
# 根据所属类别向图中添加灰度值
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i,j),int(256/(label[i][j]+1)))
# 以JPG格式保存图像
pic_new.save("D:\\workspace\\FileForder\\基于聚类的整图分割\\result_bull.jpg","JPEG")