# coding=utf-8
import KNN
from sklearn.naive_bayes import GaussianNB
#导入预测结果评估模块
from sklearn.metrics import classification_report

"""
朴素贝叶斯
"""
print('Start training Bayes')
gnb = GaussianNB().fit(KNN.x_train, KNN.y_train)
print('Training done')
answer_gnb = gnb.predict(KNN.x_test)
print('Prediction done')

print('\n\nThe classification report for Bayes:')
print(classification_report(KNN.y_test, answer_gnb))