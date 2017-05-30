# coding=utf-8
import KNN
from sklearn.tree import DecisionTreeClassifier
#导入预测结果评估模块
from sklearn.metrics import classification_report
"""
决策树
"""
print('Start training DT')
dt = DecisionTreeClassifier().fit(KNN.x_train, KNN.y_train)
print('Training done')
answer_dt = dt.predict(KNN.x_test)
print('Prediction done')

print('\n\nThe classification report for DT:')
print(classification_report(KNN.y_test, answer_dt))
