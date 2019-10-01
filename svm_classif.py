from sklearn.metrics import roc_auc_score, recall_score, accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm

data = pd.read_csv('pre_data.csv', header=None)
x_train, x_test, y_train, y_test = train_test_split(data.ix[:, 1:3], data.ix[:, [5]], test_size=0.75)
clf = svm.SVC(C=0.95, kernel='rbf', gamma=20)
clf.fit(x_train, y_train.values.ravel())

print(clf.score(x_train, y_train))  # 精度
y_hat = clf.predict(x_train)

print(clf.score(x_test, y_test))
y_pre = clf.predict(x_test)

auc = roc_auc_score(y_test, y_pre)
print('测试集 AUC: ')
print(auc)
rs = recall_score(y_test, y_pre)
print('测试集 召回率: ')
print(rs)
auccuracy = accuracy_score(y_test, y_pre)
print('测试集 准确率: ')
print(auccuracy)
