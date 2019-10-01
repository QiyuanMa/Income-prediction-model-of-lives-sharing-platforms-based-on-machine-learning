from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import timeit
# coding=gbk
# import xlrd
# data = xlrd.open_workbook('wtrain1.xlsx')
# table_old = data.sheets()[0]
# # 输出读取的数据参数
# print('table1:', table_old)
train_data = pd.read_csv("wtrain1.csv")
train_data = np.array(train_data)
train_data = train_data[:,1:len(train_data[0])]
x_train,y_train = np.split(train_data,(3,),axis = 1)


test_data = pd.read_csv("wtest1.csv")
test_data = np.array(test_data)
test_data = test_data[:,1:len(test_data[0])]
x_test,y_test = np.split(test_data,(3,),axis = 1)


# clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
# scores = cross_val_score(clf, X, y)
# print(scores.mean())1
#

timeit

#n_estimators=25, n_jobs=-1, max_depth=None, min_samples_split=2
clf = RandomForestClassifier(n_estimators=30, oob_score=True, n_jobs=-1, random_state=50, max_features="auto")
clf.fit(x_train, y_train.ravel())
print('______________')
print(clf.feature_importances_)
print('______________')
print(clf.score(x_train, y_train))  # 精度

print(clf.score(x_test, y_test))
y_pre = clf.predict(x_test)

print('++++++++++++++++++++++++++++++++=')
print(y_pre)
print('++++++++++++++++++++++++++++++++=')

#auc
auc = roc_auc_score(y_test, y_pre)
auc = str(np.float64(auc).item())
print('auc:'+auc)

#交叉验证

#交叉验证的方法可以帮助我们进行调参，最终得到一组最佳的模型参数使得测试数据的准确率和泛化能力最佳
#交叉验证的准确率
accuracy = cross_val_score(clf, x_test, y_test.ravel(), cv=9, scoring='accuracy')
print(accuracy.mean())

#交叉验证的auc
roc_auc = cross_val_score(clf, x_test, y_test.ravel(), cv=9, scoring='roc_auc')
print('AUC: '+roc_auc.mean())


#召回率
recall_score = cross_val_score(clf, x_test, y_test.ravel(), cv=9, scoring='recall')
print('召回率: '+recall_score.mean())

#模型准确性 https://blog.csdn.net/tttwister/article/details/81138865
print('_________________________________________')
f1 = str(np.float64(f1_score(y_test, y_pre)).item())
print('F1: '+f1)
# # recall = str(np.float64(recall_score(y_test, y_pre)).item())
# print('recall_score: '+recall_score(y_test, y_pre).values)
# precision_score= str(np.float64(precision_score(y_test, y_pre)).item())
# print('precision_score: '+precision_score)
# print('_________________________________________')
# accuracy = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
# print(accuracy.mean())
# #
# # recall_score = cross_val_score(clf, X, y, cv=3, scoring='recall')
# # print(recall_score.mean())
# auc_score = cross_val_score(clf, X, y, cv=3, scoring='roc_auc')
# print(auc_score.mean())

# clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
# scores = cross_val_score(clf, X, y)
# print(scores.mean())
