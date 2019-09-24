import matplotlib.pyplot as plt
plt.style.use('seaborn')
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
import warnings; warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
import tushare as ts
import sklearn
from sklearn import linear_model
from sklearn.svm import SVC

hs300=ts.get_k_data('hs300','2015-01-01','2018-06-25')
hs300.set_index('date',inplace=True)
hs300['returns']=hs300['close'].pct_change()
hs300.dropna(inplace=True)
# print(hs300.head())

#   数据处理——特征工程处理
for i in range(1,8,1):
    hs300['close - %dd' % i]=hs300['close'].shift(i)
# print(hs300.head(8))
hs_7d=hs300[[x for x in hs300.columns if 'close' in x]].iloc[7:]
X_train=hs_7d
X_train=sklearn.preprocessing.scale(X_train)

#    逻辑回归预测股价趋势算法实现
lm=linear_model.LogisticRegression(C=1000)
y_train=np.sign(hs_7d['close'].pct_change().shift(-1))   # 计算出训练集的labels；
y_train.replace(to_replace=np.NaN,value=0,inplace=True)
# print(y_train.value_counts())
y_train=y_train.values.reshape(-1,1)
# print(y_train[-10:])
lm.fit(X_train,y_train)
lm.score(X_train,y_train)
hs300['prediction']=np.NaN
hs300['prediction'].ix[7:]=lm.predict(X_train)
# print(hs300['prediction'].value_counts())
hs300['strategy']=(hs300['prediction'].shift(1)*hs300['returns']+1).cumprod()
hs300['CumReturns']=(hs300['returns']+1).cumprod()
hs300[['strategy','CumReturns']].dropna().plot(figsize=(10,6))


#    改变算法：SVM
X_train=hs_7d
clf_SVC=SVC(kernel='linear')
clf_SVC.fit(X_train,y_train)
clf_SVC.score(X_train,y_train)
hs300['prediction_1']=np.NaN
hs300['prediction_1'].ix[7:]=clf_SVC.predict(X_train)
# print(hs300['prediction_1'].value_counts())
hs300['strategy_1']=(hs300['prediction_1'].shift(1)*hs300['returns']+1).cumprod()
hs300[['strategy_1','CumReturns']].dropna().plot(figsize=(10,6))


#    逻辑回归算法在测试集的验证
hs300_test=ts.get_k_data('hs300','2018-07-01','2019-06-30')
hs300_test.set_index('date',inplace=True)
hs300_test['returns']=hs300_test['close'].pct_change()
hs300_test.dropna(inplace=True)
for i in range(1,8,1):
    hs300_test['close - %dd' % i]=hs300_test['close'].shift(i)
hs_7d_test=hs300_test[[x for x in hs300_test.columns if 'close' in x]].iloc[7:]
X_test=hs_7d_test
X_test=sklearn.preprocessing.scale(X_test)
# print(X_test)
hs300_test['prediction']=np.NaN
hs300_test['prediction'].ix[7:]=lm.predict(X_test)
# print(hs300_test['prediction'].value_counts())
hs300_test['strategy']=(hs300_test['prediction'].shift(1)*hs300_test['returns']+1).cumprod()
hs300_test['CumReturns']=(hs300_test['returns']+1).cumprod()
hs300_test[['strategy','CumReturns']].dropna().plot(figsize=(10,6))


#     SVM算法在测试集的验证
X_test=hs_7d_test
hs300_test['prediction_1']=np.NaN
hs300_test['prediction_1'].ix[7:]=clf_SVC.predict(X_test)
hs300_test['strategy_1']=(hs300_test['prediction_1'].shift(1)*hs300_test['returns']+1).cumprod()
hs300_test[['strategy_1','CumReturns']].dropna().plot(figsize=(10,6))
plt.show()





