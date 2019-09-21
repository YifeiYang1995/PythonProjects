import matplotlib.pyplot as plt
plt.style.use('seaborn')
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
import warnings; warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import tushare as ts

data1=ts.get_k_data('600199','2013-06-01','2014-12-31')[['date','close']]
data2=ts.get_k_data('600702','2013-06-01','2014-12-31')['close']
data=pd.concat([data1,data2],axis=1)
data.set_index('date',inplace=True)
stocks_pair = ['600199', '600702']
data.columns=stocks_pair
data.plot(figsize= (8,6))

data['PriceDelta']=data['600199']-data['600702']
data['PriceDelta'].plot(figsize=(8,6))
plt.ylabel('Spread')
plt.axhline(data['PriceDelta'].mean())
data['z-score']=(data['PriceDelta']-np.mean(data['PriceDelta']))/np.std(data['PriceDelta'])
# print(data[data['z-score']<-1.5].head())
# len(data[data['z-score']<-1.5])
data['position_1']=np.where(data['z-score']>1.5,-1,np.nan)
data['position_1']=np.where(data['z-score']<-1.5,1,data['position_1'])
data['position_1']=np.where(abs(data['z-score'])<0.5,0,data['position_1'])
data['position_1']=data['position_1'].fillna(method='ffill')
# print(data.head(50))
data['position_1'].plot(ylim=[-1.1,1.1],figsize=(10,6))
data['position_2']=-np.sign(data['position_1'])
data['position_2'].plot(ylim=[-1.1,1.1],figsize=(10,6))

data['returns_1']=np.log(data['600199']/data['600199'].shift(1))
data['returns_2']=np.log(data['600702']/data['600702'].shift(1))
data['strategy']=0.5*data['position_1'].shift(1)*data['returns_1']+0.5*data['position_2'].shift(1)*data['returns_2']
data[['strategy','returns_1','returns_2']].cumsum().apply(np.exp).plot(figsize=(10,6))
plt.show()

