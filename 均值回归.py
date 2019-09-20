import matplotlib.pyplot as plt
plt.style.use('seaborn')
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
import warnings; warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import tushare as ts

data=ts.get_k_data('hs300','2013-01-01','2019-06-30')[['date','close']]
data.rename(columns={'close':'price'},inplace=True)
data.set_index('date',inplace=True)
data['price'].plot(figsize=(10,6))
# plt.show()

data['returns']=np.log(data['price']/data['price'].shift(1))
SMA=50
data['SMA']=data['price'].rolling(SMA).mean()
threshold=250
data['distance']=data['price']-data['SMA']
data['distance'].dropna().plot(figsize=(10,6))
plt.axhline(threshold,color='r')
plt.axhline(-threshold,color='r')
plt.axhline(0,color='r')
# plt.show()
data['position']=np.where(data['distance']>threshold,-1,np.nan)
data['position']=np.where(data['distance']<-threshold,1,data['position'])
data['position']=np.where(data['distance']*data['distance'].shift(1)<0,0,data['position'])
data['position']=data['position'].ffill().fillna(0)
#print(data.tail(40))
data['position'].ix[SMA:].plot(ylim=[-1.1,1.1],figsize=(10, 6))
data['strategy']=data['position'].shift(1)*data['returns']
data[['returns','strategy']].dropna().cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()
