import matplotlib.pyplot as plt
import seaborn
import matplotlib as mpl
mpl.rcParams['font.family']='serif'
import warnings; warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import tushare as ts

hs300=ts.get_k_data('hs300','2012-01-01','2019-06-30')[['date','close']]
hs300.rename(columns={'close':'price'},inplace=True)
hs300.set_index('date',inplace=True)
# print(hs300.head())
hs300['SMA_10']=hs300['price'].rolling(10).mean()
hs300['SMA_60']=hs300['price'].rolling(60).mean()
hs300[['price','SMA_10','SMA_60']].plot(grid=True,figsize=(10,6))
hs300['10-60']=hs300['SMA_10']-hs300['SMA_60']
SD=20
hs300['regime']=np.where(hs300['10-60']>SD,1,0)
hs300['regime']=np.where(hs300['10-60']<-SD,-1,hs300['regime'])
# print(hs300['regime'].value_counts())
hs300['market']=np.log(hs300['price']/hs300['price'].shift(1))
hs300['strategy']=hs300['regime'].shift(1)*hs300['market']
hs300[['market','strategy']].cumsum().apply(np.exp).plot(grid=True,figsize=(10,6))
plt.show()