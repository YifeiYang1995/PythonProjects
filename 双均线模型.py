import matplotlib.pyplot as plt
import seaborn
import matplotlib as mpl
mpl.rcParams['font.family']='serif'
import warnings; warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import tushare as ts

data=ts.get_k_data('hs300',start='2012-01-01',end='2019-06-30')
data=pd.DataFrame(data)
# print(data.head())
data.rename(columns={'close':'price'},inplace=True)
data.set_index('date',inplace=True)
data['SMA_10']=data['price'].rolling(10).mean()
data['SMA_60']=data['price'].rolling(60).mean()
# print(data.tail())
data[['price','SMA_10','SMA_60']].plot(title='HS300 stock price | 10 & 60 days SMAs',figsize=(10,6))
# plt.show()

data['position']=np.where(data['SMA_10']>data['SMA_60'],1,-1)
data.dropna(inplace=True)
data['position'].plot(ylim=[-1.1,1.1],title='Market position')
data['returns']=np.log(data['price']/data['price'].shift(1))
data['returns'].hist(bins=35)
data['strategy']=data['position'].shift(1)*data['returns']
data[['returns','strategy']].cumsum().apply(np.exp).plot(figsize=(10,6))

# 策略收益风险评估

# data[['returns','strategy']].mean()*252
# data[['returns','strategy']].std()*252**0.5
data['CumReturns']=data['strategy'].cumsum().apply(np.exp)
data['CumMax']=data['CumReturns'].cummax()
# print(data.tail())
data[['CumReturns','CumMax']].plot(figsize=(10,6))
plt.show()
