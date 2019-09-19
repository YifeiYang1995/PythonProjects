import matplotlib.pyplot as plt
plt.style.use('seaborn')
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
import warnings; warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import tushare as ts

data=ts.get_k_data('hs300','2012-01-01','2019-06-30')[['date','close']]
data.rename(columns={'close':'price'},inplace=True)
data.set_index('date',inplace=True)
data['returns']=np.log(data['price']/data['price'].shift(1))
data['position']=np.sign(data['returns'])
data['strategy']=data['returns']*data['position'].shift(1)
data[['returns','strategy']].cumsum().apply(np.exp).plot(figsize=(10,6))

#策略优化
data['position_5']=np.sign(data['returns'].rolling(5).mean())
data['strategy_5']=data['returns']*data['position_5'].shift(1)
data[['returns','strategy_5']].dropna().cumsum().apply(np.exp).plot(figsize=(10,6))

#参数寻优——使用离散Return计算方法
data['returns_dis']=data['price']/data['price'].shift(1)-1
#data['returns_dis']=data['price'].pct_change()
data['returns_dis_cum']=(data['returns_dis']+1).cumprod()
price_plot=['returns_dis_cum']
for days in [10,20,30,60]:
    price_plot.append('strategy_dis_cum_%d' % days)
    data['position_%d' % days]=np.where(data['returns'].rolling(days).mean()>0,1,-1)
    data['strategy_%d' % days]=data['returns']*data['position_%d' % days].shift(1)
    data['strategy_dis_cum_%d' % days]=(data['strategy_%d' % days]+1).cumprod()
#   data['strategy_%d' % days]=np.sign(data['returns'].rolling(days).mean())
# print(data.head())
data[price_plot].dropna().plot(title='HS300 Multi Parameters Momuntum Strategy',figsize=(10, 6),style=['--', '--', '--', '--','--'])
plt.show()
