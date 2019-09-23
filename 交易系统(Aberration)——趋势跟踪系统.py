import matplotlib.pyplot as plt
plt.style.use('seaborn')
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
import warnings; warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import tushare as ts

code='002397'
length=10
OpenTrigger=0.5
StopwinTrigger=3
StoploseTrigger=1

data=ts.get_k_data(code,'2014-01-01','2019-01-01')
data['pct_change']=data['close'].pct_change()
data['MA']=data['pct_change'].rolling(window=length,min_periods=3).mean()
data['std']=data['pct_change'].rolling(window=length,min_periods=3).std()
# print(data.head())
data['MaxStd']=data['std'].rolling(window=length).max()
data['yes_MA']=data['MA'].shift(1)
data['yes_MaxStd']=data['MaxStd'].shift(1)
data['long_open_price']=data['yes_MA']+data['yes_MaxStd']*OpenTrigger
data['long_stopwin_price']=data['yes_MA']+data['yes_MaxStd']*StopwinTrigger
# print(data.loc[10:15,['date','pct_change','MA','std','yes_MaxStd','long_open_price', 'long_stopwin_price']])
data['long_open_signal']=np.where(data['high']>data['long_open_price'],1,0)
data['long_stopwin_signal']=np.where(data['high']>data['long_stopwin_price'],1,0)

flag = 0    # 记录持仓情况，0代表空仓，1代表持仓；
# 前12个数据因均值计算无效所以不作为待处理数据
# 终止数据选择倒数第二个以防止当天止盈情况会以第二天开盘价平仓导致无数据情况发生
# 最后一天不再进行操作；可能会面临最后一天开仓之后当天触发平仓，要用下一天开盘价卖出，无法得到；
for i in range(12,(len(data)-1)):
    if flag==1:
        StoplosePrice=max(data.loc[i,'yes_MA'],long_open_price_1-long_open_delta * StoploseTrigger)
        if data.loc[i,'long_stopwin_signal']:
            data.loc[i,'return']=data.loc[i,'long_stopwin_price']/data.loc[i-1,'close']-1
            flag=0
        elif data.loc[i,'low']<StoplosePrice:
            data.loc[i,'return']=min(data.loc[i,'open'],StoplosePrice)/data.loc[i-1,'close']-1
            flag=0
        else:
            data.loc[i, 'return'] = data.loc[i, 'close'] / data.loc[i - 1, 'close'] - 1
    else:
        if data.loc[i,'long_open_signal']:
            flag=1
            long_open_price_1=max(data.loc[i,'open'],data.loc[i,'long_open_price'])
            long_open_delta=data.loc[i, 'yes_MaxStd']
            data.loc[i,'return']=data.loc[i,'close']/long_open_price_1-1
            StoplosePrice=max(data.loc[i,'yes_MA'],long_open_price_1-long_open_delta * StoploseTrigger)
            if (data.loc[i,'low']<StoplosePrice or data.loc[i, 'long_stopwin_signal']):
                data.loc[i,'return']=data.loc[i+1, 'open']/long_open_price_1 - 1
                flag=0
# print(data.tail())
data['return'].fillna(0,inplace=True)
data['strategy_return']=(data['return']+1).cumprod()
data['stock_return'] = (data['pct_change'] + 1).cumprod()
fig=plt.figure(figsize=(10,6))
ax = fig.add_subplot(1,1,1)
ax.plot(data.stock_return)
ax.plot(data.strategy_return)
plt.title(code)
plt.legend()
plt.show()









