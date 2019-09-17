import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tushare as ts
from functools import reduce
import warnings; warnings.simplefilter('ignore')


hs300 = ts.get_hs300s()['code'].tolist()
# print(hs300[:10])
StockBasics=ts.get_stock_basics()
StockBasics.reset_index(inplace=True)
# print(StockBasics.head())
data1=StockBasics.loc[StockBasics['code'].isin(hs300),['code','name','industry','pe','pb','esp','rev','profit']]
data1.columns=['代码','名称','行业','PE','PB','EPS','收入%','利润%']
# print(data1.head())
StockProfit=ts.get_profit_data(2019,1)
# print(StockProfit.head())
data2=StockProfit.loc[StockProfit['code'].isin(hs300),['code','roe','gross_profit_rate','net_profit_ratio']]
data2.columns=['代码', 'ROE', '毛利率', '净利率']
data2=round(data2,2)
# print(data2.head())
StockGrowth=ts.get_growth_data(2019,1)
data3=StockGrowth.loc[StockGrowth['code'].isin(hs300),['code','nprg']]
data3.columns=['代码','NI%']
data3=round(data3,2)
# print(data3.head())

merge=lambda x,y: pd.merge(x,y,how='left',on='代码')
data=reduce(merge,[data1,data2,data3])
data.drop_duplicates(inplace=True)
# print(data.head())
data['估值系数']=data['PE']*data['PB']
data=round(data,2)
# print(data.head())
FilteredData=data.loc[(data['估值系数']<60)&(data['ROE']>5),['代码', '名称', 'PE', 'PB', '估值系数', 'ROE', '收入%']]
# print(FilteredData.head())
print('筛选结果共%d只股' % len(FilteredData))
FilteredData.sort_values(['估值系数'],ascending=True,inplace=True)
# print(FilteredData.head(10))
def MapFunction(x):
    if x['ROE']>5:
        return '高成长'
    elif x['ROE']>=0:
        return '低成长'
    elif x['ROE']<0:
        return '亏损'
data['成长性']=data.apply(MapFunction,axis=1)
# print(data.tail(10))
DataGrowth=data[data['成长性']=='高成长'].sort_values(['估值系数'],ascending=True)
# print(DataGrowth.head(20))
DataProfit=DataGrowth.sort_values(['ROE'],ascending=True)
# print(DataProfit.head())
def GroupFunction(df):
    return df.sort_values(['估值系数'],ascending=True)[:2]
GroupedData=data.groupby('成长性').apply(GroupFunction)
# print(GroupedData)
GroupedData=data.groupby('行业').apply(GroupFunction)