from dask import dataframe as dd
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from dask.multiprocessing import get
from itertools import chain
from functools import reduce
from datetime import datetime, timedelta
# from pyfinance.ols import PandasRollingOLS as rolling_ols
from pyfinance.utils import rolling_windows
from barra_cne6.barra_template import Data
from utility.factor_data_preprocess import adjust_months, add_to_panels
from roe_select.reorganize_data import append_df
from utility.relate_to_tushare import generate_months_ends
from barra_cne6.compute_factor_1 import CALFUNC


def scaler(se, scaler_max, scaler_min):
    # 归一化的算法逻辑
    # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # X_scaled = X_std * (max - min) + min

    tmp = se.sort_values(ascending=False)

    tmp_non = tmp[tmp.index[pd.isna(tmp)]]
    tmp = tmp.dropna()
    # 先自然排序，把收益率序列换成N到1的排序，收益率越高越大，
    after_sort = pd.Series(range(len(tmp), 0, -1), index=tmp.index)

    # 然后再归一化
    res = (after_sort - after_sort.min())/(after_sort.max() - after_sort.min())
    res = res*(scaler_max - scaler_min) + scaler_min

    res = pd.concat([res, tmp_non])

    return res


def df_from_series(s, col):
    res = pd.DataFrame(s.values, s.index, [col])
    return res


def tomax_dates(s):
    try:
        loc = np.where(s == s.max())[0][0]
        res = loc/(len(s)-1)
    except Exception as e:
        res = np.nan
    return res


def tomin_dates(s):
    try:
        loc = np.where(s == s.min())[0][0]
        res = loc / (len(s) - 1)
    except Exception as e:
        res = np.nan
    return res


# 过去n个周期的最高价格的list
def hhv(dat_df, m):
    dat_v = dat_df.values
    high_v = np.full(dat_df.shape, np.nan)
    for i in range(m, len(dat_df.columns)):
        count_sec = dat_v[:, i - m:i].max(axis=1)
        high_v[:, i] = count_sec

    high_df = pd.DataFrame(data=high_v, index=dat_df.index, columns=dat_df.columns)

    return high_df


# 过去n个周期的内有条件为真
def true_in_past_cond(dat_df, m):
    dat_v = dat_df.values
    cond_v = np.full(dat_df.shape, np.nan)
    for i in range(m, len(dat_df.columns)):
        cond_tmp = dat_v[:, i - m:i].any(axis=1)
        if cond_tmp.dtype != bool:
            print(cond_tmp.dtype)
        cond_v[:, i] = cond_tmp

    cond_df = pd.DataFrame(data=cond_v, index=dat_df.index, columns=dat_df.columns)

    cond_df.dropna(axis=1, how='all', inplace=True)

    cond_df[cond_df == 0.0] = False
    cond_df[cond_df == 1] = True

    return cond_df


# 过去n个周期的内有条件为真
def count_in_past_cond(dat_df, m):
    dat_v = dat_df.values
    count_v = np.full(dat_df.shape, np.nan)
    for i in range(m, len(dat_df.columns)):
        count_sec = dat_v[:, i - m:i].sum(axis=1)
        count_v[:, i] = count_sec

    count_df = pd.DataFrame(data=count_v, index=dat_df.index, columns=dat_df.columns)

    return count_df


# 均线
def ma(dat_df, n):

    if n == 0:
        return dat_df

    dat_v = dat_df.values
    ma_v = np.full(dat_df.shape, np.nan)
    if len(dat_df.columns) > 1:
        for i in range(n, len(dat_df.columns)):
            count_sec = dat_v[:, i - n:i].mean(axis=1)
            ma_v[:, i] = count_sec
    else:
        for i in range(n, len(dat_df.index)):
            count_sec = dat_v[i - n:i, :].mean(axis=0)
            ma_v[i, :] = count_sec

    ma_df = pd.DataFrame(data=ma_v, index=dat_df.index, columns=dat_df.columns)

    return ma_df


def multipyle_and(df1, df2, *dfs):
    pass
    # chain 是把多个迭代器合成一个迭代器
    dfs_all = [df for df in chain([df1, df2], dfs)]

    mut_index = sorted(reduce(lambda x, y: x.intersection(y), (df.index for df in dfs_all)))
    # 对columns求交集
    mut_cols = sorted(reduce(lambda x, y: x.intersection(y), (df.columns for df in dfs_all)))
    dfs_all = [df.loc[mut_index, mut_cols] for df in dfs_all]
    df_v_all = [df.values for df in dfs_all]
    res_v = reduce(lambda x, y: x & y, df_v_all)

    for i in df_v_all:
        print(i.dtype)

    res_df = pd.DataFrame(data=res_v, index=mut_index, columns=mut_cols)

    return res_df


class Factor(CALFUNC):


    def reverse(self):
        data = Data()
        all_codes = data.all_stocks_code
        all_codes = all_codes.set_index('wind_code')
        all_codes = all_codes['ipo_date']

        close_daily = data.closeprice_daily
        adjfactor = data.adjfactor
        close_price = close_daily * adjfactor
        close_price.dropna(axis=1, how='all', inplace=True)

        rps_50 = data.RPS_50
        rps_50.fillna(0, inplace=True)
        # rps_50 大于85
        if_rps_over_50 = rps_50.where(rps_50 > 85, False)
        if_rps_over_50 = if_rps_over_50.applymap(lambda x: x if isinstance(x, bool) else True)

        # 日线收盘价站上年线
        ma_close = ma(close_price, 240)
        close_price = close_price[ma_close.columns]
        if_up_ma = close_price > ma_close

        # 一月内曾创50日新高
        highest = hhv(close_price, 50)
        close_price = close_price[highest.columns]
        highest_shift = highest.shift(1, axis=1)
        new_high = close_price > highest_shift
        if_new_high_past_22 = true_in_past_cond(new_high, 22)

        # 收盘价站上年线的天数大于2，小于30
        up_ma_cound = count_in_past_cond(if_up_ma, 30)
        if_up_ma_over_2 = up_ma_cound > 2

        # 最高价距离120日内的最高价不到10%；
        highest_120 = hhv(close_price, 120)
        if_low_10p = close_price / highest_120 > 0.9

        res = multipyle_and(if_rps_over_50, if_up_ma, if_new_high_past_22, if_up_ma_over_2, if_low_10p)

        return res

    # 得到月度价格变化
    def month_changePCT(self):

        data = Data()
        # 日度收盘价
        close_daily = data.closeprice_daily
        adjfactor = data.adjfactor
        close_d = close_daily * adjfactor
        close_d.dropna(axis=1, how='all', inplace=True)
        close_d.dropna(axis=0, how='all', inplace=True)
        # 月末交易日日期
        month_ends = generate_months_ends()
        # 得到月度收盘价
        close_month = close_d.loc[:, month_ends]

        close_month.dropna(axis=1, how='all', inplace=True)
        close_month.dropna(axis=0, how='all', inplace=True)

        res = pd.DataFrame()
        for i in range(1, len(close_month.columns)):
            tmp = close_month[close_month.columns[i]]/close_month[close_month.columns[i-1]] - 1
            tmp_df = pd.DataFrame(tmp.values.reshape(-1, 1), index=tmp.index, columns=[close_month.columns[i]])
            res = pd.concat([res, tmp_df], axis=1)

        return res

    # 得到n个月的月度收益率
    def return_m(self, n):

        data = Data()
        # 月度收益率
        pct_month = data.changepct_monthly

        if n == 1:
           return pct_month
        else:
            pct_month = pct_month + 1
            res = pd.DataFrame()
            for i in range(n, len(pct_month.columns)):
                tmp = pct_month.iloc[:, i-n:i]
                tmp_cp = tmp.cumprod(axis=1)
                tmp_cp = tmp_cp[tmp_cp.columns[-1]] - 1
                tmp_df = pd.DataFrame(tmp_cp.values.reshape(-1, 1), index=tmp_cp.index, columns=[pct_month.columns[i]])
                res = pd.concat([res, tmp_df], axis=1)

            return res

    # 动量效应
    def ret_rank(self):
        data = Data()

        all_codes = data.all_stocks_code
        all_codes = all_codes.set_index('wind_code')
        all_codes = all_codes['ipo_date']

        return_12m = data.RETURN_12M

        # todo 先硬编码一下，后面再改
        firstindus = data.firstindustryname
        firstindus = data.reindex(firstindus)
        firstindus = firstindus[firstindus.columns[-1]]
        index_range = list(firstindus.index[firstindus == '银行'])
        return12 = return_12m.loc[index_range, :]

        all_codes = data.all_stocks_code
        all_codes = all_codes.set_index('wind_code')
        all_codes = all_codes['ipo_date']

        return12_scalered = return12.apply(scaler, scaler_max=100, scaler_min=1)
        # 剔除上市一年以内的情况，把上市一年以内的股票数据都设为nan
        for i, row in return12_scalered.iterrows():
            if i not in all_codes.index:
                row[:] = np.nan
                continue

            d = all_codes[i]
            row[row.index[row.index < d + timedelta(550)]] = np.nan

        return return12_scalered

    # 距离区间最高价最低价的距离
    def toMaxMin_distance_dates(self):
        data = Data()
        closeprice = data.closeprice_daily
        adj = data.adjfactor

        closeprice = closeprice * adj
        closeprice.dropna(how='all', axis=1, inplace=True)
        closeprice.dropna(how='all', axis=0, inplace=True)
        month_ends = generate_months_ends()

        new_dates = []
        for d in month_ends:
            if d in closeprice.columns:
                loc = np.where(closeprice.columns == d)[0][0]
                if loc < 225:
                    continue
                else:
                    new_dates.append(d)

        high_distance_1m = pd.DataFrame()
        high_distance_3m = pd.DataFrame()
        high_distance_6m = pd.DataFrame()

        low_distance_1m = pd.DataFrame()
        low_distance_3m = pd.DataFrame()
        low_distance_6m = pd.DataFrame()

        high_dates_1m = pd.DataFrame()
        high_dates_3m = pd.DataFrame()
        high_dates_6m = pd.DataFrame()

        low_dates_1m = pd.DataFrame()
        low_dates_3m = pd.DataFrame()
        low_dates_6m = pd.DataFrame()

        for d in new_dates:
            print(d)
            loc = np.where(closeprice.columns == d)[0][0]
            idx_e = loc
            idx_s1 = loc - 20
            idx_s3 = loc - 60
            idx_s6 = loc - 120

            closeprice_section_1m = closeprice.iloc[:, idx_s1:idx_e]
            closeprice_section_3m = closeprice.iloc[:, idx_s3:idx_e]
            closeprice_section_6m = closeprice.iloc[:, idx_s6:idx_e]

            highest_1m = closeprice_section_1m.max(axis=1)/closeprice_section_1m.iloc[:, -1]
            highest_3m = closeprice_section_3m.max(axis=1)/closeprice_section_3m.iloc[:, -1]
            highest_6m = closeprice_section_6m.max(axis=1)/closeprice_section_6m.iloc[:, -1]

            highest_1m = df_from_series(highest_1m, d)
            highest_3m = df_from_series(highest_3m, d)
            highest_6m = df_from_series(highest_6m, d)

            lowest_1m = closeprice_section_1m.min(axis=1)/closeprice_section_1m.iloc[:, -1]
            lowest_3m = closeprice_section_3m.min(axis=1)/closeprice_section_3m.iloc[:, -1]
            lowest_6m = closeprice_section_6m.min(axis=1)/closeprice_section_6m.iloc[:, -1]

            lowest_1m = df_from_series(lowest_1m, d)
            lowest_3m = df_from_series(lowest_3m, d)
            lowest_6m = df_from_series(lowest_6m, d)

            tomax_dates_1m = closeprice_section_1m.apply(tomax_dates, axis=1)
            tomax_dates_3m = closeprice_section_3m.apply(tomax_dates, axis=1)
            tomax_dates_6m = closeprice_section_6m.apply(tomax_dates, axis=1)
            tomin_dates_1m = closeprice_section_1m.apply(tomin_dates, axis=1)
            tomin_dates_3m = closeprice_section_3m.apply(tomin_dates, axis=1)
            tomin_dates_6m = closeprice_section_6m.apply(tomin_dates, axis=1)

            tomax_dates_1m = df_from_series(tomax_dates_1m, d)
            tomax_dates_3m = df_from_series(tomax_dates_3m, d)
            tomax_dates_6m = df_from_series(tomax_dates_6m, d)
            tomin_dates_1m = df_from_series(tomin_dates_1m, d)
            tomin_dates_3m = df_from_series(tomin_dates_3m, d)
            tomin_dates_6m = df_from_series(tomin_dates_6m, d)

            high_dates_1m = pd.concat([high_dates_1m, tomax_dates_1m], axis=1)
            high_dates_3m = pd.concat([high_dates_3m, tomax_dates_3m], axis=1)
            high_dates_6m = pd.concat([high_dates_6m, tomax_dates_6m], axis=1)

            low_dates_1m = pd.concat([low_dates_1m, tomin_dates_1m], axis=1)
            low_dates_3m = pd.concat([low_dates_3m, tomin_dates_3m], axis=1)
            low_dates_6m = pd.concat([low_dates_6m, tomin_dates_6m], axis=1)

            high_distance_1m = pd.concat([high_distance_1m, highest_1m], axis=1)
            high_distance_3m = pd.concat([high_distance_3m, highest_3m], axis=1)
            high_distance_6m = pd.concat([high_distance_6m, highest_6m], axis=1)

            low_distance_1m = pd.concat([low_distance_1m, lowest_1m], axis=1)
            low_distance_3m = pd.concat([low_distance_3m, lowest_3m], axis=1)
            low_distance_6m = pd.concat([low_distance_6m, lowest_6m], axis=1)

        return high_distance_1m, high_distance_3m, high_distance_6m, \
               low_distance_1m, low_distance_3m, low_distance_6m, \
               high_dates_1m, high_dates_3m, high_dates_6m, \
               low_dates_1m, low_dates_3m, low_dates_6m,

    # 正交后的一些因子
    def orthogonal_factor(self, orig_f, basic_f, file_path):
        '''
        orig_f 是原始的因子名称，被正交的
        basic_f 是用来正交化处理的因子名称，
        save_name 是正交化的残差存储的名字
        path 是截面因子存储的地址
        '''
        fp_list = os.listdir(file_path)
        res = pd.DataFrame()
        for fp in fp_list:
            fp_d = fp.split('.')[0]
            data = pd.read_csv(os.path.join(file_path, fp), engine='python',
                           encoding='gbk')
            data.set_index('code', inplace=True)
            tmpdata = data[[orig_f, basic_f]]
            tmpdata.where(tmpdata != np.inf, np.nan, inplace=True)
            tmpdata.where(tmpdata != -np.inf, np.nan, inplace=True)
            tmpdata.dropna(how='any', axis=0, inplace=True)

            y = tmpdata[orig_f]
            x = tmpdata[basic_f]
            x = sm.add_constant(x)
            # np.any(x['return_1m'] == np.inf)
            model = sm.OLS(y, x)
            results = model.fit()

            params = results.params
            resid = y - pd.Series(np.dot(x, params), index=y.index)
            resid = df_from_series(resid, fp_d)
            res = pd.concat([res, resid], axis=1)

        return res

    def factor_scaler(self, factor_name):
        data = Data()
        dat_df = eval('data.'+factor_name)

        scalered = dat_df.apply(scaler, scaler_max=100, scaler_min=1)
        return scalered


    # 距离区间最高价最低价的距离
    def peg_factor(self):
        data = Data()
        epsyoy = data.basicepsyoy
        # 财务指标常规处理，移动月份，改月末日期
        # 调整为公告日期
        epsyoy = adjust_months(epsyoy)
        # 用来扩展月度数据
        epsyoy = append_df(epsyoy)

        ep_daily = data.ep_daily
        months_end = generate_months_ends()
        new_columns = [col for col in ep_daily.columns if col in months_end]
        ep = ep_daily[new_columns]

        def ffun(x):
            try:
                res = 1/x
            except Exception as e:
                res = None
            return res

        pe_monthly = ep.applymap(lambda x: ffun(x))
        peg = pe_monthly/epsyoy
        peg.index.name = 'Code'

        return peg

    def profit_ratio_diff(self):
        data = Data()
        gir = data.grossincomeratio
        # 财务指标常规处理，移动月份，改月末日期
        # 调整为公告日期

        gir_d = data.generate_diff(gir)
        gir_d = adjust_months(gir_d)
        # 用来扩展月度数据
        gir_d = append_df(gir_d)

        return gir_d





if __name__ == "__main__":
    f = Factor()
    res = f.deal_macro_dat()
    # factor_name = 'macro_data'
    # f.save(res, factor_name.upper())

    peg = f.peg_factor()
    factor_name = 'PEG'
    f.save(peg, factor_name)
    # 如果数据的最后一列不是月末的话，要删除
    peg = peg.drop(peg.columns[-1], axis=1)

    panel_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子'
    add_to_panels(peg, panel_path, factor_name, freq_in_dat='M')

    '''
    # 计算一个因子，并添加到现有的因子数据中
    panel_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子'

    f = Factor()
    # factor_name = 'main_net_buy_ratio'
    # ss = f.factor_scaler(factor_name)
    # f.save(ss, factor_name.upper() + '_SCALERED')

    rps, rps_50 = f.rps()
    rps.index.name = 'code'
    add_to_panels(rps, panel_path, 'RPS', freq_in_dat='D')
    f.save(rps, 'RPS')
    # f.save(rps_50, 'RPS_50')
    #
    # reverse = f.reverse()
    # f.save(reverse, 'REVERSE')

    # rps_industry = f.rps_by_industry()
    # rps_industry.index.name = 'code'
    # add_to_panels(rps_industry, panel_path, 'rps_industry'.upper(), freq_in_dat='D')
    # f.save(rps_industry, 'rps_industry'.upper())

    # changePCT_monthly = f.month_changePCT()
    # save_path = r'D:\pythoncode\IndexEnhancement\barra_cne6\download_from_juyuan'
    # f.save(changePCT_monthly, 'changePCT_monthly', save_path)
    #
    # return_1m = f.return_m(1)
    # return_3m = f.return_m(3)
    # return_6m = f.return_m(6)
    # return_12m = f.return_m(12)
    #
    # f.save(return_1m, 'RETURN_1M')
    # f.save(return_3m, 'RETURN_3M')
    # f.save(return_6m, 'RETURN_6M')
    # f.save(return_12m, 'RETURN_12M')

    # return12_scalered = f.ret_rank()
    # f.save(return12_scalered, 'RETURN_12M_SCALERED')

    # h_dis_1m, h_dis_3m, h_dis_6m, l_dis_1m, l_dis_3m, l_dis_6m, \
    # h_dates_1m, h_dates_3m, h_dates_6m, \
    # l_dates_1m, l_dates_3m, l_dates_6m = f.toMaxMin_distance_dates()
    #
    # f.save(h_dis_1m, 'TO_MAX_DIS_1M')
    # f.save(h_dis_3m, 'TO_MAX_DIS_3M')
    # f.save(h_dis_6m, 'TO_MAX_DIS_6M')
    # f.save(l_dis_1m, 'TO_MIN_DIS_1M')
    # f.save(l_dis_3m, 'TO_MIN_DIS_3M')
    # f.save(l_dis_6m, 'TO_MIN_DIS_6M')
    #
    # f.save(h_dates_1m, 'TO_MAX_TIME_1M')
    # f.save(h_dates_3m, 'TO_MAX_TIME_3M')
    # f.save(h_dates_6m, 'TO_MAX_TIME_6M')
    # f.save(l_dates_1m, 'TO_MIN_TIME_1M')
    # f.save(l_dates_3m, 'TO_MIN_TIME_3M')
    # f.save(l_dates_6m, 'TO_MIN_TIME_6M')
    #
    # h_dis_1m.index.name = 'code'
    # h_dis_3m.index.name = 'code'
    # h_dis_6m.index.name = 'code'
    #
    # add_to_panels(h_dis_1m, panel_path, 'TO_MAX_DIS_1M')
    # add_to_panels(h_dis_3m, panel_path, 'TO_MAX_DIS_3M')
    # add_to_panels(h_dis_6m, panel_path, 'TO_MAX_DIS_6M')
    #
    # l_dis_1m.index.name = 'code'
    # l_dis_3m.index.name = 'code'
    # l_dis_6m.index.name = 'code'
    #
    # add_to_panels(l_dis_1m, panel_path, 'TO_MIN_DIS_1M')
    # add_to_panels(l_dis_3m, panel_path, 'TO_MIN_DIS_3M')
    # add_to_panels(l_dis_6m, panel_path, 'TO_MIN_DIS_6M')
    #
    # h_dates_1m.index.name = 'code'
    # h_dates_3m.index.name = 'code'
    # h_dates_6m.index.name = 'code'
    #
    # add_to_panels(h_dates_1m, panel_path, 'TO_MAX_TIME_1M')
    # add_to_panels(h_dates_3m, panel_path, 'TO_MAX_TIME_3M')
    # add_to_panels(h_dates_6m, panel_path, 'TO_MAX_TIME_6M')
    #
    # l_dates_1m.index.name = 'code'
    # l_dates_3m.index.name = 'code'
    # l_dates_6m.index.name = 'code'
    #
    # add_to_panels(l_dates_1m, panel_path, 'TO_MIN_TIME_1M')
    # add_to_panels(l_dates_3m, panel_path, 'TO_MIN_TIME_3M')
    # add_to_panels(l_dates_6m, panel_path, 'TO_MIN_TIME_6M')

    # f = Factor()
    # file_path = r'D:\pythoncode\IndexEnhancement\因子预处理模块\因子'
    # # orig_f_list = ['TO_MAX_DIS_1M', 'TO_MAX_DIS_3M', 'TO_MAX_DIS_6M', 'TO_MIN_DIS_1M', 'TO_MIN_DIS_3M', 'TO_MIN_DIS_6M',
    # #                'TO_MAX_TIME_1M', 'TO_MAX_TIME_3M', 'TO_MAX_TIME_6M', 'TO_MIN_TIME_1M', 'TO_MIN_TIME_3M',
    # #                'TO_MIN_TIME_6M']
    # orig_f_list = ['TO_MAX_DIS_3M']
    # basic_f = 'return_1m'
    # for orig_f in orig_f_list:
    #     print(orig_f)
    #     res = f.orthogonal_factor(orig_f, basic_f, file_path)
    #     save_name = orig_f + '_RESID'
    #     f.save(res, save_name)
    #     res.index.name = 'code'
    #     add_to_panels(res, panel_path, save_name)

'''
