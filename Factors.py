
import numpy as np
import pandas as pd


# 横截面动量因子
def momentumX(cumReturns, col, t, dt):

    m = 2                                                                # 取排名前m的资产重仓
    momentX = cumReturns.iloc[t-1, :] / cumReturns.iloc[t-dt, :]         # 本周期各资产收益率
    momentX = momentX.sort_values(ascending=False)                       # 各资产按收益率排序
    topCol = momentX[:m].index & col                                     # 头部资产名称

    return topCol.to_list()


def momentumT(cumReturns, col, t, dt):

    if (t - 2 * dt >= 0):

        momentT_prev = cumReturns.iloc[t-dt-1, :] / cumReturns.iloc[t-2*dt, :]  # 上一周期收益率
        momentT      = cumReturns.iloc[t-1, :] / cumReturns.iloc[t-dt, :]       # 本周期收益率
        topCol = momentT.index[momentT > momentT_prev] & col                    # 收益率环比增长>0的资产

        return topCol.to_list()

    else:
        return []          


def reverseX(cumReturns, col, t, dt):

    m = 2                                                                  # 取排名前m的资产重仓
    reverseX = cumReturns.iloc[t-1, :] / cumReturns.iloc[t-dt, :]          # 本周期各资产收益率
    reverseX = reverseX.sort_values(ascending=True)                        # 各资产按收益率排序
    bottomCol = reverseX[:m].index & col                                   # 尾部资产名称

    return bottomCol.to_list()


def reverseT(cumReturns, col, t, dt):

    if (t - 2 * dt >= 0):

        reverseT_prev = cumReturns.iloc[t-dt-1, :] / cumReturns.iloc[t-2*dt, :]  # 上一周期收益率
        reverseT      = cumReturns.iloc[t-1, :] / cumReturns.iloc[t-dt, :]       # 本周期收益率
        bottomCol = reverseT.index[reverseT < reverseT_prev] & col               # 收益率环比增长>0的资产

        return bottomCol.to_list()

    else:
        return []     


def turnover(Turnovers, col, t, dt):
        
    tmpTurnovers = Turnovers.ewm(span=dt, axis=0).mean()
    tmp_prev = tmpTurnovers.iloc[t-dt, :]
    tmp_now  = tmpTurnovers.iloc[t, :]
    topCol = tmp_now.index[tmp_now > tmp_prev] & col
    
    return topCol.to_list()
    

def copperGold(Prices, col, t, dt):
    
    ratioCG = Prices['中信证券COMEX铜期货'] / Prices['中信证券COMEX黄金期货']

    ratioCG = ratioCG.ewm(span=dt, axis=0).mean()
    ratio_prev = ratioCG[t-dt]
    ratio_now  = ratioCG[t]
    topCol = set(['10年国债', '信用债3-5AAA']) & set(col.to_list())
        
    if (ratio_now < ratio_prev):
        return list(topCol)
        
    else:
        return []  


def copperGas(Prices, col, t, dt):

    ratioCG = Prices['中信证券COMEX铜期货'] / Prices['中信证券WTI原油期货']

    ratioCG = ratioCG.ewm(span=dt, axis=0).mean()
    ratio_prev = ratioCG[t-dt]
    ratio_now  = ratioCG[t]
    topCol = set(['沪深300', '中证500']) & set(col.to_list())
        
    if (ratio_now > ratio_prev):
        return list(topCol)
    
    else:
        return []     


def fxRate(FXs, col, t, dt):
    
    fxs = FXs.ewm(span=dt, axis=0).mean()
    fx_prev = fxs['美元汇率'][t-dt]
    fx_now = fxs['美元汇率'][t]
    
    # 重仓中国国债
    if (fx_now > fx_prev):  
        topCol = set(['10年国债']) & set(col.to_list())
    
    # 重仓美国国债
    elif (fx_now < fx_prev): 
        topCol = set(['10年美债']) & set(col.to_list())
        
    else:
        topCol = []
        
    return topCol











