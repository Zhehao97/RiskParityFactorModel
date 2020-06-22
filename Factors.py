

import numpy as np
import pandas as pd


# 横截面动量因子
def momentumX(cumReturns, col, t, dt):

    m = int(len(col) / 4)                                                # 取排名前m的资产重仓
    momentX = cumReturns.iloc[t-1, :] / cumReturns.iloc[t-dt, :]         # 本周期各资产收益率
    momentX = momentX.sort_values(ascending=False)                       # 各资产按收益率排序
    topCol = momentX[:m].index & col                                     # 头部资产名称

    return topCol.to_list()

# 时序动量因子
def momentumT(cumReturns, col, t, dt):

	if (t - 2 * dt >= 0):
	    
	    momentT_prev = cumReturns.iloc[t-dt-1, :] / cumReturns.iloc[t-2*dt, :]  # 上一周期收益率
	    momentT      = cumReturns.iloc[t-1, :] / cumReturns.iloc[t-dt, :]       # 本周期收益率
	    topCol = momentT.index[momentT > momentT_prev] & col                    # 收益率环比增长>0的资产

	    return topCol.to_list()

	else:
		return []          

def turnover(Turnovers, col, t, dt):

	tmpTurnovers = Turnovers.ewm(span=dt, axis=0).mean()
	tmp_prev = tmpTurnovers.iloc[t-dt, :]
	tmp_now  = tmpTurnovers.iloc[t, :]
	topCol = tmp_now.index[tmp_now > tmp_prev] & col

	return topCol.to_list()

