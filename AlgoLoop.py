

import numpy as np
import pandas as pd
import RiskParity 
import Factors

# 记录投资权重
def recordWeights(WeightDF, idx, colNames, w):
    for i in range(len(colNames)):
        WeightDF.loc[idx, colNames[i]] = w[i]
    return WeightDF


# 记录交易数据
def recordTrades(TradeDF, idx, col, assetsValue, totalValue, maxDrawdown, pos=0):
    for i in range(len(col)):
        TradeDF.loc[idx, col[i]] = assetsValue[i]
    TradeDF.loc[idx, '投资组合净值'] = totalValue
    TradeDF.loc[idx, '最大回撤'] = maxDrawdown
    TradeDF.loc[idx, '仓位调整'] = pos
    return TradeDF



# 主回测程序
def AlgoTrade(Returns, cumReturns, Turnovers, mode='plain', dt=120, up=0.2, threshold=0.9, 
                                factorDict={'momentumX':False, 'momentumT':False, 'turnover':False}):

    # 1.初始化数据变量
    alpha = 2 / dt                # 指数加权系数  
    totVal = 10000                # 初始资产 
    maxVal = 10000                # 历史最大净值
    maxDd = 0.0                   # 最大回撤
    flag = 0.0                    # flag = 1 有持仓，flag = 0 无持仓 
    
    tmpReturns = Returns.copy()   # 创建一个副本
    tmpCumReturns = cumReturns.copy()
    
    
    # 2.初始化表格
    Weights = pd.DataFrame(columns=tmpReturns.columns, index=tmpReturns.index)  # 投资权重
    Trades  = pd.DataFrame(columns=tmpReturns.columns, index=tmpReturns.index)  # 投资组合价值
    Trades['投资组合净值'] = np.nan
    Trades['最大回撤'] = np.nan
    Trades['仓位调整'] = np.nan
   

    # 3. 主循环
    for t in range(dt, tmpReturns.shape[0]):

        # 3.1 截取时间窗口
        idx_prev = tmpReturns.index[t-1]                                    # t-1时刻对应日期
        idx      = tmpReturns.index[t]                                      # t时刻对应日期
        subFrame = tmpReturns.iloc[t-dt:t-1, :].dropna(axis=1, how='any')   # 剔除缺失数据的资产
        col = subFrame.columns                                              # 资产名称
        n   = subFrame.shape[1]                                             # 资产数量

        
        # 3.2 计算等风险权重
        # 初始化
        w0 = np.repeat(0.4, n)                                           # 初始化权重
        V  = RiskParity.CovrainceMatrix(subFrame)                        # 各资产日收益率协方差矩阵


        # 优化后的各资产权重
        w_prev = Weights.loc[idx_prev, col].values                       # t-1时刻各资产权重
        w = RiskParity.ComputeWeight(w0, V, threshold)                   # 等风险权重

        # 是否将权重进行指数平均
        if mode == 'ema':
            if ( np.isnan(w_prev.sum()) ):                               # 检查权重是否全为空值
                pass
            else:
                w = alpha * w + (1 - alpha) * w_prev                     # 指数加权平均权重
        
        # 记录权重数据
        Weights = recordWeights(Weights, idx, col, w)
                
                
        # 3.3 调整仓位（每dt个交易日）
        if (t % dt == 0) and (flag == 0):                                # 调仓日  

            topCol = []                                                  # 重仓股列表

            if factorDict['momentumX']:
            
                # 横截面动量因子筛选出的重仓资产
                topCol = Factors.momentumX(tmpCumReturns, col, t, dt)

                # 调整权重
                if (len(topCol) > 0):
                    Weights.loc[idx, topCol] = Weights.loc[idx, topCol] * (1.0 + up)            # 上调重仓资产权重 
                    Weights.loc[idx, col] = Weights.loc[idx, col] / Weights.loc[idx, col].sum() # 标准化处理, 无杠杆  
      

            if factorDict['momentumT']:

                # 时序动量因子筛选出的重仓资产
                topCol = Factors.momentumT(tmpCumReturns, col, t, dt)

                # 调整权重
                if (len(topCol) > 0):
                    Weights.loc[idx, topCol] = Weights.loc[idx, topCol] * (1.0 + up)            # 上调重仓资产权重 
                    Weights.loc[idx, col] = Weights.loc[idx, col] / Weights.loc[idx, col].sum() # 标准化处理, 无杠杆  


            if factorDict['turnover']:

                # 时序换手率因子决定是否重仓股指
                topCol = Factors.turnover(Turnovers, col, t, dt)

                                # 调整权重
                if (len(topCol) > 0):
                    Weights.loc[idx, topCol] = Weights.loc[idx, topCol] * (1.0 + up)            # 上调重仓资产权重 
                    Weights.loc[idx, col] = Weights.loc[idx, col] / Weights.loc[idx, col].sum() # 标准化处理, 无杠杆  

                
            # 计算各资产配比
            assetsVal = 0.999 * totVal * Weights.loc[idx, col]               # 各资产净值，千1手续费  
            flag = 1

        else:                                                                
            # 逐日盯市
            assetsVal = Trades.loc[idx_prev, col] * (1.0 + Returns.loc[idx, col])
            flag = 0

            
        # 3.4 更新净值并计算最大回撤
        totVal = np.sum(assetsVal)                                           # 更新投资组合净值

        if maxVal < totVal:
            maxVal = totVal                                                  # 更新历史最大净值

        maxDd = (totVal - maxVal) / maxVal                                   # 计算最大回撤
        
        
        # 3.5 记录交易数据
        Trades  = recordTrades(Trades, idx, col, assetsVal, totVal, maxDd, pos=flag)   
        
#     Weights.to_csv(str(dt) + mode + '权重.csv', encoding='utf_8_sig')
#     Trades.to_csv(str(dt) + mode + '交易.csv', encoding='utf_8_sig')
    
    return Trades, Weights

