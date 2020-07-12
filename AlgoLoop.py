
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
def AlgoTrade(Prices, Returns, cumReturns, Turnovers, mode='plain', dt=120, up=0.50, 
                    thresholds={'Equity':0.25, 'FixedIncome':0.45, 'Commodity':0.10}, 
                    factorDict={'momentumX':False, 'momentumT':False, 
                                'reverseX':False, 'reverseT':False, 
                                'turnover':False, 
                                'copperGold':False, 'copperGas':False}):

    # 1.初始化数据变量
    alpha = 2 / dt                # 指数加权系数  
    totVal = 10000                # 初始资产 
    maxVal = 10000                # 历史最大净值
    maxDd = 0.0                   # 最大回撤
    flag = 0.0                    # flag = 1 有持仓，flag = 0 无持仓 
    
    # 1.1 创建一个副本
    tmpPrices     = Prices.copy()
    tmpReturns    = Returns.copy()   
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
        w = RiskParity.ComputeWeight(w0, V, col, thresholds)             # 等风险权重

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

            if factorDict['momentumX']:
            
                # 动量因子（横向比较）
                momentumX_col = Factors.momentumX(tmpCumReturns, col, t, dt)
                
                print(t, '横截面动量', momentumX_col)

                # 调整权重
                if (len(momentumX_col) > 0):
                    Weights.loc[idx, momentumX_col] = Weights.loc[idx, momentumX_col] * (1.0 + up)  # 上调重仓资产权重 
                    Weights.loc[idx, col] = Weights.loc[idx, col] / Weights.loc[idx, col].sum()     # 标准化处理, 无杠杆  
      

            if factorDict['momentumT']:

                # 动量因子（时序比较）
                momentumT_col = Factors.momentumT(tmpCumReturns, col, t, dt)
                
                print(t, '时序动量', momentumT_col)

                # 调整权重
                if (len(momentumT_col) > 0):
                    Weights.loc[idx, momentumT_col] = Weights.loc[idx, momentumT_col] * (1.0 + up)  # 上调重仓资产权重 
                    Weights.loc[idx, col] = Weights.loc[idx, col] / Weights.loc[idx, col].sum()     # 标准化处理, 无杠杆  


            if factorDict['reverseX']:
                
                # 反转因子（横向比较）
                reverseX_col = Factors.reverseX(tmpCumReturns, col, t, dt)
                
                print(t, '横截面反转', reverseX_col)

                # 调整权重
                if (len(reverseX_col) > 0):
                    Weights.loc[idx, reverseX_col] = Weights.loc[idx, reverseX_col] * (1.0 + up)    # 上调重仓资产权重 
                    Weights.loc[idx, col] = Weights.loc[idx, col] / Weights.loc[idx, col].sum()     # 标准化处理, 无杠杆 


            if factorDict['reverseT']:
                                
                # 反转因子（时序比较）
                reverseT_col = Factors.reverseT(tmpCumReturns, col, t, dt)
                
                print(t, '时序反转', reverseT_col)

                # 调整权重
                if (len(reverseT_col) > 0):
                    Weights.loc[idx, reverseT_col] = Weights.loc[idx, reverseT_col] * (1.0 + up)    # 上调重仓资产权重 
                    Weights.loc[idx, col] = Weights.loc[idx, col] / Weights.loc[idx, col].sum()     # 标准化处理, 无杠杆 


            if factorDict['turnover']:

                # 情绪因子（股指换手率）
                turnover_col = Factors.turnover(Turnovers, col, t, dt)
                
                print(t, '换手率', turnover_col)

                # 调整权重
                if (len(turnover_col) > 0):
                    Weights.loc[idx, turnover_col] = Weights.loc[idx, turnover_col] * (1.0 + up)    # 上调重仓资产权重 
                    Weights.loc[idx, col] = Weights.loc[idx, col] / Weights.loc[idx, col].sum()     # 标准化处理, 无杠杆  


            if factorDict['copperGold']:

                # 铜金价格比因子
                copper_gold = Factors.copperGold(tmpPrices, col, t, dt)
                
                print(t, '铜金', copper_gold)

                # 调整权重
                if (len(copper_gold) > 0):
                    Weights.loc[idx, copper_gold] = Weights.loc[idx, copper_gold] * (1.0 + up)      # 上调重仓资产权重 
                    Weights.loc[idx, col] = Weights.loc[idx, col] / Weights.loc[idx, col].sum()     # 标准化处理, 无杠杆 


            if factorDict['copperGas']:

                # 铜金价格比因子
                copper_gas = Factors.copperGas(tmpPrices, col, t, dt)
                
                print(t, '铜油', copper_gas)

                # 调整权重
                if (len(copper_gas) > 0):
                    Weights.loc[idx, copper_gas] = Weights.loc[idx, copper_gas] * (1.0 + up)        # 上调重仓资产权重 
                    Weights.loc[idx, col] = Weights.loc[idx, col] / Weights.loc[idx, col].sum()     # 标准化处理, 无杠杆  


            ###############################                
            # 计算各资产配比
            ###############################
            assetsVal = 0.999 * totVal * Weights.loc[idx, col]               # 各资产净值，千1手续费  
            flag = 1


        else:                
            ##############################                                                
            # 逐日盯市
            ##############################
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

