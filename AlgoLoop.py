import numpy as np
import pandas as pd
import RiskParity as RP

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
def AlgoTrade(Returns, cumReturns, dt, up=0.2, totVal=10000, mode='plain', momentumX=False, momentumT=False):
    
    # 1.初始化数据变量
    alpha = 2 / dt                # 指数加权系数   
    totVal = 10000.0              # 组合总净值
    maxVal = 10000.0              # 历史最大净值
    maxDd = 0.0                   # 最大回撤
    flag = 0.0                    # flag = 1 有持仓，flag = 0 无持仓 
    
    
    # 2.初始化表格
    Weights = pd.DataFrame(columns=Returns.columns, index=Returns.index)  # 投资权重
    Trades  = pd.DataFrame(columns=Returns.columns, index=Returns.index)  # 投资组合价值
    Trades['投资组合净值'] = np.nan
    Trades['最大回撤'] = np.nan
    Trades['仓位调整'] = np.nan

    # 3. 主循环
    for t in range(dt, Returns.shape[0]):

        # 3.1 截取时间窗口
        idx_prev = Returns.index[t-1]                                    # t-1时刻对应日期
        idx      = Returns.index[t]                                      # t时刻对应日期
        subFrame = Returns.iloc[t-dt:t-1, :].dropna(axis=1, how='any')   # 剔除缺失数据的资产
        col = subFrame.columns                                           # 资产名称
        n   = subFrame.shape[1]                                          # 资产数量

        
        # 3.2 计算等风险权重
        # 初始化
        w0 = np.repeat(0.4, n)                                           # 初始化权重
        V  = RP.CovrainceMatrix(subFrame)                                # 各资产日收益率协方差矩阵


        # 优化后的各资产权重
        w_prev = Weights.loc[idx_prev, col].values                       # t-1时刻各资产权重
        w = RP.ComputeWeight(w0, V)                                      # 等风险权重
        w = w / np.sum(w)                                                # 标准化处理

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
            
            # 引入横截面动量因子（dt时间内收益率横向比较）
            if momentumX:

                # 动量因子分组
                m = int(n / 4)                                                 # 取排名前m的资产重仓
                momentX = cumReturns.iloc[t-1, :] / cumReturns.iloc[t-dt, :]   # 本周期各资产收益率
                momentX = momentX.sort_values(ascending=False)                 # 各资产按收益率排序
                topCol = momentX[:m].index & col                               # 头部资产名称
                
                # 计算混合权重
                if not topCol.empty:
                    Weights.loc[idx, topCol] = Weights.loc[idx, topCol] * (1.0 + up)            # 上调重仓资产权重 
                    Weights.loc[idx, col] = Weights.loc[idx, col] / Weights.loc[idx, col].sum() # 标准化处理, 无杠杆


            # 引入时序动量因子（dt时间内收益率环比增长）
            if momentumT:
                if (t - 2 * dt >= 0):
                    
                    # 计算动量因子
                    momentT_prev = cumReturns.iloc[t-dt-1, :] / cumReturns.iloc[t-2*dt, :]  # 上一周期收益率
                    momentT      = cumReturns.iloc[t-1, :] / cumReturns.iloc[t-dt, :]       # 本周期收益率
                    topCol = momentT.index[momentT > momentT_prev] & col                      # 收益率环比增长>0的资产
                    
                    # 计算混合权重
                    if not topCol.empty:
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
    
    return Trades

