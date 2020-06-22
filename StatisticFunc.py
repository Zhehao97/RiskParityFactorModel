

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 统计每年末各资产净值
def annualReturns(NetValueDF):
    # 0. 创建备用表
    tmpDF = NetValueDF.reset_index()

    # 1. 计算年份并创建新表
    yearIdx = np.unique(tmpDF['日期'].apply(lambda x:str(x)[:4]))                # 筛选出所有年份
    yearNetValue = pd.DataFrame(index = yearIdx, columns=NetValueDF.columns)    # 创建新的记录表

    # 2. 计算每年末资产净值
    for y in yearIdx:                                 
        mask = tmpDF['日期'].apply(lambda x:str(x)[:4]) == str(y)
        end  = tmpDF['日期'][mask].max()                                         # 提取年末日期
        yearNetValue.loc[y, :] = NetValueDF.loc[end, :].values
    
    # 3. 计算年间收益率
    yearReturns = yearNetValue / yearNetValue.shift(1, axis=0) - 1.0
    
    return yearReturns


def monthlyReturns(NetValueDF):
    # 0. 创建备用表
    tmpDF = NetValueDF.reset_index()

    # 1. 计算年份并创建新表
    monthlyIdx = np.unique(tmpDF['日期'].apply(lambda x:str(x)[:7]))             # 筛选出所有年月
    monthlyNetValue = pd.DataFrame(index = monthlyIdx, columns=NetValueDF.columns) # 创建新的记录表

    # 2. 计算每年末资产净值
    for m in monthlyIdx:                                 
        mask = tmpDF['日期'].apply(lambda x:str(x)[:7]) == str(m)                
        end  = tmpDF['日期'][mask].max()                                         # 提取月末日期
        monthlyNetValue.loc[m, :] = NetValueDF.loc[end, :].values
    
    # 3. 计算年间收益率
    monthlyReturns = monthlyNetValue / monthlyNetValue.shift(1, axis=0) - 1.0
    
    return monthlyReturns


def ReturnDist(ReturnsDF, Ncol=2):
    
    N = len(ReturnsDF.columns)
    Nrow = int(np.ceil(N / Ncol))
    flag = 0
    
    fig, axs = plt.subplots(nrows=Nrow, ncols=Ncol, figsize=(16, 12), constrained_layout=True)
    for i in range(Nrow):
        for j in range(Ncol):
            if (flag >= N):
                break
            axs[i][j].hist(ReturnsDF.iloc[:, flag], bins=15, label='年间收益率')
            axs[i][j].set_xlabel('收益率')
            axs[i][j].set_ylabel('频数')
            axs[i][j].set_xlim(-0.20, 0.20)
            axs[i][j].legend(loc='best')
            axs[i][j].set_title(ReturnsDF.columns[flag])
            flag += 1
    plt.show()
    
    return 


def WeightPlot(tradeDF, weightDF, ttl):
    
    # 1. Find the data of trading day
    tradingDay = tradeDF.index[tradeDF['仓位调整']==1.0]
    tmpDF = weightDF.loc[tradingDay, :]
    tmpDF = tmpDF.fillna(0)
    
    # 2. Prepare the data for plotting
    labs = tmpDF.columns.to_list()

    x = tmpDF.index.to_list()
    y = []
    for col in tmpDF.columns:
        y.append( tmpDF[col].values.tolist() )

    y = np.vstack(y)
    
    # 2. Plot the figure and save
    fig = plt.figure(figsize=(16, 8), dpi=150)
    plt.stackplot(x, y, labels=labs)
    plt.legend(loc='best')
    plt.title(str(ttl) + '资产权重走势')
    plt.savefig( str(ttl)+'资产权重走势.jpg' )
    
    return
