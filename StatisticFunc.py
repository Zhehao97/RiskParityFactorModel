
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###############################################
# 基本统计指标
###############################################
def AnnualReturns(NetValueDF):
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
    handleNA = NetValueDF.dropna().iloc[0, :]
    yearReturns = yearNetValue / yearNetValue.shift(1, axis=0).fillna(handleNA) - 1.0
    
    return yearReturns


def MonthlyReturns(NetValueDF):
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


def DailyReturns(NetValueDF):
    return NetValueDF / NetValueDF.shift(1, axis=0) - 1.0

def AnnualVolatility(NetValueDF):
    return DailyReturns(NetValueDF).groupby(NetValueDF.index.year).std() * np.sqrt(250)

def AnnualMaxDrawdown(MDD):
    return MDD.groupby(MDD.index.year).min()


########################################
# 画柱形图
########################################
def BarPlot(DF, ttl, Ncol=2):
    
    N = len(DF.columns)
    Nrow = int(np.ceil(N / Ncol))
    flag = 0
    
    fig, axs = plt.subplots(nrows=Nrow, ncols=Ncol, figsize=(16, 12), constrained_layout=True)
    for i in range(Nrow):
        for j in range(Ncol):
            if (flag >= N):
                break
            axs[i][j].bar(DF.index, height=DF.iloc[:, flag], label=DF.columns[flag])
            axs[i][j].set_xlabel('时间')
            axs[i][j].set_ylabel('贡献度')
            axs[i][j].set_ylim(-1.0, 1.0)
            axs[i][j].legend(loc='best')
            axs[i][j].set_title( str(DF.columns[flag])+'对组合损益的贡献度' )
            flag += 1
    plt.savefig('贡献度_' + str(ttl) + '.png')
    
    return 
 


########################################
# 权重走势
########################################
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
    plt.savefig('资产权重走势_' + str(ttl) + '.png')
    
    return


############################################################
# 贡献度计算
############################################################

def PeriodContribution(tradeDF, weightDF, assetDF):

    # 计算调仓日
    trading_day = tradeDF.index[tradeDF['仓位调整'] == 1.0]

    # 一个交易周期内各资产所占权重
    W = weightDF.loc[trading_day, :].shift(1, axis=0)

    # 一个交易周期内投资组合损益
    PortfolioPnL = tradeDF.loc[trading_day, '投资组合净值'] / tradeDF.loc[trading_day, '投资组合净值'].shift(1, axis=0) - 1.0

    # 一个交易周期内各资产损益
    AssetPnL = assetDF.loc[trading_day, :] / assetDF.loc[trading_day, :].shift(1, axis=0) - 1.0

    # 一个交易周期内各资产对组合损益的贡献度
    weightedPnL = AssetPnL * W 

    # 计算各资产对组合损益的贡献度
    for col in weightedPnL.columns:
        weightedPnL[col] /= PortfolioPnL

    return weightedPnL


def AnnualContribution(tradeDF, weightDF, assetDF):

    # 一个周期内各资产对组合的贡献
    C = PeriodContribution(tradeDF, weightDF, assetDF)

    # 以组合的净值差作为权重系数
    W = tradeDF.loc[C.index, '投资组合净值'] - tradeDF.loc[C.index, '投资组合净值'].shift(1, axis=0)

    # 先加权 x = (b - a) * x1 + (c - b) * x2
    for col in C.columns:
        C[col] = C[col] * W
    C = C.groupby(C.index.year).sum()

    # 后平均 x = x / (c-a)
    W = W.groupby(W.index.year).sum()
    for col in C.columns:
        C[col] = C[col] / W

    return C
