
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###############################################
# 基本统计指标
###############################################
# def AnnualReturns(NetValueDF):
#     # 0. 创建备用表
#     tmpDF = NetValueDF.reset_index()

#     # 1. 计算年份并创建新表
#     yearIdx = np.unique(tmpDF['日期'].apply(lambda x:str(x)[:4]))                # 筛选出所有年份
#     yearNetValue = pd.DataFrame(index = yearIdx, columns=NetValueDF.columns)    # 创建新的记录表

#     # 2. 计算每年末资产净值
#     for y in yearIdx:                                 
#         mask = tmpDF['日期'].apply(lambda x:str(x)[:4]) == str(y)
#         end  = tmpDF['日期'][mask].max()                                         # 提取年末日期
#         yearNetValue.loc[y, :] = NetValueDF.loc[end, :].values
    
#     # 3. 计算年间收益率
#     handleNA = NetValueDF.dropna().iloc[0, :]
#     yearReturns = yearNetValue / yearNetValue.shift(1, axis=0).fillna(handleNA) - 1.0
    
#     return yearReturns


# def MonthlyReturns(NetValueDF):
#     # 0. 创建备用表
#     tmpDF = NetValueDF.reset_index()

#     # 1. 计算年份并创建新表
#     monthlyIdx = np.unique(tmpDF['日期'].apply(lambda x:str(x)[:7]))             # 筛选出所有年月
#     monthlyNetValue = pd.DataFrame(index = monthlyIdx, columns=NetValueDF.columns) # 创建新的记录表

#     # 2. 计算每年末资产净值
#     for m in monthlyIdx:                                 
#         mask = tmpDF['日期'].apply(lambda x:str(x)[:7]) == str(m)                
#         end  = tmpDF['日期'][mask].max()                                         # 提取月末日期
#         monthlyNetValue.loc[m, :] = NetValueDF.loc[end, :].values
    
#     # 3. 计算年间收益率
#     monthlyReturns = monthlyNetValue / monthlyNetValue.shift(1, axis=0) - 1.0
    
#     return monthlyReturns




def PeriodReturns(tradeDF):

    # 计算调仓日
    trading_day = tradeDF.index[tradeDF['仓位调整'] == 1.0]
    
    # 投资组合周期初净值
    valueDF_prev = tradeDF.loc[trading_day, '投资组合净值'].shift(1, axis=0)
    
    # 投资组合周期末净值
    valueDF_now = tradeDF.shift(1, axis=0).loc[trading_day, '投资组合净值']
    
    # 投资组合净值变化
    valueDF_delta = valueDF_now / valueDF_prev
    
    return valueDF_delta 


def AnnualReturns(tradeDF):

    # 一个交易周期内各资产的损益
    valueDF_delta = PeriodReturns(tradeDF)
    
    # 一年度内各资产的损益
    valueDF_annual = valueDF_delta.groupby(valueDF_delta.index.year).prod() - 1.0
    
    return valueDF_annual    


def DailyReturns(tradeDF):
    return tradeDF['投资组合净值'] / tradeDF['投资组合净值'].shift(1, axis=0) - 1.0

def AnnualVolatility(tradeDF):
    return DailyReturns(tradeDF).groupby(tradeDF.index.year).std() * np.sqrt(250)

def AnnualMaxDrawdown(tradeDF):
    return tradeDF['最大回撤'].groupby(tradeDF.index.year).min()



########################################
# 画柱形图
########################################
def BarPlot(DF, ttl, Ncol=2):
    
    N = len(DF.columns)
    Nrow = int(np.ceil(N / Ncol))
    flag = 0
    
    fig, axs = plt.subplots(nrows=Nrow, ncols=Ncol, figsize=(16, 3 * Nrow), constrained_layout=True)
    for i in range(Nrow):
        for j in range(Ncol):
            if (flag >= N):
                break
            axs[i][j].bar(DF.index, height=DF.iloc[:, flag], label=DF.columns[flag])
            axs[i][j].set_xlabel('时间')
            axs[i][j].set_ylabel('贡献度')
            axs[i][j].set_ylim(-1.5, 1.5)
            axs[i][j].legend(loc='best')
            axs[i][j].set_title( str(DF.columns[flag])+'对组合损益的贡献度' )
            flag += 1
    plt.savefig('Pics/贡献度_' + str(ttl) + '.png')
    
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
    fig = plt.figure(figsize=(16, 6), dpi=150)
    plt.stackplot(x, y, labels=labs)
    plt.legend(loc='best')
    plt.title(str(ttl) + '资产权重走势')
    plt.savefig('Pics/资产权重走势_' + str(ttl) + '.png')
    
    return


############################################################
# 贡献度计算
############################################################

def PeriodContribution(tradeDF):

    # 计算调仓日
    trading_day = tradeDF.index[tradeDF['仓位调整'] == 1.0]
    
    # 资产名称列表
    cols_name = tradeDF.columns[:-3]
    
    # 周期初资产净值
    valueDF_prev = tradeDF.loc[trading_day, cols_name].shift(1, axis=0)
    
    # 周期末资产净值
    valueDF_now = tradeDF.shift(1, axis=0).loc[trading_day, cols_name]
    
    # 资产净值变化
    valueDF_delta = valueDF_now - valueDF_prev
    valueDF_delta = valueDF_delta.fillna(0)               # 填充NaN类型
    
    return valueDF_delta 


def AnnualContribution(tradeDF):

    # 一个交易周期内各资产的损益
    valueDF_delta = PeriodContribution(tradeDF)
    
    # 一年度内各资产的损益
    valueDF_annual = valueDF_delta.groupby(valueDF_delta.index.year).sum()
    
    # 一年度内各资产对组合贡献度
    for col in valueDF_annual.columns:
        valueDF_annual[col] /= valueDF_annual['投资组合净值'].apply(abs)
    
    return valueDF_annual.iloc[:, :-1]                   # 舍去投资组合净值对应列


############################################################
# 汇总表现
############################################################

# 各年度表现汇总
def summaryDF(tradeDF):
    
    years = np.unique(tradeDF.reset_index()['日期'].apply(lambda x:str(x)[:4])) 
    DF = pd.DataFrame(index=years)

    DF["年收益率"] = AnnualReturns( tradeDF ).values
    DF["年波动率"] = AnnualVolatility( tradeDF ).values
    DF["信息比"] = DF['年收益率'] / DF['年波动率']
    DF["最大回撤"] = AnnualMaxDrawdown( tradeDF ).values

    return DF
    
    
def performanceDF(smryDF, tradeDF, name):
    DF = pd.DataFrame(index=[str(name)], 
                      columns=['年化收益','年化波动率','最大回撤','最长不创新高时间','信息比','Calmar比率'])

    DF['年化收益'] = smryDF['年收益率'].mean()
    DF['年化波动率'] = smryDF['年波动率'].mean()
    DF['最大回撤'] = smryDF['最大回撤'].min()
    DF['最长不创新高时间'] = tradeDF['最长不创新高时间'].max()
    DF['信息比'] = DF['年化收益'] / DF['年化波动率']
    DF['Calmar比率'] = DF['年化收益'] / np.abs( DF['最大回撤'] )

    return DF
