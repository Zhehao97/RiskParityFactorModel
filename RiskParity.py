
import numpy as np
from scipy import optimize


# 20天滑动平均计算协方差矩阵
def CovrainceMatrix(dataFrame):
    return np.cov(dataFrame.values.astype(np.float32), rowvar=False, ddof=1)  # covraiance matrix of assets at time t


# 优化问题中的目标函数
def TargetFunction(weightVec, covarianceMat):
    n = len(weightVec)                                 # scalar type, number of assets
    
    W = np.array(weightVec)                            # vector type, weights of assets
    V = np.matrix(covarianceMat)                       # matrix type, covariance matrix of assets
    
    sigma = W[np.newaxis, :] * V * W[:, np.newaxis]    # matrix type 
    sigma = np.sqrt(sigma[0, 0])                       # scalar type, portfolio volatility
    
    tmp = W - sigma ** 2.0 / n / np.dot(V, W.T)        # matrix type
    tmp = tmp.A1                                       # vector type
    
    return np.sum( tmp * tmp )


# 计算各个资产权重
def ComputeWeight(W0, covarianceMat, cols, thrds={'Equity':0.25, 'FixedIncome':0.45, 'Commodity':0.10}):  
    
    # Boundarys
    bnds = []

    for i in range(len(cols)):
        if (cols[i] == '沪深300') or (cols[i] == '中证500') or (cols[i] == '标普500'):
            bnds.append((0, thrds['Equity']))   
        elif (cols[i] == '中国10年国债') or (cols[i] == '信用债3-5AAA') or (cols[i] == '美国10年国债'):
            bnds.append((0, thrds['FixedIncome']))
        else:
            bnds.append((0, thrds['Commodity']))

    bnds = tuple(bnds)


    # Constaints
    cons = ({'type': 'eq', 'fun': lambda x:np.sum(x) - 1.0},
            {'type': 'ineq', 'fun': lambda x:x}
           )
    
    # Optimization results
    result = optimize.minimize(TargetFunction, W0, args=(covarianceMat), method='SLSQP', 
                               bounds=bnds, constraints=cons, tol=None, callback=None)
    
    # Normalization
    W = np.around(result.x, decimals=5)
    W = W / np.sum(W)           # standarlization
    
    return W
    

