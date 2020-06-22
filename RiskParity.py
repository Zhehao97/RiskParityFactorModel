

import numpy as np
from scipy import optimize


# 20天滑动平均计算协方差矩阵
def CovrainceMatrix(dataFrame):
    return np.cov(dataFrame.values, rowvar=False, ddof=1)  # covraiance matrix of assets at time t


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
def ComputeWeight(W0, covarianceMat, thrd):  
    
    n = len(W0)
    
    bnds = []
    for i in range(n):
        bnds.append((0, thrd))                         # 0 < w < threshold
    bnds = tuple(bnds)
    
    cons = ({'type': 'eq', 'fun': lambda x:np.sum(x) - 1.0},
            {'type': 'ineq', 'fun': lambda x:x}
           )
    
    result = optimize.minimize(TargetFunction, W0, args=(covarianceMat), method='SLSQP', 
                               bounds=bnds, constraints=cons, tol=None, callback=None)
    
    W = np.around(result.x, decimals=5)
    W = W / np.sum(W)           # standarlization
    
    return W
    

