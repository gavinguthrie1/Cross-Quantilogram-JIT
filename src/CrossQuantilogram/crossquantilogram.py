import pandas as pd
import numpy as np
from numba import jit

def CrossQuantilogram(x1,alpha1,x2,alpha2,k):
    '''
    Calculate and return Cross-Quantilogram ρ.

    Input
    -----
        x1:array-like, x1's quantile level.
        alpha1: float between (0,1), quantile of serie-1.
        x2:array-like, x2's quantile level (k lagged).
        alpha2: float between (0,1), quantile of serie-2.
        k: non-negative integer, the serie-2's lag.

    Output
    ------
        The Cross-Quantilogram statistics, a float number.

    '''
    if k==0:
        array_x1 = np.array(x1)
        array_x2 = np.array(x2)
    elif k > 0:
        array_x1 = np.array(x1[k:])
        array_x2 = np.array(x2[:-k])
    elif k < 0:
        array_x1 = np.array(x1[:k])
        array_x2 = np.array(x2[-k:])

    if len(array_x2.shape)>1:
        raise ValueError("x2 must be 1D array")
        
    if len(array_x1.shape)==1:
        array_x1=array_x1.reshape(array_x1.shape[0],1)

    q1 = np.percentile(array_x1, alpha1*100, axis=0, interpolation='higher')
    q2 = np.percentile(array_x2, alpha2*100, axis=0, interpolation='higher')
    
    psi1 = (array_x1 < q1) - alpha1
    psi2 = (array_x2 < q2) - alpha2
    
    numerator = np.sum(np.multiply(psi1, psi2.reshape(psi2.shape[0],1)),axis=0)
    denominator = np.multiply(np.sqrt(np.sum(np.square(psi1),axis=0)), 
                                np.sqrt(np.sum(np.square(psi2))))
    if numerator.shape[0]==1:
        return np.divide(numerator, denominator)[0]
    else:
        return np.divide(numerator, denominator) 

def jitCrossQuantilogram(x1,alpha1,x2,alpha2,k):
    '''
    Calculate and return Cross-Quantilogram ρ using JIT.

    Input
    -----
        x1: Pandas Dataframe or array-like, x1's quantile level.
        alpha1: float between (0,1), quantile of series-1.
        x2: Pandas Dataframe or array-like, x2's quantile level (k lagged).
        alpha2: float between (0,1), quantile of series-2.
        k: non-negative integer, the serie-2's lag.

    Output
    ------
        The Cross-Quantilogram statistics, a float number.

    '''
    
    x1 = np.array(x1)
    x2 = np.array(x2)

    return jitNumpyCrossQuantilogram(x1,alpha1,x2,alpha2,k)

@jit(nopython=True)
def jitNumpyCrossQuantilogram(x1,alpha1,x2,alpha2,k):
    '''
    Calculate and return Cross-Quantilogram ρ using JIT.

    Input
    -----
        x1: Numpy array, x1's quantile level.
        alpha1: float between (0,1), quantile of series-1.
        x2: Numpy array, x2's quantile level (k lagged).
        alpha2: float between (0,1), quantile of series-2.
        k: non-negative integer, the serie-2's lag.

    Output
    ------
        The Cross-Quantilogram statistics, a float number.

    '''
    if k > 0:
        array_x1 = x1[k:]
        array_x2 = x2[:-k]
    elif k < 0:
        array_x1 = x1[:k]
        array_x2 = x2[-k:]
    else:
        array_x1 = x1
        array_x2 = x2
        
    if len(array_x2.shape)>1:
        raise ValueError("x2 must be 1D array")

    array_x1=array_x1.reshape(array_x1.shape[0],1)
    
    q1 = np.nanpercentile(array_x1, alpha1*100)
    q2 = np.nanpercentile(array_x2, alpha1*100)
        
    psi1 = (array_x1 < q1) - alpha1
    psi2 = (array_x2 < q2) - alpha2
    
    numerator = np.sum(np.multiply(psi1, psi2.reshape(psi2.shape[0],1)),axis=0)
    denominator = np.multiply(np.sqrt(np.sum(np.square(psi1),axis=0)), 
                                np.sqrt(np.sum(np.square(psi2))))
     
    return np.divide(numerator, denominator)[0]






    
        
    
    
    