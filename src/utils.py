
import numpy as np

def minL2NormToMinima(x, minima):
    minL2Norm = np.linalg.norm(x=x-minima[0,:], ord=2)
    for i in range(1, minima.shape[0]):
        l2Norm = np.linalg.norm(x=x-minima[i,:], ord=2)
        if l2Norm<minL2Norm:
            minL2Norm = l2Norm
    return minL2Norm
