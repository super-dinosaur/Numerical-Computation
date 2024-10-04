import numpy as np
import os
import os.path as osp

from numpy.typing import NDArray
from numpy import floating  
from typing import List, Tuple, Any
from icecream import ic

def gaussian_elimination(A, b):
    n = len(b)
    
    # Augment the matrix A with vector b to form [A|b]
    Ab = np.hstack([A, b.reshape(-1, 1)])

    # Forward elimination
    for i in range(n):
        # Pivoting: Swap rows to move the largest pivot element to the diagonal
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]
        
        # Make the pivot element 1 and eliminate the below rows
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] = Ab[j, i:] - factor * Ab[i, i:]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    
    return x

class EvalTool():
    @staticmethod   
    def gaussian_elimination(A,b):
        return gaussian_elimination(A,b)
    
    @staticmethod
    def mean_diff_table(x: np.ndarray, 
                        y: np.ndarray
        ) -> np.ndarray:
        n = len(x)-1
        table = np.zeros((n+1,n+1))
        table[:,0] = y
        for j in range(1,n+1):
            for i in range(j,n+1):
                table[i,j] = (table[i,j-1] - table[i-1,j-1])/(x[i]-x[i-j])
        return table
    
    @staticmethod
    def div_diff_table(x: np.ndarray, 
                       y: np.ndarray
        ) -> np.ndarray:
        n = len(x)-1
        table = np.zeros((n+1,n+1))
        table[:,0] = y
        for j in range(1,n+1):
            for i in range(j,n+1):
                table[i,j] = table[i,j-1] - table[i-1,j-1]
        return table
    
    @staticmethod
    def find_nearest(array: np.ndarray, 
                     value: floating
        ) -> Tuple[int, int]:
        idx1 = (np.abs(array - value)).argmin()
        array[idx1] = np.inf
        idx2 = (np.abs(array - value)).argmin()
        return idx1, idx2

if __name__ == "__main__":
    x = np.linspace(0,5,5)
    y = np.sin(x)
    table = EvalTool.mean_diff_table(x,y)
    x_table = np.hstack([x.reshape(-1,1),table])
    ic(osp.join(os.getcwd(),"x_table.txt"))
    with open("x_table.txt","w") as f:
        f.write(str(x_table))
        
