import numpy as np
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

if __name__ == '__main__':
    array = np.array([1,2,3,4,5])
    min_arg = np.argmin(array)
    ic(min_arg)
