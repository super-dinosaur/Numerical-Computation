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
    A = np.array([[2.34, 1, -1], [-3, -1, 2], [-2, 1, 2]])
    b = np.array([8, -11, -3])
    ic(np.linalg.solve(A, b))
    x = gaussian_elimination(A, b)
    ic(x)
    exit()
    ic(np.linalg.solve(A, b))
    ic(np.allclose(np.dot(A, x), b))
