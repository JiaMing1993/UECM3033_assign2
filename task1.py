import numpy as np
import scipy.linalg as sp

def lu(A, b):
    x = sp.lu_factor(A,False,True)
    sol = sp.lu_solve(x,b,0,True)
    return list(sol)

def sor(A, b):
    L= -1*sp.tril(A,-1)
    U=-1*sp.triu(A,1)
    D=np.diag(np.diag(A))
    Kj=np.linalg.inv(D).dot(L+U)
    Y=np.max(np.abs(sp.eigvals(Kj)))
    omega=(2-(1-(np.sqrt(1-Y**2))))/(Y**2)
    Q = D/omega -L
    K = np.linalg.inv(Q).dot(Q-A)
    c = np.linalg.inv(Q).dot(b)
    x = np.zeros_like(b)
    for itr in range(10):
        x    = K.dot(x) + c
    sol=x
    return list(sol)

def solve(A, b): 
    try:       
        np.linalg.cholesky(A)
    except np.linalg.linalg.LinAlgError:
        print('Solve by LU')
        return lu(A,b)
    print('Solve by SOR')    
    return sor(A,b)

if __name__ == "__main__":

    A = [[2,1,6], [8,3,2], [1,5,1]]
    b = [9, 13, 7]
    sol = solve(A,b)
    print(sol)
    
    A = [[6566, -5202, -4040, -5224, 1420, 6229],
         [4104, 7449, -2518, -4588,-8841, 4040],
         [5266,-4008,6803, -4702, 1240, 5060],
         [-9306, 7213,5723, 7961, -1981,-8834],
         [-3782, 3840, 2464, -8389, 9781,-3334],
         [-6903, 5610, 4306, 5548, -1380, 3539.]]
    b = [ 17603,  -63286,   56563,  -26523.5, 103396.5, -27906]
    sol = solve(A,b)
    print(sol)
    
    
