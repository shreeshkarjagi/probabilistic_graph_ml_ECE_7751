import numpy as np

def calc_likelihood(X, Y, Z, a, b, g, th, mu):
    m, n = X.shape
    
    # count Z=1
    n_black = np.sum(Z == 1)
    
    # horiz neighbors matching
    n_h = 0
    for i in range(m):
        for j in range(n-1):
            if Z[i,j] == Z[i,j+1]:
                n_h += 1
    
    # vert neighbors matching
    n_v = 0
    for i in range(m-1):
        for j in range(n):
            if Z[i,j] == Z[i+1,j]:
                n_v += 1
    
    # X matches Z
    n_obs = np.sum(X == Z)
    
    # stuck row & black pixel
    n_stuck = 0
    for i in range(m):
        if Y[i] == 1:
            n_stuck += np.sum(X[i] == 1)
    
    L = (a**n_black) * (b**n_h) * (g**n_v) * (th**n_obs) * (mu**n_stuck)
    
    return L, {'black': n_black, 'horiz': n_h, 'vert': n_v, 'obs': n_obs, 'stuck': n_stuck}


if __name__ == "__main__":
    X = np.array([[0,0,1], [0,1,1], [0,0,1]])
    Y = np.array([1,0,0])
    Z = np.array([[0,1,1], [1,1,1], [0,0,1]])
    
    # setting a
    La, ca = calc_likelihood(X, Y, Z, a=1.3, b=1.3, g=1.4, th=1.0, mu=0.8)
    print(f"(a): {La:.4f}, counts={ca}")
    
    # setting b
    Lb, cb = calc_likelihood(X, Y, Z, a=0.5, b=1.5, g=1.0, th=0.8, mu=1.2)
    print(f"(b): {Lb:.4f}, counts={cb}")
    
    print(f"\nratio a/b = {La/Lb:.2f}")
    print(f"winner: {'(a)' if La > Lb else '(b)'}")
