import numpy as np
import os
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(SCRIPT_DIR, 'figures'), exist_ok=True)

#load data
data = sio.loadmat(os.path.join(SCRIPT_DIR, 'Xinput.mat'))
X = data['X']  #(50, 10)
n, p = X.shape
print(f"Data: n={n}, p={p}")

#sample covariance
S = X.T @ X / n
print(f"Sample covariance shape: {S.shape}")
print(f"S condition number: {np.linalg.cond(S):.2f}")

def glasso(S, lam, max_iter=500, tol=1e-6):
    """graphical lasso via block coordinate descent (Friedman et al. 2008)"""
    p = S.shape[0]
    #initialize with diagonal + regularization
    W = S.copy() + lam * np.eye(p)
    Theta = np.linalg.inv(W)

    for iteration in range(max_iter):
        W_old = W.copy()
        for j in range(p):
            #partition: W_11 is W without row/col j, s_12 is col j without diagonal
            idx = [i for i in range(p) if i != j]
            W_11 = W[np.ix_(idx, idx)]
            s_12 = S[idx, j]

            #solve lasso subproblem: min 0.5 * beta^T W_11 beta - s_12^T beta + lam * |beta|_1
            #coordinate descent on beta
            beta = np.zeros(p - 1)
            W_11_diag = np.diag(W_11)

            for cd_iter in range(500):
                beta_old = beta.copy()
                for k in range(p - 1):
                    #partial residual
                    r = s_12[k] - W_11[k, :] @ beta + W_11_diag[k] * beta[k]
                    #soft threshold
                    beta[k] = np.sign(r) * max(abs(r) - lam, 0) / W_11_diag[k]
                if np.max(np.abs(beta - beta_old)) < tol:
                    break

            #update W
            w_12 = W_11 @ beta
            W[idx, j] = w_12
            W[j, idx] = w_12

        if np.max(np.abs(W - W_old)) < tol:
            print(f"  converged at iteration {iteration+1}")
            break

    #recover precision matrix
    Theta = np.linalg.inv(W)
    return Theta, W

lambdas = [0, 0.2, 0.5, 0.8]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
results = {}

for idx, lam in enumerate(lambdas):
    print(f"\nlambda = {lam}")
    if lam == 0:
        #no regularization, just invert S (add small ridge for stability)
        Theta = np.linalg.inv(S + 1e-10 * np.eye(p))
    else:
        Theta, W = glasso(S, lam)

    results[lam] = Theta

    #binary matrix: black = nonzero, white = zero
    #threshold small values to zero for display
    thresh = 1e-4
    binary = (np.abs(Theta) > thresh).astype(float)
    nonzero_count = np.sum(binary) - p  #subtract diagonal
    print(f"  nonzero off-diagonal entries: {int(nonzero_count)}")

    ax = axes[idx]
    ax.imshow(1 - binary, cmap='gray', interpolation='nearest')
    ax.set_title(f'$\\lambda = {lam}$', fontsize=14)
    ax.set_xlabel('Variable index')
    if idx == 0:
        ax.set_ylabel('Variable index')
    ax.set_xticks(range(p))
    ax.set_yticks(range(p))

plt.suptitle('Estimated Precision Matrices (black=nonzero, white=zero)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'figures/glasso_precision.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved {os.path.join(SCRIPT_DIR, 'figures/glasso_precision.png')}")

#also print the actual precision matrices for small lambda
print("\nTrue precision (lambda=0) nonzero pattern:")
print((np.abs(results[0]) > 1e-4).astype(int))

print("\nPrecision (lambda=0.8) nonzero pattern:")
print((np.abs(results[0.8]) > 1e-4).astype(int))