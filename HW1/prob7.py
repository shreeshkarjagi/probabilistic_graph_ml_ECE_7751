"""
Problem 7: Image Denoising using Markov Network with ICM (Iterated Conditional Modes)

Energy function: E(Z=z, X=x) = h * sum_i(z_i) - beta * sum_{i,j}(z_i * z_j) - nu * sum_i(z_i * x_i)

where z_i, x_i in {+1, -1}, beta > 0, nu > 0, h in R.

We want to minimize E (maximize probability P propto exp(-E)).

ICM: For each pixel, flip to the value that gives lower energy.
Local energy contribution for z_i:
  E_local(z_i) = h * z_i - beta * z_i * sum_{j in neighbors(i)} z_j - nu * z_i * x_i

Pick z_i = +1 if E_local(+1) < E_local(-1), else z_i = -1.

E_local(+1) - E_local(-1) = 2h - 2*beta*S - 2*nu*x_i
where S = sum of neighboring z_j values.

So z_i = +1 if h < beta*S + nu*x_i, else z_i = -1.
Equivalently, z_i = sign(beta*S + nu*x_i - h).  (ties go to +1 or -1, doesn't matter much)
"""
import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load data
data = scipy.io.loadmat('/mnt/user-data/uploads/hw1-Prob7_images.mat')
origImg = data['origImg'].astype(np.float64)   # shape (400, 700), values in {-1, +1}
noisyImg = data['noisyImg'].astype(np.float64) # shape (400, 700), values in {-1, +1}

rows, cols = origImg.shape
print(f"Image size: {rows} x {cols}")
print(f"Noise rate: {np.mean(origImg != noisyImg):.4f}")


def compute_energy(Z, X, h, beta, nu):
    """Compute the full energy E(Z, X)."""
    E = h * np.sum(Z)
    
    # Horizontal neighbors
    E -= beta * np.sum(Z[:, :-1] * Z[:, 1:])
    # Vertical neighbors
    E -= beta * np.sum(Z[:-1, :] * Z[1:, :])
    # Observation term
    E -= nu * np.sum(Z * X)
    
    return E


def denoise_icm(X, h, beta, nu, max_iters=20):
    """
    Run ICM (Iterated Conditional Modes) to denoise image.
    
    For each pixel z_i, pick the value in {+1, -1} that minimizes local energy.
    """
    Z = X.copy()  # Initialize Z to noisy values
    
    for iteration in range(max_iters):
        num_flipped = 0
        energy = compute_energy(Z, X, h, beta, nu)
        
        for i in range(rows):
            for j in range(cols):
                # Sum of neighbors
                S = 0.0
                if i > 0:       S += Z[i-1, j]
                if i < rows-1:  S += Z[i+1, j]
                if j > 0:       S += Z[i, j-1]
                if j < cols-1:  S += Z[i, j+1]
                
                # Local energy for z_i = +1:  h - beta*S - nu*x_i
                # Local energy for z_i = -1: -h + beta*S + nu*x_i
                # Pick +1 if h - beta*S - nu*x_i < -h + beta*S + nu*x_i
                # i.e., if beta*S + nu*X[i,j] - h > 0
                
                val = beta * S + nu * X[i, j] - h
                new_z = 1.0 if val > 0 else -1.0
                
                if new_z != Z[i, j]:
                    num_flipped += 1
                    Z[i, j] = new_z
        
        new_energy = compute_energy(Z, X, h, beta, nu)
        print(f"  Iter {iteration+1}: flipped {num_flipped} pixels, energy = {new_energy:.2f}")
        
        if num_flipped == 0:
            print(f"  Converged at iteration {iteration+1}")
            break
    
    return Z


def error_rate(Z, orig):
    """Fraction of pixels recovered incorrectly."""
    return np.mean(Z != orig)


# -------------------------------------------------------------------
# Three parameter settings
# -------------------------------------------------------------------
settings = [
    {"h": 0.0, "beta": 1.0, "nu": 2.0,  "label": "Setting 1 (h=0, β=1.0, ν=2.0)"},
    {"h": 0.0, "beta": 0.1, "nu": 1.0,  "label": "Setting 2 (h=0, β=0.1, ν=1.0)"},
    {"h": 0.0, "beta": 2.0, "nu": 1.0,  "label": "Setting 3 (h=0, β=2.0, ν=1.0)"},
]

results = []

for idx, s in enumerate(settings):
    print(f"\n{'='*60}")
    print(f"{s['label']}")
    print(f"{'='*60}")
    
    Z = denoise_icm(noisyImg, s['h'], s['beta'], s['nu'], max_iters=15)
    err = error_rate(Z, origImg)
    results.append((s['label'], err, Z.copy()))
    print(f"Error rate: {err:.6f} ({err*100:.2f}%)")

# Find best result
best_idx = np.argmin([r[1] for r in results])
print(f"\n{'='*60}")
print(f"BEST: {results[best_idx][0]} with error rate {results[best_idx][1]:.6f}")
print(f"{'='*60}")

# Noisy error rate for reference
noisy_err = error_rate(noisyImg, origImg)
print(f"Noisy image error rate (baseline): {noisy_err:.6f} ({noisy_err*100:.2f}%)")

# -------------------------------------------------------------------
# Generate figures
# -------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Row 1: Original, Noisy, Best Denoised
axes[0, 0].imshow(origImg, cmap='gray', vmin=-1, vmax=1)
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(noisyImg, cmap='gray', vmin=-1, vmax=1)
axes[0, 1].set_title(f'Noisy Image (error={noisy_err:.4f})')
axes[0, 1].axis('off')

axes[0, 2].imshow(results[best_idx][2], cmap='gray', vmin=-1, vmax=1)
axes[0, 2].set_title(f'Best Denoised\n{results[best_idx][0]}\nerror={results[best_idx][1]:.6f}')
axes[0, 2].axis('off')

# Row 2: All three denoised results
for i, (label, err, Z) in enumerate(results):
    axes[1, i].imshow(Z, cmap='gray', vmin=-1, vmax=1)
    axes[1, i].set_title(f'{label}\nerror={err:.6f}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('/home/claude/code/prob7_denoising.png', dpi=150, bbox_inches='tight')
plt.close()

# Save just the best denoised image
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
ax2.imshow(results[best_idx][2], cmap='gray', vmin=-1, vmax=1)
ax2.set_title(f'Best Denoised Image: {results[best_idx][0]}, error = {results[best_idx][1]:.6f}')
ax2.axis('off')
plt.savefig('/home/claude/code/prob7_best.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nFigures saved.")
