import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#part (a) generate data
np.random.seed(0)
N = 100
components = np.random.binomial(1, 0.5, N)
data = np.where(components == 0,
                np.random.normal(-5, 1, N),
                np.random.normal(5, 1, N))
print(f"Data n={N}, mean={data.mean():.2f}, std={data.std():.2f}")

#log unnormalized posterior
def log_posterior(mu1, mu2, x):
    #prior N(0, 100) for each
    lp = -mu1**2 / 200.0 - mu2**2 / 200.0
    #likelihood  product of 0.5*N(xi|mu1,1) + 0.5*N(xi|mu2,1)
    for xi in x:
        l1 = -0.5*(xi - mu1)**2
        l2 = -0.5*(xi - mu2)**2
        mx = max(l1, l2)
        lp += mx + np.log(0.5*np.exp(l1 - mx) + 0.5*np.exp(l2 - mx))
    return lp

#part (b) Metropolis Hastings
def run_mh(x, sigma_prop, n_burn=10000, n_samples=1000, seed=0):
    np.random.seed(seed)
    mu1, mu2 = 0.0, 0.0
    lp_curr = log_posterior(mu1, mu2, x)
    
    total = n_burn + n_samples
    accepts = 0
    samples = []
    
    for t in range(total):
        mu1_prop = mu1 + np.random.normal(0, sigma_prop)
        mu2_prop = mu2 + np.random.normal(0, sigma_prop)
        lp_prop = log_posterior(mu1_prop, mu2_prop, x)
        
        log_alpha = lp_prop - lp_curr
        if np.log(np.random.rand()) < log_alpha:
            mu1, mu2 = mu1_prop, mu2_prop
            lp_curr = lp_prop
            accepts += 1
        
        if t >= n_burn:
            samples.append((mu1, mu2))
    
    acc_rate = accepts / total
    return np.array(samples), acc_rate

for sigma in [0.5, 5.0]:
    print(f"\nsigma = {sigma}")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    for trial in range(6):
        samples, acc_rate = run_mh(data, sigma, seed=trial*10)
        mu1_est = samples[:, 0].mean()
        mu2_est = samples[:, 1].mean()
        print(f"trial {trial+1} acc={acc_rate:.3f}, "
              f"E[mu1]={mu1_est:.2f}, E[mu2]={mu2_est:.2f}")
        
        r, c = trial // 3, trial % 3
        axes[r, c].scatter(samples[:, 0], samples[:, 1], s=6, alpha=0.5)
        axes[r, c].set_xlabel('$\\mu_1$')
        axes[r, c].set_ylabel('$\\mu_2$')
        axes[r, c].set_title(f'Trial {trial+1}, acc={acc_rate:.2f}')
        axes[r, c].set_xlim(-10, 10)
        axes[r, c].set_ylim(-10, 10)
    
    plt.suptitle(f'MH samples ($\\sigma={sigma}$)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'/Users/skarjagi6/Library/CloudStorage/Dropbox-GaTech/Shreesh Karjagi/coursework/GRAPH_ML/HWK3/figs/prob4_mh_sigma{sigma}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

#part (c) Gibbs sampling with latent variables
def run_gibbs(x, n_burn=10000, n_samples=1000, seed=0):
    np.random.seed(seed)
    N = len(x)
    mu1, mu2 = 0.0, 0.0
    z = np.random.randint(0, 2, N) #latent component indicators
    
    samples = []
    total = n_burn + n_samples
    
    for t in range(total):
        #sample z_i | mu1, mu2, x_i
        for i in range(N):
            l1 = -0.5*(x[i] - mu1)**2
            l2 = -0.5*(x[i] - mu2)**2
            mx = max(l1, l2)
            p1 = np.exp(l1 - mx) / (np.exp(l1 - mx) + np.exp(l2 - mx))
            z[i] = 0 if np.random.rand() < p1 else 1
        
        #sample mu1 | z, x (conjugate Gaussian)
        #prior N(0, 100), likelihood prod N(xi|mu1,1) for zi=0
        idx1 = (z == 0)
        n1 = idx1.sum()
        if n1 > 0:
            xbar1 = x[idx1].mean()
        else:
            xbar1 = 0.0
        #posterior precision = n1/1 + 1/100 = n1 + 0.01
        prec1 = n1 + 0.01
        post_mean1 = (n1 * xbar1) / prec1
        post_var1 = 1.0 / prec1
        mu1 = np.random.normal(post_mean1, np.sqrt(post_var1))
        
        #sample mu2 | z, x
        idx2 = (z == 1)
        n2 = idx2.sum()
        if n2 > 0:
            xbar2 = x[idx2].mean()
        else:
            xbar2 = 0.0
        prec2 = n2 + 0.01
        post_mean2 = (n2 * xbar2) / prec2
        post_var2 = 1.0 / prec2
        mu2 = np.random.normal(post_mean2, np.sqrt(post_var2))
        
        if t >= n_burn:
            samples.append((mu1, mu2))
    
    return np.array(samples)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for trial in range(6):
    samples = run_gibbs(data, seed=trial*10)
    mu1_est = samples[:, 0].mean()
    mu2_est = samples[:, 1].mean()
    print(f"trial {trial+1} E[mu1]={mu1_est:.2f}, E[mu2]={mu2_est:.2f}")
    
    r, c = trial // 3, trial % 3
    axes[r, c].scatter(samples[:, 0], samples[:, 1], s=6, alpha=0.5)
    axes[r, c].set_xlabel('$\\mu_1$')
    axes[r, c].set_ylabel('$\\mu_2$')
    axes[r, c].set_title(f'Trial {trial+1}')
    axes[r, c].set_xlim(-10, 10)
    axes[r, c].set_ylim(-10, 10)

plt.suptitle('Gibbs samples', fontsize=14)
plt.tight_layout()
plt.savefig('/Users/skarjagi6/Library/CloudStorage/Dropbox-GaTech/Shreesh Karjagi/coursework/GRAPH_ML/HWK3/figs/prob4_gibbs.png', dpi=300, bbox_inches='tight')
plt.close()

