import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

N = 7
num_nodes = N * N

#build adjacency (1 indexed label)
def get_neighbors(label):
    r = (label - 1) // N
    c = (label - 1) % N
    nbrs = []
    if r > 0: nbrs.append((r-1)*N + c + 1)
    if r < N-1: nbrs.append((r+1)*N + c + 1)
    if c > 0: nbrs.append(r*N + c) #c-1 + 1
    if c < N-1: nbrs.append(r*N + c + 2) #c+1 + 1
    return nbrs

#observed nodes
observed = {}
for i in range(2, 25):
    if i % 2 == 0:
        observed[i] = 1
for i in range(26, 49):
    if i % 2 == 0:
        observed[i] = 0

unobserved = [i for i in range(1,50) if i not in observed]

print(f"observed nodes: {len(observed)}, unobserved nodes: {len(unobserved)}")
print(f"observed=1: {[k for k,v in observed.items() if v==1]}")
print(f"observed=0: {[k for k,v in observed.items() if v==0]}")

#Gibbs sampling
def gibbs_full_conditional(node, state, neighbors):
    nbr_vals = [state[n] for n in neighbors]
    #energy for xi=1: sum I[1 == xj] = number of neighbors equal to 1
    #energy for xi=0: sum I[0 == xj] = number of neighbors equal to 0
    e1 = sum(1 for v in nbr_vals if v == 1)
    e0 = sum(1 for v in nbr_vals if v == 0)
    #p(xi=1) = exp(e1) / (exp(e1) + exp(e0))
    #numerical stability
    max_e = max(e1, e0)
    p1 = np.exp(e1 - max_e) / (np.exp(e1 - max_e) + np.exp(e0 - max_e))
    return p1

def run_gibbs(num_samples, burn_in=500):
    #init state: unobserved nodes random
    state = {}
    for k, v in observed.items():
        state[k] = v
    for node in unobserved:
        state[node] = np.random.randint(0, 2)
    
    samples = []
    for t in range(burn_in + num_samples):
        for node in unobserved:
            nbrs = get_neighbors(node)
            p1 = gibbs_full_conditional(node, state, nbrs)
            state[node] = 1 if np.random.rand() < p1 else 0
        
        if t >= burn_in:
            samples.append({k: state[k] for k in unobserved})
    
    return samples

#part (ii) collect 100 samples
samples = run_gibbs(100, burn_in=1000)

x1_samples = [s[1] for s in samples]
x49_samples = [s[49] for s in samples]
x25_samples = [s[25] for s in samples]

print(f"X1:  mean={np.mean(x1_samples):.2f}")
print(f"X49: mean={np.mean(x49_samples):.2f}")
print(f"X25: mean={np.mean(x25_samples):.2f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, data, name in [(axes[0], x1_samples, '$X_1$'),
                        (axes[1], x49_samples, '$X_{49}$'),
                        (axes[2], x25_samples, '$X_{25}$')]:
    ax.hist(data, bins=[-0.25, 0.25, 0.75, 1.25], edgecolor='black',
            rwidth=0.6, color='steelblue')
    ax.set_xticks([0, 1])
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.set_title(f'Histogram of {name} (Gibbs, 100 samples)')

plt.tight_layout()
plt.savefig('/Users/skarjagi6/Library/CloudStorage/Dropbox-GaTech/Shreesh Karjagi/coursework/GRAPH_ML/HWK3/figs/prob3_gibbs_histograms.png', dpi=300, bbox_inches='tight')
plt.close()

#part (iv) rejection sampling

def rejection_sample(num_target, max_tries=2000000):
    #sample from Ising prior, reject if evidence doesn't match
    accepted = []
    total_tries = 0
    while len(accepted) < num_target and total_tries < max_tries:
        total_tries += 1
        #sample from p(x) by Gibbs on the FULL model (no evidence)
        state = {i: np.random.randint(0, 2) for i in range(1, 50)}
        
        #run a few gibbs sweeps on full model
        for _ in range(10):
            for node in range(1, 50):
                nbrs = get_neighbors(node)
                p1 = gibbs_full_conditional(node, state, nbrs)
                state[node] = 1 if np.random.rand() < p1 else 0
        
        #check evidence
        match = all(state[k] == v for k, v in observed.items())
        if match:
            accepted.append(state.copy())
        
        if total_tries % 100000 == 0:
            print(f"  tried {total_tries}, accepted {len(accepted)}")
    
    return accepted, total_tries

#try rejection sampling
rej_samples, total_tries = rejection_sample(1000)

acc_rate = len(rej_samples) / total_tries if total_tries > 0 else 0
print(f"Rejection: accepted {len(rej_samples)} / {total_tries} (rate = {acc_rate:.2e})")

if len(rej_samples) > 0:
    rej_x1 = [s[1] for s in rej_samples]
    p_x1_rej = np.mean(rej_x1)
    print(f"Rejection estimate: P(X1=1 | evidence) = {p_x1_rej:.4f}")
    print(f"Gibbs estimate:     P(X1=1 | evidence) = {np.mean(x1_samples):.4f}")
    print("These should be consistent -- both sample from the same conditional.")
else:
    print("No accepted samples. Rejection sampling is impractical here.")

print("\nDone.")