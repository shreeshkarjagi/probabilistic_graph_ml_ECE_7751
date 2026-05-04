import numpy as np
import os
import scipy.io as sio

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

#load data: states 1,2,nan -> 0,1,missing
data = sio.loadmat(os.path.join(SCRIPT_DIR, 'EMprinter.mat'))
X = data['x']  #shape (10, 15): 10 variables, 15 datapoints
print(f"Data shape: {X.shape}")
print(f"Variables x Datapoints")

#remap: 1->0, 2->1, nan->-1
D = np.full_like(X, -1, dtype=float)
D[X == 1] = 0
D[X == 2] = 1
#nan stays as -1
D[np.isnan(X)] = -1
N = D.shape[1]  #15 datapoints
print(f"N = {N} datapoints")

#BN structure (from figure):
#0: Fuse (root)
#1: Drum (root)
#2: Toner (root)
#3: Paper (root)
#4: Roller (root)
#5: Burning -> parents: {Fuse}
#6: Quality -> parents: {Fuse, Drum, Toner}
#7: Wrinkled -> parents: {Fuse, Drum, Paper}
#8: MultPages -> parents: {Paper, Roller}
#9: PaperJam -> parents: {Drum, Toner, Paper, Roller}

var_names = ['Fuse', 'Drum', 'Toner', 'Paper', 'Roller',
             'Burning', 'Quality', 'Wrinkled', 'MultPages', 'PaperJam']

parents = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [0],
    6: [0, 1, 2],
    7: [0, 1, 3],
    8: [3, 4],
    9: [1, 2, 3, 4],
}

n_vars = 10

def init_cpts():
    """initialize CPTs randomly but valid"""
    cpts = {}
    for v in range(n_vars):
        pa = parents[v]
        n_pa_configs = 2 ** len(pa)
        #P(v=1 | pa_config) for each parent configuration
        #initialize near 0.5 with small noise
        cpts[v] = 0.3 + 0.4 * np.random.rand(n_pa_configs)
    return cpts

def pa_config_index(assignment, pa_list):
    """convert parent assignment to index (binary encoding)"""
    idx = 0
    for k, p in enumerate(pa_list):
        idx += int(assignment[p]) * (2 ** k)
    return idx

def log_prob_sample(assignment, cpts):
    """log P(assignment) given CPTs. assignment is length-10 array of 0/1"""
    lp = 0.0
    for v in range(n_vars):
        pa = parents[v]
        pa_idx = pa_config_index(assignment, pa)
        theta = cpts[v][pa_idx]  #P(v=1|pa)
        if assignment[v] == 1:
            lp += np.log(theta + 1e-300)
        else:
            lp += np.log(1 - theta + 1e-300)
    return lp

def em_learn(D, max_iter=200, tol=1e-6, seed=0):
    """EM algorithm for BN with missing data"""
    np.random.seed(seed)
    cpts = init_cpts()
    N = D.shape[1]

    for iteration in range(max_iter):
        #E-step: for each datapoint, compute posterior over missing variables
        #by enumerating all completions of missing values
        expected_counts = {}
        for v in range(n_vars):
            pa = parents[v]
            n_pa_configs = 2 ** len(pa)
            #counts[pa_config, v_val]
            expected_counts[v] = np.zeros((n_pa_configs, 2))

        total_ll = 0.0

        for n in range(N):
            obs = D[:, n]
            missing = np.where(obs == -1)[0]
            observed = np.where(obs != -1)[0]
            n_missing = len(missing)

            if n_missing == 0:
                #fully observed
                assignment = obs.astype(int)
                lp = log_prob_sample(assignment, cpts)
                total_ll += lp
                for v in range(n_vars):
                    pa = parents[v]
                    pa_idx = pa_config_index(assignment, pa)
                    expected_counts[v][pa_idx, assignment[v]] += 1.0
            else:
                #enumerate all 2^n_missing completions
                log_probs = []
                completions = []
                for combo in range(2 ** n_missing):
                    assignment = obs.copy().astype(int)
                    for k, m_idx in enumerate(missing):
                        assignment[m_idx] = (combo >> k) & 1
                    lp = log_prob_sample(assignment, cpts)
                    log_probs.append(lp)
                    completions.append(assignment.copy())

                #normalize
                log_probs = np.array(log_probs)
                max_lp = np.max(log_probs)
                weights = np.exp(log_probs - max_lp)
                weights /= np.sum(weights)
                total_ll += max_lp + np.log(np.sum(np.exp(log_probs - max_lp)))

                for w, assignment in zip(weights, completions):
                    for v in range(n_vars):
                        pa = parents[v]
                        pa_idx = pa_config_index(assignment, pa)
                        expected_counts[v][pa_idx, assignment[v]] += w

        #M-step: update CPTs from expected counts
        old_cpts = {v: cpts[v].copy() for v in range(n_vars)}
        for v in range(n_vars):
            n_pa_configs = expected_counts[v].shape[0]
            for pa_idx in range(n_pa_configs):
                total = expected_counts[v][pa_idx, 0] + expected_counts[v][pa_idx, 1]
                if total > 1e-10:
                    cpts[v][pa_idx] = expected_counts[v][pa_idx, 1] / total
                else:
                    cpts[v][pa_idx] = 0.5

        #check convergence
        max_change = 0
        for v in range(n_vars):
            max_change = max(max_change, np.max(np.abs(cpts[v] - old_cpts[v])))

        if iteration % 20 == 0 or max_change < tol:
            print(f"  iter {iteration}: LL={total_ll:.4f}, max_change={max_change:.8f}")

        if max_change < tol:
            print(f"  converged at iteration {iteration}")
            break

    return cpts, total_ll

#run EM with multiple restarts
best_ll = -np.inf
best_cpts = None
n_restarts = 5

for seed in range(n_restarts):
    print(f"\n--- EM restart {seed+1} ---")
    cpts, ll = em_learn(D, max_iter=500, tol=1e-8, seed=seed)
    print(f"  final LL = {ll:.4f}")
    if ll > best_ll:
        best_ll = ll
        best_cpts = cpts

print(f"\nBest LL across restarts: {best_ll:.4f}")

#print learned CPTs
print("\n=== Learned CPTs ===")
for v in range(n_vars):
    pa = parents[v]
    print(f"\n{var_names[v]} (parents: {[var_names[p] for p in pa]}):")
    n_configs = 2 ** len(pa)
    for pa_idx in range(n_configs):
        #decode parent config
        pa_vals = {}
        for k, p in enumerate(pa):
            pa_vals[var_names[p]] = (pa_idx >> k) & 1
        theta = best_cpts[v][pa_idx]
        if len(pa) == 0:
            print(f"  P({var_names[v]}=1) = {theta:.4f}")
        else:
            pa_str = ', '.join([f"{k}={v}" for k, v in pa_vals.items()])
            print(f"  P({var_names[v]}=1 | {pa_str}) = {theta:.4f}")

#query: P(Drum=1 | Wrinkled=0, Burning=0, Quality=1)
#evidence: var 7=0, var 5=0, var 6=1
print("\n=== Query: P(Drum=1 | Wrinkled=0, Burning=0, Quality=1) ===")

evidence = {5: 0, 6: 1, 7: 0}  #burning=0, quality=1 (poor), wrinkled=0
query_var = 1  #drum

#enumerate all non-evidence variables' values
non_evidence = [v for v in range(n_vars) if v not in evidence]
n_free = len(non_evidence)

log_probs_drum0 = []
log_probs_drum1 = []

for combo in range(2 ** n_free):
    assignment = np.zeros(n_vars, dtype=int)
    #set evidence
    for v, val in evidence.items():
        assignment[v] = val
    #set free variables
    for k, v in enumerate(non_evidence):
        assignment[v] = (combo >> k) & 1

    lp = log_prob_sample(assignment, best_cpts)

    if assignment[query_var] == 0:
        log_probs_drum0.append(lp)
    else:
        log_probs_drum1.append(lp)

#compute P(drum=1 | evidence)
log_probs_drum0 = np.array(log_probs_drum0)
log_probs_drum1 = np.array(log_probs_drum1)

max_all = max(np.max(log_probs_drum0), np.max(log_probs_drum1))
p0 = np.sum(np.exp(log_probs_drum0 - max_all))
p1 = np.sum(np.exp(log_probs_drum1 - max_all))

prob_drum1 = p1 / (p0 + p1)
prob_drum0 = p0 / (p0 + p1)

print(f"P(Drum=0 | evidence) = {prob_drum0:.4f}")
print(f"P(Drum=1 | evidence) = {prob_drum1:.4f}")