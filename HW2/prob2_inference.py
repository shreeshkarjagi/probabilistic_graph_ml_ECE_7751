
import numpy as np
from collections import defaultdict
import time

def load_joint(filepath):
    joint = {}
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            joint[int(parts[0])] = float(parts[1])
    return joint

def load_dataset(filepath):
    samples = []
    with open(filepath, 'r') as f:
        for line in f:
            samples.append(int(line.strip()))
    return samples

def int_to_bits(n, num_bits=12):
    return [(n >> i) & 1 for i in range(num_bits)]
parents = {
    0: [], # IsSummer
    1: [0], # HasFlu
    2: [0], # HasFoodPoisoning
    3: [0], # HasHayFever
    4: [0], # HasPneumonia
    5: [1, 3, 4], # HasRespiratoryProblems
    6: [1, 2, 4], # HasGastricProblems
    7: [1, 2, 3], # HasRash
    8: [1, 2, 3, 4], # Coughs
    9: [1, 2, 3, 4], # IsFatigued
    10: [1, 2, 3, 4],# Vomits
    11: [1, 2, 3, 4] # HasFever
}

def estimate_parameters(samples):
    num_vars = 12
    assignment_counts = defaultdict(int)
    for s in samples:
        assignment_counts[s] += 1

    cpds = {}
    for var in range(num_vars):
        par = parents[var]
        counts = defaultdict(lambda: [0, 0])
        for s, cnt in assignment_counts.items():
            bits = int_to_bits(s, num_vars)
            parent_vals = tuple(bits[p] for p in par)
            counts[parent_vals][bits[var]] += cnt

        cpd = {}
        for parent_vals, c in counts.items():
            total = c[0] + c[1]
            cpd[parent_vals] = [c[0]/total, c[1]/total] if total > 0 else [0.5, 0.5]
        cpds[var] = cpd
    return cpds

def compute_model_joint(cpds):
    model_joint = {}
    for a in range(4096):
        bits = int_to_bits(a, 12)
        prob = 1.0
        for var in range(12):
            pv = tuple(bits[p] for p in parents[var])
            prob *= cpds[var].get(pv, [0.5, 0.5])[bits[var]]
        model_joint[a] = prob
    return model_joint

class Factor:
    def __init__(self, variables, values):
        self.variables = sorted(variables)
        self.values = values

def multiply_factors(f1, f2):
    all_vars = sorted(set(f1.variables + f2.variables))
    new_values = {}
    for ai in range(2**len(all_vars)):
        asgn = {v: (ai >> i) & 1 for i, v in enumerate(all_vars)}
        k1 = tuple(asgn[v] for v in f1.variables)
        k2 = tuple(asgn[v] for v in f2.variables)
        kn = tuple(asgn[v] for v in all_vars)
        new_values[kn] = f1.values.get(k1, 0) * f2.values.get(k2, 0)
    return Factor(all_vars, new_values)

def sum_out(factor, var):
    if var not in factor.variables:
        return factor
    new_vars = [v for v in factor.variables if v != var]
    new_values = defaultdict(float)
    for key, val in factor.values.items():
        new_key = tuple(k for i, k in enumerate(key) if factor.variables[i] != var)
        new_values[new_key] += val
    return Factor(new_vars, dict(new_values))

def create_cpd_factor(var, cpds):
    par = parents[var]
    all_vars = sorted(par + [var])
    values = {}
    for ai in range(2**len(all_vars)):
        asgn = {v: (ai >> i) & 1 for i, v in enumerate(all_vars)}
        pv = tuple(asgn[p] for p in par)
        values[tuple(asgn[v] for v in all_vars)] = cpds[var].get(pv, [0.5,0.5])[asgn[var]]
    return Factor(all_vars, values)

def variable_elimination(cpds, query_vars, evidence):
    factors = [create_cpd_factor(var, cpds) for var in range(12)]

    for evar, eval_ in evidence.items():
        new_factors = []
        for f in factors:
            if evar in f.variables:
                nv = {}
                new_vars = [v for v in f.variables if v != evar]
                for key, val in f.values.items():
                    if key[f.variables.index(evar)] == eval_:
                        nk = tuple(k for i, k in enumerate(key) if f.variables[i] != evar)
                        nv[nk] = val
                new_factors.append(Factor(new_vars, nv))
            else:
                new_factors.append(f)
        factors = new_factors
    elim_order = [v for v in range(12) if v not in query_vars and v not in evidence]
    for var in elim_order:
        involved = [f for f in factors if var in f.variables]
        remaining = [f for f in factors if var not in f.variables]
        if not involved:
            continue
        product = involved[0]
        for f in involved[1:]:
            product = multiply_factors(product, f)
        factors = remaining + [sum_out(product, var)]

    result = factors[0]
    for f in factors[1:]:
        result = multiply_factors(result, f)

    total = sum(result.values.values())
    if total > 0:
        result.values = {k: v/total for k, v in result.values.items()}
    return result

def query_true_joint(joint, query_vars, evidence):
    result = defaultdict(float)
    for a, prob in joint.items():
        bits = int_to_bits(a, 12)
        if all(bits[ev] == val for ev, val in evidence.items()):
            result[tuple(bits[qv] for qv in query_vars)] += prob
    total = sum(result.values())
    return {k: v/total for k, v in result.items()} if total > 0 else result

def get_bit(sample, var):
    return (sample >> var) & 1

if __name__ == "__main__":
    data_dir = '/Users/skarjagi6/Library/CloudStorage/Dropbox-GaTech/Shreesh Karjagi/coursework/GRAPH_ML/HWK2/Data-HWK2/Data-Problem-2'
    joint = load_joint(f'{data_dir}/joint.dat')
    samples = load_dataset(f'{data_dir}/dataset.dat')
    print(f"Loaded joint.dat: {len(joint)} entries, sum={sum(joint.values()):.6f}")
    print(f"Loaded dataset.dat: {len(samples)} samples")

   #part c
    cpds = estimate_parameters(samples)
    model_joint = compute_model_joint(cpds)

   #part d
    l1 = sum(abs(joint.get(i,0) - model_joint.get(i,0)) for i in range(4096))
    ll_true = sum(np.log(max(joint.get(s,1e-30),1e-30)) for s in samples)
    ll_model = sum(np.log(max(model_joint.get(s,1e-30),1e-30)) for s in samples)
    print(f"\nL1 distance: {l1:.6f}")
    print(f"Log-lik (true): {ll_true:.2f} (avg {ll_true/len(samples):.6f})")
    print(f"Log-lik (model): {ll_model:.2f} (avg {ll_model/len(samples):.6f})")

   #part e
    queries = [
        ("Q1: P(HasFlu | HasFever=T, Coughs=T)", [1], {11:1, 8:1}),
        ("Q2: P(HasRash,Coughs,Fatigued,Vomits,Fever | Pneumonia=T)",
         [7,8,9,10,11], {4:1}),
        ("Q3: P(Vomits | IsSummer=T)", [10], {0:1}),
    ]

    for name, qvars, ev in queries:
        print(f"\n{name}")
        true_r = query_true_joint(joint, qvars, ev)
        t0 = time.time()
        ve_r = variable_elimination(cpds, qvars, ev)
        t_ve = time.time() - t0
        print(f"True joint: {dict(sorted(true_r.items()))}")
        print(f"VE model:   {dict(sorted(ve_r.values.items()))}")
        print(f"VE time: {t_ve:.4f}s")
    
    samples_arr = np.array(samples)

    t0 = time.time()
    mask_q1 = (samples_arr >> 11 & 1 == 1) & (samples_arr >> 8 & 1 == 1)
    matching = samples_arr[mask_q1]
    flu1 = np.mean((matching >> 1) & 1)
    t_sc1 = time.time() - t0
    print(f"\nQ1 sample counting: P(HasFlu=0)={1-flu1:.4f}, P(HasFlu=1)={flu1:.4f}, time: {t_sc1:.2f}s")

    t0 = time.time()
    mask_q3 = (samples_arr >> 0 & 1 == 1)
    matching3 = samples_arr[mask_q3]
    vom1 = np.mean((matching3 >> 10) & 1)
    t_sc3 = time.time() - t0
    print(f"Q3 sample counting: P(Vomits=0)={1-vom1:.4f}, P(Vomits=1)={vom1:.4f}, time: {t_sc3:.2f}s")
