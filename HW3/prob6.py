import numpy as np
import scipy.io

#load the pMRF data
data = scipy.io.loadmat('Data-Problem-6/pMRF.mat')
phi_raw = data['phi'][0]

#parse potential functions
#phi_raw[k] is a tuple (variables_array, table_array)
potentials = []
for k in range(5):
    varr = phi_raw[k][0].flatten() #variables
    table = phi_raw[k][1]  #2x2 table           
    potentials.append({'vars': tuple(varr), 'table': table})
    print(f"phi({k+1}): vars={tuple(varr)}, table=\n{table}")

def phi_val(pot, v1, v2):
    #get potential value for variable assignments v1, v2 in {0,1}
    return pot['table'][v1, v2]

#marginals by enumeration
#enumerate all 2^4 = 16 configurations
exact_joint = np.zeros((2, 2, 2, 2))

for x1 in range(2):
    for x2 in range(2):
        for x3 in range(2):
            for x4 in range(2):
                val = 1.0
                for pot in potentials:
                    v = pot['vars']
                    assign = {1: x1, 2: x2, 3: x3, 4: x4}
                    val *= phi_val(pot, assign[v[0]], assign[v[1]])
                exact_joint[x1, x2, x3, x4] = val

Z = exact_joint.sum()
exact_joint /= Z
print(f"Z = {Z:.6f}")

exact_marginals = []
for i in range(4):
    axes_to_sum = tuple(j for j in range(4) if j != i)
    marg = exact_joint.sum(axis=axes_to_sum)
    exact_marginals.append(marg)
    print(f"p(x{i+1}=1) = {marg[0]:.6f}, p(x{i+1}=2) = {marg[1]:.6f}")

#part 1 loopy belief propagation/factor graph formalism

#init all to uniform
max_iters = 200
tol = 1e-8

#msg_v2f[(var, factor_idx)] = array of size 2
#msg_f2v[(factor_idx, var)] = array of size 2
msg_v2f = {}
msg_f2v = {}

for fi, pot in enumerate(potentials):
    for v in pot['vars']:
        msg_v2f[(v, fi)] = np.ones(2) / 2.0
        msg_f2v[(fi, v)] = np.ones(2) / 2.0

for iteration in range(max_iters):
    old_f2v = {k: v.copy() for k, v in msg_f2v.items()}
    #update factor-->variable messages
    for fi, pot in enumerate(potentials):
        v1, v2 = pot['vars']
        #message from factor fi to v1 sum over v2 of (pot * msg_v2f[v2-->fi])
        incoming_v2 = msg_v2f[(v2, fi)]
        new_msg = np.zeros(2)
        for a in range(2):
            for b in range(2):
                new_msg[a] += pot['table'][a, b] * incoming_v2[b]
        new_msg /= new_msg.sum()
        msg_f2v[(fi, v1)] = new_msg
        #message from factor fi to v2 sum over v1 of (pot * msg_v2f[v1-->fi])
        incoming_v1 = msg_v2f[(v1, fi)]
        new_msg = np.zeros(2)
        for b in range(2):
            for a in range(2):
                new_msg[b] += pot['table'][a, b] * incoming_v1[a]
        new_msg /= new_msg.sum()
        msg_f2v[(fi, v2)] = new_msg
    #update variable-->factor messages
    for fi, pot in enumerate(potentials):
        for v in pot['vars']:
            #msg from v to fi = product of all incoming f2v msg except fi
            msg = np.ones(2)
            for fj, pot2 in enumerate(potentials):
                if fj == fi:
                    continue
                if v in pot2['vars']:
                    msg *= msg_f2v[(fj, v)]
            msg /= msg.sum()
            msg_v2f[(v, fi)] = msg
    #check convergence
    max_diff = 0
    for k in msg_f2v:
        max_diff = max(max_diff, np.max(np.abs(msg_f2v[k] - old_f2v[k])))
    
    if max_diff < tol:
        print(f"BP converged at iteration {iteration+1}")
        break

if max_diff >= tol:
    print(f"BP did not converge after {max_iters} iters (max_diff={max_diff:.2e})")

#compute BP beliefs
bp_marginals = []
for vi in range(1, 5):
    belief = np.ones(2)
    for fi, pot in enumerate(potentials):
        if vi in pot['vars']:
            belief *= msg_f2v[(fi, vi)]
    belief /= belief.sum()
    bp_marginals.append(belief)
    print(f"q_BP(x{vi}=1) = {belief[0]:.6f}, q_BP(x{vi}=2) = {belief[1]:.6f}")

#part 2 mean field 

q = [np.ones(2) / 2.0 for _ in range(4)]  #q[0] = q(x1), etc.

for iteration in range(500):
    old_q = [qi.copy() for qi in q]
    
    for vi in range(4): #update q[vi] for variable vi+1
        log_q = np.zeros(2)
        
        for fi, pot in enumerate(potentials):
            v1, v2 = pot['vars']
            if vi + 1 == v1:
                #factor involves vi+1 as first var other var is v2
                other_idx = v2 - 1  #0 indexed
                for a in range(2):
                    for b in range(2):
                        log_q[a] += q[other_idx][b] * np.log(pot['table'][a, b] + 1e-300)
            elif vi + 1 == v2:
                #factor involves vi+1 as second var, other var is v1
                other_idx = v1 - 1
                for b in range(2):
                    for a in range(2):
                        log_q[b] += q[other_idx][a] * np.log(pot['table'][a, b] + 1e-300)
        
        #normalize
        log_q -= log_q.max()
        q[vi] = np.exp(log_q)
        q[vi] /= q[vi].sum()
    
    #check converge
    max_diff = max(np.max(np.abs(q[i] - old_q[i])) for i in range(4))
    if max_diff < 1e-10:
        print(f"MF converged at iteration {iteration+1}")
        break

if max_diff >= 1e-10:
    print(f"MF did not converge after 500 iters (max_diff={max_diff:.2e})")

mf_marginals = []
for vi in range(4):
    mf_marginals.append(q[vi])
    print(f"q_MF(x{vi+1}=1) = {q[vi][0]:.6f}, q_MF(x{vi+1}=2) = {q[vi][1]:.6f}")

#part 4 compare
def mean_deviation(approx_marginals, exact_marginals):
    total = 0.0
    for i in range(4):
        for j in range(2):
            total += abs(approx_marginals[i][j] - exact_marginals[i][j])
    return total / (4 * 2)

bp_dev = mean_deviation(bp_marginals, exact_marginals)
mf_dev = mean_deviation(mf_marginals, exact_marginals)

print(f"BP mean deviation: {bp_dev:.6f}")
print(f"MF  mean deviation: {mf_dev:.6f}")

print("\n more comparison:")
print(f"{'Var':>4} {'Val':>4} {'Exact':>10} {'BP':>10} {'MF':>10}")
for i in range(4):
    for j in range(2):
        print(f"x{i+1:>2}  {j+1:>3}  {exact_marginals[i][j]:>10.6f} "
              f"{bp_marginals[i][j]:>10.6f} {mf_marginals[i][j]:>10.6f}")

print("\nDone.")
