import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(SCRIPT_DIR, 'figures'), exist_ok=True)

#load breast cancer data
raw = []
with open(os.path.join(SCRIPT_DIR, 'breast.csv'), 'r') as f:
    header = f.readline().strip().split(',')
    for line in f:
        line = line.strip()
        if not line:
            continue
        raw.append([int(x) for x in line.split(',')])

D = np.array(raw)
print(f"Data shape: {D.shape}")
print(f"Columns: {header}")

#features are cols 0-8 (binary 0/1), class is col 9 (2=benign, 4=malignant)
features = D[:, :9]
classes = D[:, 9]
#remap class: 2->0 (benign), 4->1 (malignant)
classes = (classes == 4).astype(int)
n_feat = 9
feat_names = header[:9]

print(f"Class distribution: benign={np.sum(classes==0)}, malignant={np.sum(classes==1)}")
print(f"Feature values: min={features.min()}, max={features.max()}")

def empirical_counts(data_feats, data_class, n_feat):
    """compute all needed empirical distributions from training data"""
    N = len(data_class)
    #P(C)
    pc = np.zeros(2)
    for c in range(2):
        pc[c] = np.sum(data_class == c) / N

    #P(Xi, C) for all i
    pxc = np.zeros((n_feat, 2, 2))  #(feat, feat_val, class)
    for i in range(n_feat):
        for xi in range(2):
            for c in range(2):
                pxc[i, xi, c] = np.sum((data_feats[:, i] == xi) & (data_class == c)) / N

    #P(Xi, Xj, C) for all pairs
    pxixjc = np.zeros((n_feat, n_feat, 2, 2, 2))  #(i, j, xi, xj, c)
    for i in range(n_feat):
        for j in range(i+1, n_feat):
            for xi in range(2):
                for xj in range(2):
                    for c in range(2):
                        cnt = np.sum((data_feats[:, i] == xi) & (data_feats[:, j] == xj) & (data_class == c)) / N
                        pxixjc[i, j, xi, xj, c] = cnt
                        pxixjc[j, i, xj, xi, c] = cnt

    return pc, pxc, pxixjc

def conditional_mutual_info(i, j, pc, pxc, pxixjc):
    """I(Xi; Xj | C) = sum_{xi,xj,c} P(xi,xj,c) log P(xi,xj|c) / (P(xi|c) P(xj|c))"""
    cmi = 0.0
    for xi in range(2):
        for xj in range(2):
            for c in range(2):
                p_xixjc = pxixjc[i, j, xi, xj, c]
                if p_xixjc < 1e-12:
                    continue
                #P(xi,xj|c) = P(xi,xj,c) / P(c)
                p_xixj_c = p_xixjc / pc[c] if pc[c] > 0 else 0
                #P(xi|c) = P(xi,c) / P(c)
                p_xi_c = pxc[i, xi, c] / pc[c] if pc[c] > 0 else 0
                p_xj_c = pxc[j, xj, c] / pc[c] if pc[c] > 0 else 0

                if p_xi_c < 1e-12 or p_xj_c < 1e-12 or p_xixj_c < 1e-12:
                    continue
                cmi += p_xixjc * np.log(p_xixj_c / (p_xi_c * p_xj_c))
    return cmi

def learn_tan_structure(train_feats, train_class, n_feat):
    pc, pxc, pxixjc = empirical_counts(train_feats, train_class, n_feat)

    #step 1: compute CMI for all pairs
    weights = np.zeros((n_feat, n_feat))
    for i in range(n_feat):
        for j in range(i+1, n_feat):
            w = conditional_mutual_info(i, j, pc, pxc, pxixjc)
            weights[i, j] = w
            weights[j, i] = w

    #step 2-3: max weight spanning tree (Kruskal's)
    edges = []
    for i in range(n_feat):
        for j in range(i+1, n_feat):
            edges.append((weights[i, j], i, j))
    edges.sort(reverse=True)

    #union-find
    parent = list(range(n_feat))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        if a == b:
            return False
        parent[a] = b
        return True

    tree_edges = []
    adj = defaultdict(list)
    for w, i, j in edges:
        if union(i, j):
            tree_edges.append((i, j, w))
            adj[i].append(j)
            adj[j].append(i)
        if len(tree_edges) == n_feat - 1:
            break

    #step 4: root at node 0, orient edges outward via BFS
    root = 0
    parent_feat = [-1] * n_feat
    visited = [False] * n_feat
    queue = [root]
    visited[root] = True
    while queue:
        node = queue.pop(0)
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                parent_feat[nb] = node
                queue.append(nb)

    return parent_feat, tree_edges, weights

def backoff_smooth(count_xgpa, count_pa, count_x, m, s=5):
    """back-off smoothing: theta = alpha * P_hat(x|pa) + (1-alpha) * P_hat(x)
    alpha = m * P_hat(pa) / (m * P_hat(pa) + s)"""
    p_pa = count_pa / m if m > 0 else 0
    alpha = (m * p_pa) / (m * p_pa + s) if (m * p_pa + s) > 0 else 0
    p_x_given_pa = count_xgpa / count_pa if count_pa > 0 else 0
    p_x = count_x / m if m > 0 else 0
    return alpha * p_x_given_pa + (1 - alpha) * p_x

def train_tan(train_feats, train_class, parent_feat, s=5):
    """learn TAN parameters with back-off smoothing
    TAN parents of Xi: C and parent_feat[i] (if != -1)
    returns CPTs"""
    m = len(train_class)
    n_feat = train_feats.shape[1]

    #P(C)
    pc = np.zeros(2)
    for c in range(2):
        pc[c] = (np.sum(train_class == c) + 1) / (m + 2)  #laplace for prior

    #P(Xi | C, Xpa) with back-off
    #for root feature (parent_feat[i]==-1): P(Xi | C)
    cpts = {}  #cpts[i] = dict mapping (c, xpa) -> P(Xi=1|c, xpa)

    for i in range(n_feat):
        cpts[i] = {}
        pa = parent_feat[i]

        if pa == -1:
            #parents = {C} only (besides naive bayes edges)
            for c in range(2):
                mask_c = train_class == c
                count_c = np.sum(mask_c)
                count_xi1_c = np.sum((train_feats[:, i] == 1) & mask_c)
                count_xi_all = np.sum(train_feats[:, i] == 1)
                #Pa = (C=c), so P_hat(Pa) = count_c / m
                p_pa = count_c / m
                alpha = (m * p_pa) / (m * p_pa + s)
                p_xi1_c = count_xi1_c / count_c if count_c > 0 else 0.5
                p_xi1 = count_xi_all / m
                theta = alpha * p_xi1_c + (1 - alpha) * p_xi1
                theta = np.clip(theta, 1e-10, 1 - 1e-10)
                cpts[i][(c, -1)] = theta  #P(Xi=1 | C=c)
        else:
            #parents = {C, Xpa}
            for c in range(2):
                for xpa in range(2):
                    mask = (train_class == c) & (train_feats[:, pa] == xpa)
                    count_pa = np.sum(mask)
                    count_xi1_pa = np.sum((train_feats[:, i] == 1) & mask)
                    count_xi_all = np.sum(train_feats[:, i] == 1)

                    p_pa_hat = count_pa / m
                    alpha = (m * p_pa_hat) / (m * p_pa_hat + s)
                    p_xi1_given_pa = count_xi1_pa / count_pa if count_pa > 0 else 0.5
                    p_xi1 = count_xi_all / m
                    theta = alpha * p_xi1_given_pa + (1 - alpha) * p_xi1
                    theta = np.clip(theta, 1e-10, 1 - 1e-10)
                    cpts[i][(c, xpa)] = theta  #P(Xi=1 | C=c, Xpa=xpa)

    return pc, cpts

def predict_tan(test_feats, pc, cpts, parent_feat, n_feat):
    """predict class for each test sample"""
    N = test_feats.shape[0]
    preds = np.zeros(N, dtype=int)
    for n in range(N):
        log_probs = np.zeros(2)
        for c in range(2):
            log_p = np.log(pc[c])
            for i in range(n_feat):
                pa = parent_feat[i]
                if pa == -1:
                    key = (c, -1)
                else:
                    key = (c, int(test_feats[n, pa]))
                theta = cpts[i][key]
                xi = int(test_feats[n, i])
                if xi == 1:
                    log_p += np.log(theta)
                else:
                    log_p += np.log(1 - theta)
            log_probs[c] = log_p
        preds[n] = np.argmax(log_probs)
    return preds

def train_nb(train_feats, train_class, s=5):
    """naive bayes with back-off smoothing: all parent_feat = -1"""
    parent_feat = [-1] * train_feats.shape[1]
    return train_tan(train_feats, train_class, parent_feat, s)

#===== run experiments =====
np.random.seed(42)

#withhold 183 records as test set
N = len(classes)
perm = np.random.permutation(N)
test_idx = perm[:183]
remaining_idx = perm[183:]

test_feats = features[test_idx]
test_class = classes[test_idx]

#learn structure on full training data first for the writeup
full_train_feats = features[remaining_idx]
full_train_class = classes[remaining_idx]
parent_feat_full, tree_edges_full, weights_full = learn_tan_structure(full_train_feats, full_train_class, n_feat)

print("\n=== TAN Structure (full training data) ===")
print(f"Feature parent assignments (feature index -> parent feature index):")
for i in range(n_feat):
    pa = parent_feat_full[i]
    if pa == -1:
        print(f"  X{i+1} ({feat_names[i]}): root (no feature parent)")
    else:
        print(f"  X{i+1} ({feat_names[i]}): parent = X{pa+1} ({feat_names[pa]})")

print(f"\nTree edges with CMI weights:")
for i, j, w in sorted(tree_edges_full, key=lambda x: -x[2]):
    print(f"  X{i+1}-X{j+1}: I = {w:.6f}")

#classification experiments
train_sizes = [100, 200, 300, 400, 500]
nb_errors = []
tan_errors = []

for m in train_sizes:
    train_idx = remaining_idx[:m]
    tr_feats = features[train_idx]
    tr_class = classes[train_idx]

    #TAN
    parent_feat_m, _, _ = learn_tan_structure(tr_feats, tr_class, n_feat)
    pc_tan, cpts_tan = train_tan(tr_feats, tr_class, parent_feat_m, s=5)
    preds_tan = predict_tan(test_feats, pc_tan, cpts_tan, parent_feat_m, n_feat)
    err_tan = np.mean(preds_tan != test_class)
    tan_errors.append(err_tan)

    #Naive Bayes
    parent_nb = [-1] * n_feat
    pc_nb, cpts_nb = train_nb(tr_feats, tr_class, s=5)
    preds_nb = predict_tan(test_feats, pc_nb, cpts_nb, parent_nb, n_feat)
    err_nb = np.mean(preds_nb != test_class)
    nb_errors.append(err_nb)

    print(f"m={m}: NB error={err_nb:.4f}, TAN error={err_tan:.4f}")

#plot
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, nb_errors, 'bo-', label='Naive Bayes', linewidth=2)
plt.plot(train_sizes, tan_errors, 'rs-', label='TAN', linewidth=2)
plt.xlabel('Training set size (m)', fontsize=12)
plt.ylabel('Classification error', fontsize=12)
plt.title('NB vs TAN Classification Error on Breast Cancer Data', fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, 'figures/tan_vs_nb_error.png'), dpi=150, bbox_inches='tight')
print(f"\nSaved {os.path.join(SCRIPT_DIR, 'figures/tan_vs_nb_error.png')}")

#print CMI weight matrix
print("\n=== CMI Weight Matrix ===")
for i in range(n_feat):
    row = []
    for j in range(n_feat):
        row.append(f"{weights_full[i,j]:.4f}")
    print(f"  X{i+1}: {' '.join(row)}")