import numpy as np
def p_win(sA, sB):
    return 1.0 / (1.0 + np.exp(sB - sA))

skills = list(range(1, 11))
posterior = np.zeros((10, 10, 10, 10))

for iA, sA in enumerate(skills):
    for iB, sB in enumerate(skills):
        for iC, sC in enumerate(skills):
            for iD, sD in enumerate(skills):
                likelihood = (
                    p_win(sA, sB) ** 2 *    # A beat B twice
                    p_win(sB, sC) ** 2 *    # B beat C twice
                    p_win(sA, sC) ** 2 *    # A beat C twice
                    p_win(sC, sA) ** 1 *    # C beat A once
                    p_win(sC, sD) ** 2      # C beat D twice (D lost to C twice)
                )
                posterior[iA, iB, iC, iD] = likelihood  # prior is uniform

Z = posterior.sum()
posterior /= Z

#6.2: Are skill levels a posteriori independent?
#see if P(sA, sB | data) = P(sA | data) * P(sB | data)
marginal_A = posterior.sum(axis=(1, 2, 3))
marginal_B = posterior.sum(axis=(0, 2, 3))
marginal_C = posterior.sum(axis=(0, 1, 3))
marginal_D = posterior.sum(axis=(0, 1, 2))

#see joint of A,B vs product of marginals
joint_AB = posterior.sum(axis=(2, 3))  # P(sA, sB | data)
product_AB = np.outer(marginal_A, marginal_B)
max_diff_AB = np.max(np.abs(joint_AB - product_AB))
print(f"Max |P(sA,sB|data) - P(sA|data)P(sB|data)| = {max_diff_AB:.6f}")
print(f"Skill levels are NOT a posteriori independent.\n")

#6.3: P(D beats A | data)
p_D_beats_A = 0.0
for iA, sA in enumerate(skills):
    for iD, sD in enumerate(skills):
        p_D_beats_A += p_win(sD, sA) * posterior.sum(axis=(1, 2))[iA, iD]

#P(D beats A | data) = sum_{sA,sB,sC,sD} P(D beats A | sA, sD) * P(sA,sB,sC,sD | data)
p_D_beats_A = 0.0
for iA, sA in enumerate(skills):
    for iB, sB in enumerate(skills):
        for iC, sC in enumerate(skills):
            for iD, sD in enumerate(skills):
                p_D_beats_A += p_win(sD, sA) * posterior[iA, iB, iC, iD]

print(f"6.3: P(D beats A | data) = {p_D_beats_A:.6f}\n")

#6.4: Posterior expected skill levels
E_sA = sum(sA * marginal_A[iA] for iA, sA in enumerate(skills))
E_sB = sum(sB * marginal_B[iB] for iB, sB in enumerate(skills))
E_sC = sum(sC * marginal_C[iC] for iC, sC in enumerate(skills))
E_sD = sum(sD * marginal_D[iD] for iD, sD in enumerate(skills))

print("6.4: Posterior expected skill levels:")
print(f"  E[sA | data] = {E_sA:.4f}")
print(f"  E[sB | data] = {E_sB:.4f}")
print(f"  E[sC | data] = {E_sC:.4f}")
print(f"  E[sD | data] = {E_sD:.4f}")

print("\nMarginal distributions:")
for name, marg in [("A", marginal_A), ("B", marginal_B), ("C", marginal_C), ("D", marginal_D)]:
    print(f"  P(s{name}=k | data):")
    for k in range(10):
        print(f"    k={k+1}: {marg[k]:.6f}")
