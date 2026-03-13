import numpy as np
import time

def compute_log_partition_naive(n):
    num_states = 2**n
    def intra(s):
        e = 0
        for i in range(n-1):
            if ((s>>i)&1) == ((s>>(i+1))&1): e += 1
        return e
    def inter(s1, s2):
        e = 0
        for i in range(n):
            if ((s1>>i)&1) == ((s2>>i)&1): e += 1
        return e

    log_msg = np.array([float(intra(s)) for s in range(num_states)])
    for col in range(1, n):
        new_log_msg = np.full(num_states, -np.inf)
        for s2 in range(num_states):
            terms = np.array([log_msg[s1] + inter(s1, s2) for s1 in range(num_states)])
            mx = np.max(terms)
            new_log_msg[s2] = mx + np.log(np.sum(np.exp(terms - mx))) + intra(s2)
        log_msg = new_log_msg
    mx = np.max(log_msg)
    return mx + np.log(np.sum(np.exp(log_msg - mx)))

def compute_log_partition_butterfly(n, verbose=False):
    """
    butterfly transfer method, horizontal transfer factors as n independent 2x2 transfers
    so matrix-vector product is O(n * 2^n) instead of O(2^{2n})
    """
    num_states = 2**n
    e_same = np.exp(1.0) # phi(a,b)=e when a==b
    e_diff = 1.0 # phi(a,b)=1 when a!=b

    intra_log = np.zeros(num_states)
    for s in range(num_states):
        for i in range(n - 1):
            if ((s >> i) & 1) == ((s >> (i + 1)) & 1):
                intra_log[s] += 1.0

    log_msg = intra_log.copy()

    for col in range(1, n):
        if verbose and col % 5 == 0:
            print(f"  Column {col+1}/{n}")
        max_log = np.max(log_msg)
        msg = np.exp(log_msg - max_log)

        #apply n independent 2x2 transfers
        for i in range(n):
            temp = msg.copy()
            mask = 1 << i
            for s in range(num_states):
                if s & mask == 0:
                    s1 = s | mask
                    msg[s]  = temp[s] * e_same + temp[s1] * e_diff
                    msg[s1] = temp[s] * e_diff + temp[s1] * e_same

        log_msg = np.log(msg + 1e-300) + max_log + intra_log

    mx = np.max(log_msg)
    return mx + np.log(np.sum(np.exp(log_msg - mx)))

if __name__ == "__main__":
    print("Validation (naive vs butterfly):")
    for test_n in [2, 3, 4, 5]:
        ln = compute_log_partition_naive(test_n)
        lb = compute_log_partition_butterfly(test_n)
        print(f"  n={test_n}: naive={ln:.6f}, butterfly={lb:.6f}, diff={abs(ln-lb):.2e}")

    #n=20
    print(f"\nComputing log(Z) for n=20")
    t0 = time.time()
    logZ = compute_log_partition_butterfly(20, verbose=True)
    print(f"\nlog(Z) for n=20 = {logZ:.6f}")
    print(f"Time: {time.time()-t0:.1f}s")
