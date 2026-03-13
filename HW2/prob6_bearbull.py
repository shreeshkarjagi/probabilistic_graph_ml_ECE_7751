import numpy as np
import scipy.io

data_dir = '/Users/skarjagi6/Library/CloudStorage/Dropbox-GaTech/Shreesh Karjagi/coursework/GRAPH_ML/HWK2/Data-Problem-6/BearBullproblem.mat'
data = scipy.io.loadmat(data_dir)
pbull = data['pbull'] #100x100: pbull[new, old]
pbear = data['pbear'] #100x100: pbear[new, old]
prices = data['p'].flatten()  # 200 observed prices

T = len(prices)
print(f"T = {T}, price range: [{prices.min()}, {prices.max()}]")
print(f"Last price: price(T={T}) = {prices[-1]}")

#market state 0=bear, 1=bull
#market_trans[new_state, old_state]
market_trans = np.array([[0.8, 0.3],   #P(bear|bear), P(bear|bull)
                         [0.2, 0.7]])  #P(bull|bear), P(bull|bull)

price_trans = {0: pbear, 1: pbull}

#Forward algorithm
#alpha[s] = P(prices(1:t), market(t)=s), normalized at each step
alpha = np.array([0.5 * (1.0/100), 0.5 * (1.0/100)]) #prior * P(price_1)

for t in range(1, T):
    new_alpha = np.zeros(2)
    cur_idx = prices[t] - 1
    prev_idx = prices[t-1] - 1
    for s_new in range(2):
        for s_old in range(2):
            new_alpha[s_new] += (alpha[s_old]
                * market_trans[s_new, s_old]
                * price_trans[s_old][cur_idx, prev_idx])
    alpha = new_alpha / new_alpha.sum()  # normalize

print(f"\nPosterior at T: P(bear)={alpha[0]:.6f}, P(bull)={alpha[1]:.6f}")

#predict price(T+1)
last_idx = prices[-1] - 1
price_dist = np.zeros(100)
for s in range(2):
    price_dist += alpha[s] * price_trans[s][:, last_idx]
price_dist /= price_dist.sum()

E_price = sum((p+1) * price_dist[p] for p in range(100))
E_price_sq = sum(((p+1)**2) * price_dist[p] for p in range(100))
var_price = E_price_sq - E_price**2

expected_gain = E_price - prices[-1]
std_gain = np.sqrt(var_price)

print(f"\nE[price(T+1)] = {E_price:.4f}")
print(f"Expected gain = E[price(T+1)] - price(T) = {expected_gain:.4f}")
print(f"Std(gain) = {std_gain:.4f}")
