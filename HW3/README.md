# HWK3 Code - Shreesh Karjagi

## Setup

All code is Python 3. You'll need `numpy`, `scipy`, and `matplotlib` installed (`pip install numpy scipy matplotlib`).

## Running the code

### Problem 3
```
python prob3.py
```
Gibbs sampling on 7x7 Ising lattice with evidence. Generates histograms of X1, X49, X25. Also attempts rejection sampling to show it's impractical.

### Problem 4
```
python prob4.py
```
MH sampling (sigma=0.5 and sigma=5) and Gibbs sampling with latent variables for a mixture of Gaussians posterior. Generates scatter plots for all 6 trials of each method.

### Problem 6
```
python prob6.py
```
Loads `pMRF.mat` and computes: (1) loopy BP marginals, (2) mean field marginals, (3) exact marginals by enumeration. Update the path to `pMRF.mat` at the top of the script.
