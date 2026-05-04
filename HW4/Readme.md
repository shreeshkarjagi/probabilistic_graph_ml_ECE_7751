# HWK4 Code - Shreesh Karjagi

## Setup
All code is Python 3. You'll need `numpy`, `scipy`, and `matplotlib` installed (`pip install numpy scipy matplotlib`).

Data files (`Xinput.mat`, `breast.csv`, `EMprinter.mat`) should be in the same directory as the scripts. Figures are saved to a `figures/` subfolder created automatically.

## Running the code

### Problem 3 (Graphical Lasso)
```
python prob3_glasso.py
```
Implements graphical lasso via block coordinate descent (Friedman et al. 2008). Loads `Xinput.mat`, computes the sample covariance, and estimates sparse precision matrices for lambda = 0, 0.2, 0.5, 0.8. Outputs a side-by-side sparsity pattern plot to `figures/glasso_precision.png` and prints nonzero off-diagonal counts.

### Problem 4 (TAN vs Naive Bayes)
```
python prob4_tan.py
```
Learns a Tree-Augmented Naive Bayes classifier on the breast cancer dataset (`breast.csv`). Computes conditional mutual information for all feature pairs, builds a max-weight spanning tree (Kruskal's), and orients it from a root via BFS. Parameters estimated with back-off smoothing. Compares TAN vs Naive Bayes classification error across training sizes m = 100, 200, 300, 400, 500 on a held-out test set of 183 samples. Saves the error plot to `figures/tan_vs_nb_error.png`.

### Problem 5 (EM for Printer Diagnosis)
```
python prob5_em_printer.py
```
Runs EM on a Bayesian network for printer fault diagnosis with missing data (`EMprinter.mat`). The BN has 10 variables (5 root causes, 5 symptoms) with structure hardcoded from the assignment figure. Missing values are handled by enumerating all completions in the E-step. Runs 5 random restarts and keeps the best log-likelihood. Prints learned CPTs and answers the query P(Drum=1 | Wrinkled=0, Burning=0, Quality=1) by exact enumeration over all free variables.
