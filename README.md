# Probabilistic Graphical Models

Coursework from ECE 7751, a graduate class on probabilistic graphical models. This repo includes my homework code, write-ups with derivations/results, and short READMEs for running each assignment.

Most of the code is written from scratch in NumPy/SciPy. I tried to keep the implementations close to the underlying algorithms rather than relying on PGM libraries, since the main goal was to understand the inference and learning methods directly.

## Topics covered

### HW1: Bayes nets, MRFs, and HMMs
- posterior inference in a small Bayesian network by enumeration
- ICM for binary image denoising with an Ising MRF
- HMM gene tagger with a baseline model and Viterbi decoding

### HW2: exact inference
- variable elimination for marginal queries
- partition function computation for a 20×20 lattice MRF
- forward algorithm for a regime-switching HMM

### HW3: approximate inference
- Gibbs sampling on an Ising lattice with evidence
- rejection sampling baseline for comparison
- Metropolis-Hastings and Gibbs sampling for a Gaussian mixture model
- loopy belief propagation and mean-field inference, checked against exact marginals

### HW4: learning
- graphical lasso with block coordinate descent
- Tree-Augmented Naive Bayes with conditional mutual information and MST structure learning
- EM for a Bayesian network with missing data

