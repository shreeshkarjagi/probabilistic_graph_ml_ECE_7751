# HWK1 Code - Shreesh Karjagi

## Setup

All code is Python 3. You'll need `numpy`, `scipy`, and `matplotlib` installed (`pip install numpy scipy matplotlib`).

## Running the code

### Problem 5 (Part 3)
```
python prob5_part3.py
```
Computes the unnormalized likelihoods for both parameter settings and prints which one is larger. No input files needed.

### Problem 6
```
python prob6.py
```
Enumerates all skill level combinations and computes the posterior. Prints the independence check (6.2), P(D beats A) (6.3), and expected skill levels (6.4). No input files needed.

### Problem 7
```
python prob7.py
```
Loads `hw1-Prob7_images.mat` and runs ICM denoising with three parameter settings. Saves `prob7_denoising.png` (comparison of all settings) and `prob7_best.png` (best result only). You'll need to update the file paths at the top of the script to point to wherever you put the `.mat` file.

### Problem 8
```
python prob8_hmm.py
```
Runs the baseline tagger (Part 8.1) and Viterbi tagger (Part 8.2). Expects `gene.train`, `gene.test`, and `gene.key` in the data directory. Again, update `DATA_DIR` at the top of the script to match your local setup. Outputs `gene_test.p1.out` (baseline) and `gene_test.p2.out` (Viterbi) in the same directory, then prints precision/recall/F1 for both.