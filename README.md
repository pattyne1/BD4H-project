# Estimating Individual Treatment Effects with Time-Varying Confounders
This repository contains code to reproduce the results from the paper "Estimating Individual Treatment Effects with Time-Varying Confounders" by Ruoqi Liu, Changchang Yin, and Ping Zhang for the synthetic dataset experiments.

## Dependencies
The code requires the following Python packages:
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- tqdm
- Matplotlib

You can install the required packages using pip. 

## Accessing the Data
The synthetic dataset is generated within the code using the following parameters:

Timesteps (T): 30
Number of covariates (k): 100
Number of static features (k_s): 5
Hidden dimension (h): 1
Number of samples (N): 4000
Number of treated samples (N_treated): 1000

The data generation process follows the procedure described in the paper to simulate:

Treatment assignments (A)
Time-varying covariates (X)
Static covariates (X_static)
Hidden confounders (Z)
Factual and counterfactual outcomes (Y)


3. Functionality of scripts 
4. Instructions to run the code
