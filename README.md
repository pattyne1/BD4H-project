# Estimating Individual Treatment Effects with Time-Varying Confounders
This repository contains code to reproduce the results from the paper "Estimating Individual Treatment Effects with Time-Varying Confounders" by Ruoqi Liu, Changchang Yin, and Ping Zhang for the synthetic dataset experiments.

## Dependencies
The code requires the following Python packages:
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- tqdm

You can install the required packages using pip. 

## Accessing the Data
The synthetic dataset is generated within the code using the following parameters:

- Timesteps (T): 30
- Number of covariates (k): 100
- Number of static features (k_s): 5
- Hidden dimension (h): 1
- Number of samples (N): 4000
- Number of treated samples (N_treated): 1000

The data generation process follows the procedure described in the paper to simulate:

- Treatment assignments (A)
- Time-varying covariates (X)
- Static covariates (X_static)
- Hidden confounders (Z)
- Factual and counterfactual outcomes (Y)

## Functionality of scripts 
The code consists of several key components:

1. Data Generation:
- Simulates synthetic time-series data with treatment effects
- Creates train/validation/test splits

2. Model Architecture:
- Implements attention-based GRU network
- Handles both factual and counterfactual predictions
- Includes IPW (Inverse Probability Weighting) components

3. Training:
- Implements the training loop with both IPW and outcome losses
- Includes validation monitoring and model checkpointing

4. Evaluation:
- Computes key metrics: PEHE, ATE, RMSE
- Provides model performance analysis

## Instructions to run the code

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```
2. Create active virtual enviroment (reccomended):
``` bash
conda create -n treatment_effects python=3.8
conda activate treatment_effects
```

2. Install requirements: 
```bash
pip install torch numpy pandas scikit-learn tqdm
```
4. Run the code:

``` bash
python main.py --observation_window 12 --epochs 30 --batch-size 128
```

### Directory Structure
- `data_synthetic/`: Generated synthetic data
- `checkpoints/`: Saved models
- `main.py`: Main code file
