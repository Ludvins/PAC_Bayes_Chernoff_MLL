import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import logsumexp
from tqdm import tqdm
from toyModel.StatisticalModel import StatisticalModel

# Parameters. Try alpha=10, alpha=0.01
alpha = 10
model_bins = [2, 10, 30]
data_bins = range(2, 100, 3)
n_test_samples = 10000
n_train_samples = 100
num_training_samples = 100  # Number of training samples to average over

# Function to generate test data samples
def generate_test_samples(binQ, noise=0.1):
    x_test, y_test = StatisticalModel.generate_sample(n_test_samples, K=binQ, noise=noise)
    return x_test, y_test

# Function to generate a single training data sample
def generate_training_sample(binQ, noise=0.1):
    x_train, y_train = StatisticalModel.generate_average_sample(n_train_samples, K=binQ, noise=noise)
    return x_train, y_train

# Function to compute log likelihood values averaged over multiple training samples
def compute_log_mll(binK):
    vals = []
    for binQ in tqdm(data_bins, desc=f'Computing logMLL for K={binK}'):
        x_test, y_test = generate_test_samples(binQ)
        log_mll_values = []
        for _ in range(num_training_samples):
            x_train, y_train = generate_training_sample(binQ)
            model = StatisticalModel(binK, x_test, y_test)
            model.set_training_data(x_train, y_train)
            model.setAlphaPrior(alpha)
            model.compute_beta_distributions()
            log_mll_values.append(model.logMLL())
        # Compute average log MLL
        avg_log_mll = logsumexp(log_mll_values) - np.log(num_training_samples)
        vals.append(avg_log_mll)
    # Proceed with normalization as before
    vals = -np.array(vals)  # Change sign to make log probs
    log_norm = logsumexp(vals)  # Normalization constant
    return np.exp(vals - log_norm)

# Function to compute expected BMA loss averaged over multiple training samples
def compute_expected_bma_loss(binK):
    vals = []
    for binQ in tqdm(data_bins, desc=f'Computing Expected BMA Loss for K={binK}'):
        x_test, y_test = generate_test_samples(binQ)
        bma_loss_values = []
        for _ in range(num_training_samples):
            x_train, y_train = generate_training_sample(binQ)
            model = StatisticalModel(binK, x_test, y_test)
            model.set_training_data(x_train, y_train)
            model.setAlphaPrior(alpha)
            model.compute_beta_distributions()
            bma_loss_values.append(model.expectedBMALoss())
        # Compute average expected BMA loss
        avg_bma_loss = np.mean(bma_loss_values)
        vals.append(avg_bma_loss)
    return vals

# Plot logMLL values
plt.figure(figsize=(10, 5))
for binK in model_bins:
    vals = compute_log_mll(binK)
    plt.plot(data_bins, vals, label=f'K={binK}')
plt.xlabel('Data Bins (Q)')
plt.ylabel('P(D|M)')
plt.legend()
plt.title(f'P(D|M) (Alpha = {alpha})')
plt.show()

# Plot expected BMA loss values
plt.figure(figsize=(10, 5))
for binK in model_bins:
    vals = compute_expected_bma_loss(binK)
    plt.plot(data_bins, vals, label=f'K={binK}')
plt.xlabel('Data Bins (Q)')
plt.ylabel('Expected BMA Loss')
plt.legend()
plt.title(f'Expected BMA Loss (Alpha = {alpha})')
plt.show()
