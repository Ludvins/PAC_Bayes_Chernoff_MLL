import numpy as np
import pandas as pd
from tqdm import tqdm

from toyModel.StatisticalModel import StatisticalModel


# Generate sample data
Q = 10
np.random.seed(2)

# Generate sample data
n = 10000
x_test, y_test = StatisticalModel.generate_sample(n, K=Q, noise=0.1)

n = 100
x_train, y_train = StatisticalModel.generate_sample(n, K=Q, noise=0.1)

# Define values of alpha_prior to test
alpha_prior_values = [0.01, 10]

bins = np.arange(2, 50, 2)

# Define lists to collect results for each method
results = {
    'n': [],
    'K': [],
    'alpha_prior': [],
    'PAC_Chernoff': [],
    'PAC_ChernoffSubGaussian': [],
    'expectedBMALoss': [],
    'logMLL': [],
    'empiricalGibbsLoss': [],
    'KLPosteriorPrior': [],
    'expectedGibbsVarLoss': [],
    'expectedGibbsLoss': [],
    'PAC_ChernoffFull': [],
    'PAC_ChernoffSubGaussianFull': [],
}


# Set the parameter lambda_ for expectedCumulant and a for rate/inv_rate
lambda_ = 1.0
a_value = 1.0

# Iterate over different values of alpha_prior
for alpha in tqdm(alpha_prior_values, desc="Alpha Prior Iteration"):
    for i in tqdm(bins, desc=f"Processing bins for alpha_prior={alpha}", leave=False):
        model = StatisticalModel(i, x_test, y_test)
        model.set_training_data(x_train, y_train)
        model.setAlphaPrior(alpha)  # Set current alpha_prior
        model.compute_beta_distributions()

        # Append results of each method to the dictionary
        results['n'].append(n)
        results['K'].append(i)
        results['alpha_prior'].append(alpha)
        results['PAC_Chernoff'].append(model.PAC_Chernoff()[0])
        results['PAC_ChernoffSubGaussian'].append(model.PAC_ChernoffSubGaussian())
        results['PAC_ChernoffFull'].append(model.PAC_ChernoffFull()[0])
        results['PAC_ChernoffSubGaussianFull'].append(model.PAC_ChernoffSubGaussianFull())
        results['expectedBMALoss'].append(model.expectedBMALoss())
        results['logMLL'].append(model.logMLL())
        results['empiricalGibbsLoss'].append(model.empiricalGibbsLoss())
        results['KLPosteriorPrior'].append(model.KLPosteriorPrior() / n)
        results['expectedGibbsVarLoss'].append(model.expectedGibbsVarLoss())
        results['expectedGibbsLoss'].append(model.expectedGibbsLoss())

# Convert results dictionary to DataFrame
df_results = pd.DataFrame(results)

# Display the DataFrame
df_results.head()

# Save DataFrame to a CSV file
df_results.to_csv('statistical_model_results.csv', index=False)
