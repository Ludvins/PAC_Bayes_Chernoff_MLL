import numpy as np
import pylab as p
from matplotlib import pyplot as plt
from toyModel.StatisticalModel import StatisticalModel

# Generate sample data
Q = 10
np.random.seed(2)

# Generate sample data
n = 10000
x_test, y_test = StatisticalModel.generate_sample(n, K=Q, noise=0.1)

n = 100
x_train, y_train = StatisticalModel.generate_sample(n, K=Q, noise=0.1)

bins = [10, 20, 30, 40]
for i in range(len(bins)):
    model = StatisticalModel(bins[i], x_test, y_test, mc_samples=100)


    # Set training data
    model.set_training_data(x_train, y_train)
    print("Bin counts (y=0, y=1):", model.bin_counts)

    model.setAlphaPrior(10)
    # Compute beta distributions

    model.compute_beta_distributions()
    for bin_idx in range(model.K):
        beta_dist = model.posterior_beta_distributions[bin_idx]
        print(f"Bin {bin_idx}: Beta distribution with a={beta_dist.args[0]}, b={beta_dist.args[1]}")

    # Compute KL divergence between posterior and prior
    kl_divergence = model.KLPosteriorPrior()
    print("Sum of KL divergences:", kl_divergence)

    # Compute empirical Gibbs loss
    empirical_loss = model.empiricalGibbsLoss()
    print("Empirical Gibbs Loss (Monte Carlo approximation):", empirical_loss)


    # Compute expected Gibbs loss
    expected_loss = model.expectedGibbsLoss()
    print("Expected Gibbs Loss (Monte Carlo approximation):", expected_loss)


    expected_Gibbs_VarLoss = model.expectedGibbsVarLoss()
    print("Expected Gibbs Var Loss (Monte Carlo approximation):", expected_Gibbs_VarLoss)

    # Compute expected cumulant using Monte Carlo
    lambda_ = 0.0
    expected_cumulant_value = model.expectedCumulant(lambda_)
    print(f"Expected cumulant (Monte Carlo approximation):", expected_cumulant_value)


    #lambdas = np.linspace(0, 0.5, 50)
    #plt.plot(lambdas, [model.expectedCumulant(lambda_) for lambda_ in lambdas], label=bins[i])
    a_vals = np.linspace(0, 3, 10)
    plt.plot(a_vals, [model.inv_rate(a)[0] for a in a_vals], label=bins[i])

plt.title("Cummulant")
plt.ylim(0,1)
plt.legend()
plt.show()



#a_vals = np.linspace(0, 0.3, 10)
#plt.plot(a_vals, [model.rate(a)[0] for a in a_vals])
#plt.title("Rate")
#plt.show()


#a_vals = np.linspace(0, 3, 10)
#plt.plot(a_vals, [model.inv_rate(a)[0] for a in a_vals])
#plt.title("Inv_rate")
#plt.show()