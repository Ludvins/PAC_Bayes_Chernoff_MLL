import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.special
from sklearn.metrics import mean_squared_error
from src.plot import save_plot_with_multiple_extensions  # If available in your environment

# Set seed for reproducibility.
np.random.seed(0)

###############################################################################
# Hyperparameters for our Bayesian approach
###############################################################################
sigma2 = 0.0001  # Noise variance (known, fixed)
alpha2 = 5   # Prior variance for the weights

###############################################################################
# Data/experiment setup (mirroring the original script structure)
###############################################################################
num_data_list = [15]
num_features_list = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 30, 40, 50, 100, 200,
]
num_repeat_list = list(range(5))

results_dir = "results/gibbs_polynomial_regression"
os.makedirs(results_dir, exist_ok=True)

###############################################################################
# True function for data generation
###############################################################################
def compute_y_from_x(X: np.ndarray):
    """
    True function: y = 2*x + cos(25*x)
    """
    return np.add(2.0 * X, np.cos(X * 25))[:, 0]


###############################################################################
# Main script
###############################################################################
low, high = -1.0, 1.0

for num_data in num_data_list:
    mse_list = []
    results_num_data_dir = os.path.join(results_dir, f"num_data={num_data}")
    os.makedirs(results_num_data_dir, exist_ok=True)

    # Generate test data and true function outputs
    X_test = np.linspace(start=low, stop=high, num=1000).reshape(-1, 1)
    y_test = compute_y_from_x(X_test)

    # Optional: plot just the true function
    plt.close()
    sns.lineplot(x=X_test[:, 0], y=y_test, label="True Function")
    plt.xlabel("x")
    plt.ylabel("y")
    for extension in ["pdf", "png"]:
        plt.savefig(
            os.path.join(results_num_data_dir, f"data.{extension}"),
            bbox_inches="tight",
            dpi=300,
        )
    plt.close()

    # Loop over polynomial feature dimensionalities
    for num_features in num_features_list:
        results_num_features_dir = os.path.join(
            results_num_data_dir, f"num_features={num_features}"
        )
        os.makedirs(results_num_features_dir, exist_ok=True)

        # Degrees for Legendre polynomials: [1, 2, 3, ..., num_features]
        feature_degrees = 1 + np.arange(num_features).astype(int)

        for repeat_idx in num_repeat_list:
            # Sample training data
            X_train = np.random.uniform(low=low, high=high, size=(num_data, 1))
            y_train = compute_y_from_x(X_train)

            # Construct design matrices using Legendre polynomials
            X_train_poly = scipy.special.eval_legendre(feature_degrees, X_train)
            X_test_poly = scipy.special.eval_legendre(feature_degrees, X_test)

            ###############################################################################
            # Bayesian Conjugate Update (Gaussian prior, Gaussian likelihood)
            #
            # Posterior for w:
            #   Posterior precision:  (1/sigma2) * (X^T X) + (1/alpha2) * I
            #   Posterior mean:       mu_post = Cov_post * (1/sigma2) * X^T y
            #   Posterior covariance: Cov_post = [ (1/sigma2)*X^T X + (1/alpha2)*I ]^-1
            ###############################################################################
            n_features = X_train_poly.shape[1]
            I = np.eye(n_features)

            posterior_precision = (1.0 / sigma2) * (X_train_poly.T @ X_train_poly) + (1.0 / alpha2) * I
            posterior_cov = np.linalg.inv(posterior_precision)
            posterior_mean = (posterior_cov @ (X_train_poly.T @ y_train)) * (1.0 / sigma2)

            # Posterior predictive mean for train/test
            y_train_pred = X_train_poly @ posterior_mean
            y_test_pred  = X_test_poly  @ posterior_mean

            # Save posterior parameters for KL divergence calculation
            np.savez(
                os.path.join(results_num_features_dir, f"posterior_params_{repeat_idx}.npz"),
                mean=posterior_mean,
                cov=posterior_cov
            )

            # ------------------------------------------------
            # Standard MSE with the posterior-mean predictor
            # ------------------------------------------------
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse  = mean_squared_error(y_test,  y_test_pred)

            # ------------------------------------------------
            # GIBBS MSE:
            # E_{w ~ p(w)}[(y - x^T w)^2] = (y - x^T mu_post)^2 + x^T Cov_post x
            # We'll average over all training (resp. test) points.
            # ------------------------------------------------
            # 1) (y_i - x_i^T mu_post)^2 part:
            train_residual_sqd = (y_train - y_train_pred) ** 2
            test_residual_sqd  = (y_test  - y_test_pred ) ** 2

            # 2) x^T Cov_post x part:
            #    A convenient way is sum((X_train_poly @ posterior_cov) * X_train_poly, axis=1)
            train_post_var = np.sum((X_train_poly @ posterior_cov) * X_train_poly, axis=1)
            test_post_var  = np.sum((X_test_poly  @ posterior_cov) * X_test_poly,  axis=1)

            # Summation
            train_gibbs_mse = np.mean(train_residual_sqd + train_post_var)
            test_gibbs_mse  = np.mean(test_residual_sqd  + test_post_var)

            # Collect stats
            mse_list.append(
                {
                    "Num. Data": num_data,
                    "Num. Parameters (Num Features)": num_features,
                    "repeat_idx": repeat_idx,
                    "Train MSE": train_mse,
                    "Test MSE": test_mse,
                    "Train Gibbs MSE": train_gibbs_mse,
                    "Test Gibbs MSE": test_gibbs_mse,
                }
            )

            print(
                f"[Bayesian] num_data={num_data}, num_features={num_features}, "
                f"repeat_idx={repeat_idx}, train_mse={train_mse:.4f}, test_mse={test_mse:.4f}, "
                f"train_gibbs={train_gibbs_mse:.4f}, test_gibbs={test_gibbs_mse:.4f}"
            )

            # Plot the posterior mean fit
            plt.close()
            sns.lineplot(x=X_test[:, 0], y=y_test, label="True Function")
            sns.lineplot(x=X_test[:, 0], y=y_test_pred, label="Posterior Mean")
            sns.scatterplot(x=X_train[:, 0], y=y_train, s=30, color="k", label="Train Data")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.ylim(-3, 3)
            for extension in ["pdf", "png"]:
                plt.savefig(
                    os.path.join(
                        results_num_features_dir, f"repeat_idx={repeat_idx}.{extension}"
                    ),
                    bbox_inches="tight",
                    dpi=300,
                )
            plt.close()

    # --------------------------------------------------------
    # Save results and produce final MSE vs #features plot
    # --------------------------------------------------------
    mse_df = pd.DataFrame(mse_list)
    mse_df.to_csv(os.path.join(results_num_data_dir, "mse.csv"), index=False)

    plt.close()
    fig, ax = plt.subplots()

    # Plot standard MSE lines
    sns.lineplot(
        data=mse_df,
        x="Num. Parameters (Num Features)",
        y="Train MSE",
        label="Train (Mean)",
        ax=ax,
    )
    sns.lineplot(
        data=mse_df,
        x="Num. Parameters (Num Features)",
        y="Test MSE",
        label="Test (Mean)",
        ax=ax,
    )

    # Plot Gibbs MSE lines
    sns.lineplot(
        data=mse_df,
        x="Num. Parameters (Num Features)",
        y="Train Gibbs MSE",
        label="Train (Gibbs)",
        ax=ax,
    )
    sns.lineplot(
        data=mse_df,
        x="Num. Parameters (Num Features)",
        y="Test Gibbs MSE",
        label="Test (Gibbs)",
        ax=ax,
    )

    ax.set_ylabel("Mean Squared Error")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(bottom=1e-1)
    ax.set_title("Bayesian Polynomial Regression (Conjugate)")

    ax.axvline(
        x=num_data,
        color="black",
        linestyle="--",
        label="Interpolation Threshold",
    )
    ax.legend()

    if "save_plot_with_multiple_extensions" in globals():
        save_plot_with_multiple_extensions(
            plot_dir=results_num_data_dir,
            plot_title=f"mse_num_data={num_data}"
        )
    else:
        plt.savefig(
            os.path.join(results_num_data_dir, f"mse_num_data={num_data}.png"),
            bbox_inches="tight",
            dpi=300,
        )
    plt.close()

    # --------------------------------------------------------
    # Calculate and plot KL divergence between posterior and prior
    # --------------------------------------------------------
    kl_divergences = []
    for num_features in num_features_list:
        # Get all results for this number of features
        feature_results = mse_df[mse_df["Num. Parameters (Num Features)"] == num_features]
        
        # Calculate average KL divergence across repeats
        avg_kl = 0
        for repeat_idx in range(len(feature_results)):
            # Get the posterior mean and covariance for this repeat
            results_num_features_dir = os.path.join(
                results_num_data_dir, f"num_features={num_features}"
            )
            
            # Load the posterior parameters from the saved results
            posterior_params = np.load(os.path.join(results_num_features_dir, f"posterior_params_{repeat_idx}.npz"))
            posterior_mean = posterior_params['mean']
            posterior_cov = posterior_params['cov']
            
            # Prior parameters
            prior_mean = np.zeros(num_features)
            prior_cov = alpha2 * np.eye(num_features)
            
            # Calculate KL divergence
            k = num_features
            kl_div = 0.5 * (
                np.trace(np.linalg.inv(posterior_cov) @ prior_cov) +
                (posterior_mean - prior_mean).T @ np.linalg.inv(posterior_cov) @ (posterior_mean - prior_mean) -
                k +
                np.log(np.linalg.det(posterior_cov) / np.linalg.det(prior_cov))
            )
            avg_kl += kl_div
        
        avg_kl /= len(feature_results)
        kl_divergences.append({
            "Num. Parameters (Num Features)": num_features,
            "KL Divergence": avg_kl
        })

    # Create and save KL divergence plot
    plt.close()
    fig, ax = plt.subplots()
    
    kl_df = pd.DataFrame(kl_divergences)
    sns.lineplot(
        data=kl_df,
        x="Num. Parameters (Num Features)",
        y="KL Divergence",
        ax=ax
    )
    
    ax.set_ylabel("KL Divergence (Posterior || Prior)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title("Evolution of KL Divergence with Model Complexity")
    
    ax.axvline(
        x=num_data,
        color="black",
        linestyle="--",
        label="Interpolation Threshold",
    )
    ax.legend()
    
    if "save_plot_with_multiple_extensions" in globals():
        save_plot_with_multiple_extensions(
            plot_dir=results_num_data_dir,
            plot_title=f"kl_divergence_num_data={num_data}"
        )
    else:
        plt.savefig(
            os.path.join(results_num_data_dir, f"kl_divergence_num_data={num_data}.png"),
            bbox_inches="tight",
            dpi=300,
        )
    plt.close()

    # --------------------------------------------------------
    # Calculate and plot the sum of Train Gibbs MSE and KL Divergence
    # --------------------------------------------------------
    # Create a DataFrame combining MSE and KL divergence data
    combined_data = []
    for num_features in num_features_list:
        # Get MSE data for this number of features
        mse_data = mse_df[mse_df["Num. Parameters (Num Features)"] == num_features]
        # Get KL divergence for this number of features
        kl_data = kl_df[kl_df["Num. Parameters (Num Features)"] == num_features]
        
        # Calculate the sum of average Train Gibbs MSE and KL divergence
        avg_train_gibbs_mse = mse_data["Train Gibbs MSE"].mean()
        kl_div = kl_data["KL Divergence"].iloc[0]
        combined_sum = avg_train_gibbs_mse + kl_div
        
        combined_data.append({
            "Num. Parameters (Num Features)": num_features,
            "Combined Sum": combined_sum
        })

    # Create and save the combined plot
    plt.close()
    fig, ax = plt.subplots()
    
    combined_df = pd.DataFrame(combined_data)
    sns.lineplot(
        data=combined_df,
        x="Num. Parameters (Num Features)",
        y="Combined Sum",
        ax=ax
    )
    
    ax.set_ylabel("Train Gibbs MSE + KL Divergence")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_title("Evolution of Combined Train Gibbs MSE and KL Divergence")
    
    ax.axvline(
        x=num_data,
        color="black",
        linestyle="--",
        label="Interpolation Threshold",
    )
    ax.legend()
    
    if "save_plot_with_multiple_extensions" in globals():
        save_plot_with_multiple_extensions(
            plot_dir=results_num_data_dir,
            plot_title=f"combined_gibbs_kl_num_data={num_data}"
        )
    else:
        plt.savefig(
            os.path.join(results_num_data_dir, f"combined_gibbs_kl_num_data={num_data}.png"),
            bbox_inches="tight",
            dpi=300,
        )
    plt.close()
