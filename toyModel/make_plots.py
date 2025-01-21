import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


def plot_metric_scatter(df, metric_x, metric_y, alpha_prior_values):
    """
    Generate scatter plots for two metrics in the DataFrame, grouped by each alpha_prior value,
    with each subplot having its own y-axis range adjusted to the data.
    Points are colored based on 'K' values, with high 'K' being more blue and low 'K' more green.

    Parameters:
    - df: pd.DataFrame - The DataFrame containing the data.
    - metric_x: str - The name of the metric to be displayed on the x-axis.
    - metric_y: str - The name of the metric to be displayed on the y-axis.
    - alpha_prior_values: list - List of alpha_prior values to filter and plot separately.
    """
    fig, axes = plt.subplots(1, len(alpha_prior_values), figsize=(6 * len(alpha_prior_values), 6),
                             constrained_layout=True)

    # Get the min and max values of K for color normalization
    k_min = df['K'].min()
    k_max = df['K'].max()

    for i, alpha in enumerate(alpha_prior_values):
        df_alpha = df[df['alpha_prior'] == alpha]
        correlation = df_alpha[metric_x].corr(df_alpha[metric_y])  # Calculate correlation

        # Normalize the K values to the range [0, 1] for colormap
        norm_k = (df_alpha['K'] - k_min) / (k_max - k_min)

        # Scatter plot with color mapped to normalized K values
        scatter = axes[i].scatter(df_alpha[metric_x], df_alpha[metric_y],
                                  c=norm_k, cmap='winter', alpha=0.7)

        # Calculate range for y-axis with a 10% padding
        ymin = df_alpha[metric_y].min()
        ymax = df_alpha[metric_y].max()
        padding = (ymax - ymin) * 0.1
        axes[i].set_ylim([ymin - padding, ymax + padding])

        # Calculate range for x-axis with a 10% padding
        xmin = df_alpha[metric_x].min()
        xmax = df_alpha[metric_x].max()
        padding = (xmax - xmin) * 0.1
        axes[i].set_xlim([xmin - padding, xmax + padding])

        axes[i].set_title(f'Alpha Prior = {alpha}\nCorrelation: {correlation:.2f}')
        axes[i].set_xlabel(metric_x)
        axes[i].set_ylabel(metric_y)
        axes[i].grid(True)

    # Add a color bar for the K values
    cbar = fig.colorbar(cm.ScalarMappable(norm=plt.Normalize(vmin=k_min, vmax=k_max), cmap='winter'),
                        ax=axes, orientation='vertical', label='K Value')

    plt.suptitle(f'Comparison of {metric_x} vs {metric_y} for Different Alpha Prior Values')
    plt.show()

df_results = pd.read_csv('statistical_model_results.csv')
alpha_prior_values = [0.01, 10]

# Example usage:
plot_metric_scatter(df_results, 'K', 'expectedBMALoss', alpha_prior_values)
plot_metric_scatter(df_results, 'K', 'empiricalGibbsLoss', alpha_prior_values)
plot_metric_scatter(df_results, 'K', 'KLPosteriorPrior', alpha_prior_values)
plot_metric_scatter(df_results, 'K', 'expectedGibbsVarLoss', alpha_prior_values)

plot_metric_scatter(df_results, 'expectedGibbsLoss', 'expectedBMALoss', alpha_prior_values)


plot_metric_scatter(df_results, 'expectedBMALoss', 'logMLL', alpha_prior_values)
plot_metric_scatter(df_results, 'expectedBMALoss', 'PAC_Chernoff', alpha_prior_values)
plot_metric_scatter(df_results, 'expectedBMALoss', 'PAC_ChernoffSubGaussian', alpha_prior_values)
plot_metric_scatter(df_results, 'expectedBMALoss', 'PAC_ChernoffFull', alpha_prior_values)
plot_metric_scatter(df_results, 'expectedBMALoss', 'PAC_ChernoffSubGaussianFull', alpha_prior_values)


plot_metric_scatter(df_results, 'expectedGibbsLoss', 'PAC_Chernoff', alpha_prior_values)


plot_metric_scatter(df_results, 'KLPosteriorPrior', 'expectedGibbsVarLoss', alpha_prior_values)

plot_metric_scatter(df_results, 'logMLL', 'PAC_Chernoff', alpha_prior_values)
