import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INTERPOLATION_THRESHOLD = 99.9

def plot_laplace_results(base_url):
    
    precisions = ["100.0", "10.0", "0.0", "1.0", "0.1", "0.01"]
    csv_files = [base_url + precision + "_results.csv" for precision in precisions]
    
    # Set a professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    


    for i, name in enumerate(csv_files):
        df = pd.read_csv(name)

        bayes_loss = df['bayes loss'].values
        gibbs_loss = df['gibbs loss'].values
        bayes_loss_train = df['bayes loss train'].values
        gibbs_loss_train = df['gibbs loss train'].values
        inverse_rate = df['inverse rate'].values
        extended_kl = df['expanded kl'].values/50000
        variance = df['variance'].values
        parameters = df['parameters'].values
        train_accuracy = df['BMA train accuracy (%)'].values*100

        x = len(bayes_loss)
        # Find index where train accuracy exceeds threshold
        accuracy_threshold = INTERPOLATION_THRESHOLD
        threshold_indices = np.where(train_accuracy > accuracy_threshold)[0]
        if len(threshold_indices) > 0:
            # Get the first index where accuracy exceeds threshold
            mid_point = threshold_indices[0]
            # Get the parameter count at the threshold for the vertical line
            threshold_param = parameters[mid_point]
        else:
            # Fallback to the middle if threshold not found
            mid_point = len(train_accuracy) // 2
            threshold_param = None
        

        gibbs_gap = gibbs_loss - gibbs_loss_train
        bayes_gap = bayes_loss - bayes_loss_train
        jensen_gap = gibbs_gap - bayes_gap


        # Create figure with better spacing and higher DPI for better quality
        fig, ax1 = plt.subplots(1, 4, figsize=(13, 6), dpi=120)
        
        # Add a light gray background to the entire figure
        fig.patch.set_facecolor('#f8f9fa')

        # First subplot - metrics vs parameters
        ax2 = ax1[0].twinx()
        ax1[0].plot(parameters, gibbs_gap, label='Gibbs gap', color='#e63946', linewidth=2.5, marker='o', markersize=6)
        ax1[0].plot(parameters, inverse_rate, label='Inverse rate', color='#457b9d', linewidth=2.5, marker='s', markersize=6)
        ax2.plot(parameters, variance, label='Variance', color='#2a9d8f', linewidth=2.5, marker='^', markersize=6)
        
        # Add vertical line for interpolation threshold in the first subplot
        if threshold_param is not None:
            ax1[0].axvline(x=threshold_param, color='#6c757d', linestyle='-.', alpha=0.7, linewidth=1.5)
            # Calculate a better position for the text - adjust based on plot bounds
            x_min, x_max = ax1[0].get_xlim()
            y_min, y_max = ax1[0].get_ylim()
            text_x_pos = threshold_param + (x_max - x_min) * 0.05  # Slight offset from line
            text_y_pos = y_min + (y_max - y_min) * 0.1  # Higher up in the plot
            
            # Add text with smaller font and better positioning
            ax1[0].text(text_x_pos, text_y_pos, 'Interpolation\nthreshold', 
                    fontsize=9, color='#6c757d', fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        ax1[0].set_xlabel('Parameter Count', fontsize=12, fontweight='bold')
        ax1[0].set_ylabel('Value', fontsize=12, fontweight='bold')
        ax1[0].set_title('Metrics vs Parameters', fontsize=14, fontweight='bold', pad=15)
        ax1[0].legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=9)
        ax2.legend(loc='center right', frameon=True, fancybox=True, shadow=True, fontsize=9)
        
        # Add grid to first subplot
        ax1[0].grid(True, linestyle='--', alpha=0.7)



        # Second subplot - first half of points
        scatter1 = ax1[1].scatter(gibbs_gap[-x:-x+mid_point], extended_kl[-x:-x+mid_point], 
                               c=parameters[-x:-x+mid_point], cmap='viridis', s=350, alpha=0.85, 
                               edgecolors='w', linewidths=1.5)
        # Calculate correlation coefficient
        corr1 = np.corrcoef(gibbs_gap[-x:-x+mid_point], extended_kl[-x:-x+mid_point])[0,1]
        ax1[1].set_xlabel('Gibbs gap', fontsize=12, fontweight='bold')
        ax1[1].set_ylabel('Normalized (extended) KL', fontsize=12, fontweight='bold')
        ax1[1].set_title(f'Non-Interpolators\nr = {corr1:.2f}', 
                      fontsize=14, fontweight='bold', pad=15)
        ax1[1].grid(True, linestyle='--', alpha=0.5)

        # Third subplot - second half of points
        scatter2 = ax1[2].scatter(gibbs_gap[-x+mid_point:], extended_kl[-x+mid_point:], 
                               c=parameters[-x+mid_point:], cmap='viridis', s=350, alpha=0.85,
                               edgecolors='w', linewidths=1.5)
        # Calculate correlation coefficient
        corr2 = np.corrcoef(gibbs_gap[-x+mid_point:], extended_kl[-x+mid_point:])[0,1]
        ax1[2].set_xlabel('Gibbs gap', fontsize=12, fontweight='bold')
        ax1[2].set_ylabel('Normalized KL', fontsize=12, fontweight='bold')
        ax1[2].set_title(f'Interpolators\nr = {corr2:.2f}', 
                      fontsize=14, fontweight='bold', pad=15)
        ax1[2].grid(True, linestyle='--', alpha=0.5)

        # Approximation.
        #inverse_rate = np.sqrt(variance*extended_kl)

        # Fourth subplot
        scatter3 = ax1[3].scatter(gibbs_gap[-x:], inverse_rate[-x:], 
                               c=parameters[-x:], cmap='viridis', s=350, alpha=0.85,
                               edgecolors='w', linewidths=1.5)
        # Calculate correlation coefficient
        corr3 = np.corrcoef(gibbs_gap[-x:], inverse_rate[-x:])[0,1]
        ax1[3].set_xlabel('Gibbs gap', fontsize=12, fontweight='bold')
        ax1[3].set_ylabel('Inverse rate', fontsize=12, fontweight='bold')
        ax1[3].set_title(f'Gibbs gap vs Inverse rate\nr = {corr3:.2f}', 
                      fontsize=14, fontweight='bold', pad=15)
        ax1[3].grid(True, linestyle='--', alpha=0.5)

        # Improved colorbar
        cbar = plt.colorbar(scatter3, ax=ax1[3], pad=0.01)
        cbar.set_label('Parameter Count', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)

        # Improve spacing between subplots
        plt.subplots_adjust(wspace=0.3)  # Increase spacing between subplots
        
        # Use tight_layout with adjusted parameters to prevent title overlap
        fig.tight_layout(rect=[0, 0.03, 1, 0.93])  # Adjust the rect parameters to leave more space for title

        # Move suptitle up slightly to avoid overlap
        fig.suptitle(f"ConvNN with prior precision {precisions[i]}", 
                   fontsize=16, fontweight='bold', y=0.98)
        
        # Add a subtle border around the figure
        for spine in ax1[0].spines.values():
            spine.set_edgecolor('#cccccc')
        
        plt.show()



def plot_train_test_results(base_url):
    precisions = ["100.0", "10.0", "0.0", "1.0", "0.1"]
    csv_files = [base_url + precision + "_results.csv" for precision in precisions]
    
    # Set a professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    


    for i, name in enumerate(csv_files):
        df = pd.read_csv(name)
        parameters = df['parameters'].values
        train_accuracy = df['BMA train accuracy (%)'].values*100
        test_accuracy = df['BMA test accuracy (%)'].values*100
        train_loss = df['gibbs loss train'].values
        test_loss = df['bayes loss train'].values
        parameters = df['parameters'].values

        # Create figure with better spacing and higher DPI for better quality
        fig, ax1 = plt.subplots(figsize=(12, 8), dpi=120)
        
        # Add a light gray background to the figure
        fig.patch.set_facecolor('#f8f9fa')

        # Plot accuracies with improved styling
        ax1.set_xlabel('Parameters', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='#1a759f')
        
        # Add horizontal line at 100% accuracy
        ax1.axhline(y=100, color='#adb5bd', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Plot train and test accuracy with better styling
        ax1.plot(parameters, train_accuracy, label='Train Accuracy', 
                color='#1a759f', marker='o', markersize=8, linewidth=2.5)
        ax1.plot(parameters, test_accuracy, label='Test Accuracy', 
                color='#34a0a4', marker='s', markersize=8, linewidth=2.5)
        
        ax1.tick_params(axis='y', labelcolor='#1a759f', labelsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Highlight the interpolation threshold
        interpolation_idx = np.where(train_accuracy >= INTERPOLATION_THRESHOLD)[0]
        if len(interpolation_idx) > 0:
            threshold_x = parameters[interpolation_idx[0]]
            ax1.axvline(x=threshold_x, color='#6c757d', linestyle='-.', alpha=0.7, linewidth=1.5)
            ax1.text(threshold_x*1.05, 50, 'Interpolation\nthreshold', 
                    fontsize=10, color='#6c757d', fontweight='bold')

        # Plot losses on secondary y-axis with improved styling
        ax2 = ax1.twinx()
        ax2.set_ylabel('Loss', fontsize=12, fontweight='bold', color='#d62828')
        ax2.plot(parameters, train_loss, label='Train Loss', 
                color='#d62828', linestyle='--', marker='x', markersize=8, linewidth=2.5)
        ax2.plot(parameters, test_loss, label='Test Loss', 
                color='#e85d04', linestyle='--', marker='+', markersize=8, linewidth=2.5)
        ax2.tick_params(axis='y', labelcolor='#d62828', labelsize=10)

        # Create better legends with styling
        accuracy_legend = ax1.legend(loc='upper left', frameon=True, fancybox=True, 
                                shadow=True, fontsize=10, title='Accuracy Metrics')
        accuracy_legend.get_title().set_fontweight('bold')
        
        loss_legend = ax2.legend(loc='upper right', frameon=True, fancybox=True, 
                            shadow=True, fontsize=10, title='Loss Metrics')
        loss_legend.get_title().set_fontweight('bold')

        # Add a main title with better styling
        plt.title('Accuracy and Loss vs Parameters in CNN', 
                fontsize=16, fontweight='bold', pad=20)
        
        # Add annotations for key observations
        max_test_acc_idx = np.argmax(test_accuracy)
        ax1.annotate(f'Max Test Acc: {test_accuracy[max_test_acc_idx]:.2f}%',
                    xy=(parameters[max_test_acc_idx], test_accuracy[max_test_acc_idx]),
                    xytext=(parameters[max_test_acc_idx]*0.7, test_accuracy[max_test_acc_idx]-10),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10, fontweight='bold')
        

        # Add a main title with better styling
        fig.suptitle(f"ConvNN with prior precision {precisions[i]}", 
                   fontsize=16, fontweight='bold', y=0.98)
        # Add subtle border
        for spine in ax1.spines.values():
            spine.set_edgecolor('#cccccc')
        
        # Improve layout
        fig.tight_layout()
        plt.show()

def plot_baseline_train_test_results():
    csv_file = "https://raw.githubusercontent.com/Ludvins/PAC_Bayes_Chernoff_MLL/refs/heads/main/results/ConvNN_train_results.csv"

    # Set a professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    df = pd.read_csv(csv_file)

    train_accuracy = df['train accuracy (%)'].values
    test_accuracy = df['test accuracy (%)'].values
    train_loss = df['train loss'].values
    test_loss = df['test loss'].values
    parameters = df['parameters'].values

    # Create figure with better spacing and higher DPI for better quality
    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=120)
    
    # Add a light gray background to the figure
    fig.patch.set_facecolor('#f8f9fa')

    # Plot accuracies with improved styling
    ax1.set_xlabel('Parameters', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='#1a759f')
    
    # Add horizontal line at 100% accuracy
    ax1.axhline(y=100, color='#adb5bd', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Plot train and test accuracy with better styling
    ax1.plot(parameters, train_accuracy, label='Train Accuracy', 
             color='#1a759f', marker='o', markersize=8, linewidth=2.5)
    ax1.plot(parameters, test_accuracy, label='Test Accuracy', 
             color='#34a0a4', marker='s', markersize=8, linewidth=2.5)
    
    ax1.tick_params(axis='y', labelcolor='#1a759f', labelsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Highlight the interpolation threshold
    interpolation_idx = np.where(train_accuracy >= 100.0)[0]
    if len(interpolation_idx) > 0:
        threshold_x = parameters[interpolation_idx[0]]
        ax1.axvline(x=threshold_x, color='#6c757d', linestyle='-.', alpha=0.7, linewidth=1.5)
        ax1.text(threshold_x*1.05, 50, 'Interpolation\nthreshold', 
                fontsize=10, color='#6c757d', fontweight='bold')

    # Plot losses on secondary y-axis with improved styling
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold', color='#d62828')
    ax2.plot(parameters, train_loss, label='Train Loss', 
             color='#d62828', linestyle='--', marker='x', markersize=8, linewidth=2.5)
    ax2.plot(parameters, test_loss, label='Test Loss', 
             color='#e85d04', linestyle='--', marker='+', markersize=8, linewidth=2.5)
    ax2.tick_params(axis='y', labelcolor='#d62828', labelsize=10)

    # Create better legends with styling
    accuracy_legend = ax1.legend(loc='upper left', frameon=True, fancybox=True, 
                                shadow=True, fontsize=10, title='Accuracy Metrics')
    accuracy_legend.get_title().set_fontweight('bold')
    
    loss_legend = ax2.legend(loc='upper right', frameon=True, fancybox=True, 
                           shadow=True, fontsize=10, title='Loss Metrics')
    loss_legend.get_title().set_fontweight('bold')

    # Add a main title with better styling
    plt.title('Accuracy and Loss vs Parameters in CNN', 
             fontsize=16, fontweight='bold', pad=20)
    
    # Add annotations for key observations
    max_test_acc_idx = np.argmax(test_accuracy)
    ax1.annotate(f'Max Test Acc: {test_accuracy[max_test_acc_idx]:.2f}%',
                xy=(parameters[max_test_acc_idx], test_accuracy[max_test_acc_idx]),
                xytext=(parameters[max_test_acc_idx]*0.7, test_accuracy[max_test_acc_idx]-10),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                fontsize=10, fontweight='bold')
    
    # Add subtle border
    for spine in ax1.spines.values():
        spine.set_edgecolor('#cccccc')
    
    # Improve layout
    fig.tight_layout()
    plt.show()

def plot_kl_vs_parameters_across_precisions(base_url):
    precisions = ["1000.0", "100.0", "10.0", "1.0", "0.0"]
    csv_files = [base_url + precision + "_results.csv" for precision in precisions]
    
    # Set a professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create figure with better spacing and higher DPI for better quality
    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=120)
    
    # Add a light gray background to the figure
    fig.patch.set_facecolor('#f8f9fa')
    
    # Define a color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot KL for each precision
    for i, name in enumerate(csv_files):
        df = pd.read_csv(name)
        parameters = df['parameters'].values
        extended_kl = df['expanded kl'].values/50000  # Normalize KL
        train_accuracy = df['BMA train accuracy (%)'].values*100
        
        # Find interpolation threshold
        threshold_indices = np.where(train_accuracy > INTERPOLATION_THRESHOLD)[0]
        if len(threshold_indices) > 0:
            threshold_param = parameters[threshold_indices[0]]
        else:
            threshold_param = None
            
        # Plot with markers at interpolation threshold
        line = ax1.plot(parameters, extended_kl, 
                label=f'Prior precision {precisions[i]}',
                color=colors[i], linewidth=2.5, marker='o', markersize=4)
        
        # Add marker at interpolation threshold if it exists
        if threshold_param is not None:
            threshold_kl = extended_kl[threshold_indices[0]]
            ax1.scatter(threshold_param, threshold_kl, 
                       color=colors[i], s=100, marker='*', 
                       zorder=5, label='_nolegend_')
    
    # Styling
    ax1.set_xlabel('Parameter Count', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Normalized Extended KL', fontsize=12, fontweight='bold')
    ax1.set_title('Extended KL vs Parameters Across Prior Precisions', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Add grid
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Create legend with better styling
    legend = ax1.legend(loc='upper left', frameon=True, fancybox=True, 
                       shadow=True, fontsize=10, title='Prior Precision')
    legend.get_title().set_fontweight('bold')
    
    # Add note about markers
    ax1.text(0.02, 0.98, '* indicates interpolation threshold', 
             transform=ax1.transAxes, fontsize=9, fontweight='bold',
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    
    # Improve layout
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    #base_url = "https://raw.githubusercontent.com/Ludvins/PAC_Bayes_Chernoff_MLL/refs/heads/main/results/laplace_ConvNN_last_layer_kron_scalar_"
    base_url = "https://raw.githubusercontent.com/Ludvins/PAC_Bayes_Chernoff_MLL/refs/heads/main/results/laplace_MLP_last_layer_kron_scalar_"
    
    plot_laplace_results(base_url)
    #plot_train_test_results(base_url)
    #plot_baseline_train_test_results()
    #plot_kl_vs_parameters_across_precisions(base_url)
