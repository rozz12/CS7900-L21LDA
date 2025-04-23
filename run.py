import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA

def between_class_scatter(data, labels):
    """Calculate between-class scatter matrix"""
    classes = np.unique(labels)
    overall_mean = np.mean(data, axis=0)
    S_b = np.zeros((data.shape[1], data.shape[1]))
    
    for i in classes:
        class_data = data[labels == i, :]
        class_mean = np.mean(class_data, axis=0)
        n_i = class_data.shape[0]
        
        diff = class_mean - overall_mean
        S_b += n_i * np.outer(diff, diff)
    
    return S_b

def within_class_scatter(data, labels):
    """Calculate within-class scatter matrix"""
    classes = np.unique(labels)
    S_w = np.zeros((data.shape[1], data.shape[1]))
    
    for i in classes:
        class_data = data[labels == i, :]
        class_mean = np.mean(class_data, axis=0)
        
        centered = class_data - np.tile(class_mean, (class_data.shape[0], 1))
        S_w += centered.T @ centered
    
    return S_w

def calculate_scatter(data, labels):
    """Calculate both scatter matrices and return as tuple"""
    return between_class_scatter(data, labels), within_class_scatter(data, labels)

if __name__ == "__main__":
    # Load dataset
    data = loadmat(r'D:\WSU-Sem 1\ABD\Rimon_Rojan_Adarsh\Robust-L21-LDA\JAFFE.mat')
    X = data['fea']
    y = data['gnd']
    print('Data loaded successfully')
    
    # Handle cell array conversion if needed (similar to MATLAB)
    if hasattr(X, 'dtype') and X.dtype == object:
        print('Converting cell array to matrix')
        # Convert cell array equivalent to flat matrix
        first_cell = X[0, 0]
        if isinstance(first_cell, np.ndarray):
            # If cells contain images or matrices, flatten them
            n_samples = X.shape[1]
            sample_dims = first_cell.size
            X_flat = np.zeros((n_samples, sample_dims))
            
            for i in range(n_samples):
                X_flat[i, :] = X[0, i].flatten()
            
            X = X_flat
    else:
        print('X is already a matrix, no conversion needed')
    
    # Choose number of dimensions to reduce to
    m = 2  # Reduce to 2 dimensions for visualization
    
    # Store X_transposed for consistent use like in MATLAB
    X_transposed = X.T
    
    # Run the RLDA algorithm - use your existing function
    from fixed import RLDA  # Import your existing function
    W, obj, OBJ1, OBJ2 = RLDA(X, y, m, svd_threshold=0.95)
    
    # Project the data
    X_projected = W.T @ X_transposed
    
    # Create a figure to visualize results
    plt.figure(figsize=(18, 5))
    
    # Convert labels to numeric if needed
    y_numeric = y.flatten()
    
    # Create a colormap with distinct colors
    num_classes = len(np.unique(y_numeric))
    cmap = plt.cm.jet(np.linspace(0, 1, num_classes))
    
    # 1. Visualize the raw data (using the first 2 dimensions/features)
    plt.subplot(1, 3, 1)
    if X_transposed.shape[0] >= 2:
        X_raw = X_transposed.T
        
        # Plot each class separately
        for i in range(1, num_classes + 1):
            class_idx = (y_numeric == i)
            plt.scatter(X_raw[class_idx, 0], X_raw[class_idx, 1], s=50, 
                        color=cmap[i-1], label=str(i), alpha=0.7)
        
        plt.title('Raw Data (First 2 Dimensions)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True)
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'Data has fewer than 2 dimensions', 
                 horizontalalignment='center')
        plt.axis('off')
    
    # 2. Visualize the PCA projection
    plt.subplot(1, 3, 2)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_transposed.T)
    
    # Plot each class separately
    for i in range(1, num_classes + 1):
        class_idx = (y_numeric == i)
        plt.scatter(X_pca[class_idx, 0], X_pca[class_idx, 1], s=50, 
                    color=cmap[i-1], label=str(i), alpha=0.7)
    
    plt.title('PCA Projection')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.legend()
    
    # 3. Visualize the RLDA projection
    plt.subplot(1, 3, 3)
    X_proj_vis = X_projected.T
    
    # Plot each class separately
    for i in range(1, num_classes + 1):
        class_idx = (y_numeric == i)
        plt.scatter(X_proj_vis[class_idx, 0], X_proj_vis[class_idx, 1], s=50, 
                    color=cmap[i-1], label=str(i), alpha=0.7)
    
    plt.title('RLDA-L21 Projection')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('data_projections_comparison.png')
    print('Visualization saved to data_projections_comparison.png')
    
    # Calculate and display class separability metrics
    print('\nCalculating class separability metrics...')
    
    # Calculate scatter for raw data (first 2 dimensions)
    if X_raw.shape[1] >= 2:
        between_raw, within_raw = calculate_scatter(X_raw[:, :2], y_numeric)
        ratio_raw = np.trace(between_raw) / np.trace(within_raw)
        print(f'Between/Within Class Scatter Ratio (Raw): {ratio_raw:.4f}')
    
    # Calculate scatter for PCA projection
    between_pca, within_pca = calculate_scatter(X_pca, y_numeric)
    ratio_pca = np.trace(between_pca) / np.trace(within_pca)
    print(f'Between/Within Class Scatter Ratio (PCA): {ratio_pca:.4f}')
    
    # Calculate scatter for RLDA projection
    between_rlda, within_rlda = calculate_scatter(X_proj_vis, y_numeric)
    ratio_rlda = np.trace(between_rlda) / np.trace(within_rlda)
    print(f'Between/Within Class Scatter Ratio (RLDA): {ratio_rlda:.4f}')
    
    # Compare RLDA to PCA
    if ratio_rlda > ratio_pca:
        print(f'RLDA improvement over PCA: {100 * (ratio_rlda/ratio_pca - 1):.2f}%')
    else:
        print(f'RLDA decrease compared to PCA: {100 * (ratio_rlda/ratio_pca - 1):.2f}%')
    
    # Compare RLDA to raw data if available
    if X_raw.shape[1] >= 2:
        if ratio_rlda > ratio_raw:
            print(f'RLDA improvement over raw data: {100 * (ratio_rlda/ratio_raw - 1):.2f}%')
        else:
            print(f'RLDA decrease compared to raw data: {100 * (ratio_rlda/ratio_raw - 1):.2f}%')
    
    # Save the results for comparison
    np.savez('rlda_python_results.npz', 
             W=W, 
             obj=obj, 
             OBJ1=OBJ1, 
             OBJ2=OBJ2,
             ratio_raw=ratio_raw if X_raw.shape[1] >= 2 else None,
             ratio_pca=ratio_pca,
             ratio_rlda=ratio_rlda)
    
    print('\nRLDA-L21 Analysis Complete')
    print('Results saved to rlda_python_results.npz')