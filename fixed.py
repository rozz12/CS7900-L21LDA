import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def orth(A):
    """Get orthogonal basis for the range of A"""
    # Use SVD to get orthogonal basis
    U, _, _ = np.linalg.svd(A, full_matrices=False)
    return U

def GPI(A, B, s=1):
    """
    Implementation of Generalized Power Iteration method (GPI) for solving 
    min_{W'W=I} Tr(W'AW-2W'B)
    
    Parameters:
    A: symmetric matrix with dimension m*m
    B: any matrix with dimension m*k, (m>=k)
    s: can be chosen as 1 or 0, for different ways of determining relaxation parameter alpha
    
    Returns:
    W: optimized orthogonal matrix
    """
    # Get dimensions
    m, k = B.shape
    
    if m < k:
        print('Warning: error input!!!')
        return np.zeros((m, k))
    
    # Ensure A is symmetric using max operation (like MATLAB)
    A_sym = np.maximum(A, A.T)
    
    # Determine alpha (relaxation parameter)
    if s == 0:
        # Use eigenvalue method
        eigenvalues = np.linalg.eigvalsh(A_sym)  # More memory efficient than eigvals
        alpha = np.abs(np.max(eigenvalues))
    elif s == 1:
        # Use power iteration method
        ww = np.random.rand(m, 1)
        for i in range(10):
            m1 = A_sym @ ww
            q = m1 / np.linalg.norm(m1, 2)
            ww = q
        alpha = np.abs(ww.T @ A_sym @ ww)[0, 0]
    else:
        print('Warning: error input!!!')
        return np.zeros((m, k))
    
    # Initialize variables
    err = 1
    t = 1
    W = orth(np.random.rand(m, k))
    
    # Create A_til more efficiently - avoid creating full matrices when possible
    # We'll use A_til @ W = alpha * W - A @ W, without creating A_til explicitly
    
    # Main iteration
    obj = []
    while err > 1e-3 and t < 100:  # Add iteration limit for safety
        # Calculate M = 2 * A_til @ W + 2 * B = 2 * (alpha * W - A @ W) + 2 * B
        AW = A_sym @ W  # This is more memory-efficient
        M = 2 * (alpha * W - AW) + 2 * B
        
        U, _, V = np.linalg.svd(M, full_matrices=False)
        W_new = U @ V.T
        
        # Calculate objective function (more efficiently)
        WTAW = W.T @ AW
        obj_val = np.trace(WTAW - 2 * W.T @ B)
        obj.append(obj_val)
        
        # Check convergence
        if t >= 2:
            err = np.abs(obj[t-2] - obj[t-1])
        
        W = W_new
        t = t + 1
    
    return W

def svd_reduce(X, target_dim, variance_threshold=0.95):
    """
    Reduce dimensionality of data using SVD to retain a specified amount of variance
    
    Parameters:
    X: data matrix (features x samples)
    target_dim: target dimension for reduction (will ensure at least this many dimensions)
    variance_threshold: amount of variance to retain (default 0.95 = 95%)
    
    Returns:
    X_reduced: reduced data matrix
    V: projection matrix to convert original data to reduced space
    """
    # Center the data
    mean_vec = np.mean(X, axis=1, keepdims=True)
    X_centered = X - mean_vec
    
    # Compute SVD - we can use randomized SVD for very large matrices
    try:
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    except:
        from sklearn.decomposition import TruncatedSVD
        print("Using TruncatedSVD for large matrix...")
        n_components = min(X_centered.shape[0], X_centered.shape[1], 1000)
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(X_centered.T)  # Note: TruncatedSVD expects samples as rows
        U = svd.components_.T
        S = svd.singular_values_
        Vt = None  # Not needed for our purposes
    
    # Determine how many components to keep
    total_variance = np.sum(S**2)
    explained_variance_ratio = (S**2) / total_variance
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # Select components that explain the desired variance
    k = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
    k = max(k, target_dim + 10)  # Ensure we keep at least target_dim+10 components
    k = min(k, len(S))  # But not more than we have
    
    print(f"Reducing dimensions from {X.shape[0]} to {k} (keeping {cumulative_variance_ratio[k-1]*100:.2f}% variance)")
    
    # Project data to reduced space
    projection_matrix = U[:, :k]
    X_reduced = projection_matrix.T @ X_centered
    
    return X_reduced, projection_matrix, mean_vec

def RLDA(X_input, y, m, svd_threshold=0.95):
    """
    Implementation of Robust L21-LDA with SVD dimensionality reduction for memory efficiency
    X_input: data, each column is a sample
    y: label vector
    m: reduced dimensions
    svd_threshold: amount of variance to retain in SVD preprocessing
    """
    # Preprocess input data if it's in image format
    if len(X_input.shape) > 2 or (hasattr(X_input, 'dtype') and X_input.dtype == object):
        # Get the first image to determine dimensions
        first_image = X_input[0, 0]
        img_height, img_width = first_image.shape
        d = img_height * img_width  # Total number of pixels/features
        n = X_input.shape[1]  # Number of samples
        
        print(f"Original data: {X_input.shape}, Each image: {first_image.shape}")
        print(f"Restructuring to feature matrix of shape ({d}, {n})")
        
        # Create proper feature matrix: each column is a flattened image
        X = np.zeros((d, n))
        for i in range(n):
            img = X_input[0, i]
            X[:, i] = img.flatten()
    else:
        # Data is already in the correct format
        X = X_input
        d, n = X.shape
    
    # Apply SVD dimensionality reduction to avoid memory issues
    X_reduced, projection_matrix, mean_vec = svd_reduce(X, m, svd_threshold)
    
    # Get dimensions of reduced data
    d_reduced = X_reduced.shape[0]
    
    # Get number of classes
    c = len(np.unique(y))
    
    print(f"Processed data dimensions: {d_reduced}x{n}, Classes: {c}, Reduced dimensions: {m}")
    
    # No need to center the data again, as it was already centered in svd_reduce
    X = X_reduced
    
    # Initialize class-specific data
    Xc = [None] * c
    nc = np.zeros(c, dtype=int)
    dc = [None] * c
    Xm = [None] * c
    
    # Separate data by class
    for i in range(c):
        # Adjust for 1-indexed labels in MATLAB
        class_indices = (y.flatten() == i+1)
        Xc[i] = X[:, class_indices]
        nc[i] = Xc[i].shape[1]
        dc[i] = np.ones(nc[i])
        print(f"Class {i+1}: {nc[i]} samples")
    
    # Initialize W as a random orthogonal matrix
    W = orth(np.random.rand(d_reduced, m))
    
    # Initialize alpha
    alpha = np.zeros((m, n))
    
    # Initialize objective arrays
    obj = np.zeros(30)
    OBJ1 = np.zeros(30)
    OBJ2 = np.zeros(30)
    
    # Main iteration loop
    for iter in range(30):
        print(f"Iteration {iter+1}/30")
        
        # Calculate A and mk
        A = np.zeros((d_reduced, d_reduced))
        B = np.zeros((d_reduced, m))
        ob1 = 0.0
        
        for i in range(c):
            Xi = Xc[i]
            ni = nc[i]
            
            # Create diagonal matrix of weights
            D = np.diag(0.5 / dc[i])
            dd = np.diag(D)
            
            # Update mk (weighted class mean)
            mi = Xi @ dd / np.sum(dd)
            
            # Calculate ||Xi-mk||
            Xmi = Xi - mi.reshape(-1, 1) @ np.ones((1, ni))
            
            # Calculate A
            A = A + Xmi @ D @ Xmi.T
            
            # Store the centered class data
            Xm[i] = Xmi
        
        # Make A symmetric using max (like MATLAB)
        A = np.maximum(A, A.T)
        
        # Calculate lambda and update alpha
        for i in range(n):
            Xx = X[:, i]
            WXi = W.T @ Xx
            a = np.sqrt(np.sum(WXi * WXi))
            
            if a != 0:
                alpha[:, i] = WXi / a
            else:
                alpha[:, i] = np.zeros(len(WXi))
                
            B = B + Xx.reshape(-1, 1) @ alpha[:, i].reshape(1, -1)
            ob1 = ob1 + a
        
        # Process each class
        ob = np.zeros(c)
        for i in range(c):
            Xmi = Xm[i]
            WX = W.T @ Xmi
            dc[i] = np.sqrt(np.sum(WX * WX, axis=0))
            ob[i] = np.sum(dc[i])
        
        # Calculate lambda and objective values
        lambda_val = np.sum(ob) / ob1
        obj[iter] = lambda_val
        OBJ1[iter] = np.sum(ob)
        OBJ2[iter] = ob1
        
        # Update W using GPI
        W = GPI(A, (lambda_val/2) * B, 1)
        
        # Update weights
        for i in range(c):
            Xmi = Xm[i]
            WX = W.T @ Xmi
            dc[i] = np.sqrt(np.sum(WX * WX, axis=0) + np.finfo(float).eps)
    
    # Project back to original space
    W_original = projection_matrix @ W
    
    return W
    # _original, obj, OBJ1, OBJ2

# def evaluate_RLDA_performance(X, Y, max_dim=30, n_folds=5, svd_threshold=0.95):
    """
    Evaluate the performance of RLDA with different dimensions using cross-validation
    
    Parameters:
    X: data matrix (each column is a sample)
    Y: labels (1-indexed)
    max_dim: maximum dimension to test
    n_folds: number of folds for cross-validation
    svd_threshold: amount of variance to retain in SVD preprocessing
    
    Returns:
    results: dictionary with performance metrics
    """
    # Get number of classes and samples
    c = len(np.unique(Y))
    n = X.shape[1] if len(X.shape) == 2 else X.shape[1]
    
    # Convert Y to 0-indexed for scikit-learn
    y_zero_indexed = Y.flatten() - 1
    
    # Initialize arrays to store results
    dims_to_test = range(1, min(max_dim+1, c))  # Can't have more dimensions than classes-1
    avg_accuracies = np.zeros(len(dims_to_test))
    std_accuracies = np.zeros(len(dims_to_test))
    
    # Create cross-validation folds
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Handle different data formats for CV
    if len(X.shape) > 2 or (hasattr(X, 'dtype') and X.dtype == object):
        # For image data, create indices for splitting
        indices = np.arange(n)
        splits = list(kf.split(indices))
    else:
        # For feature data, use X directly
        splits = list(kf.split(X.T))  # Transpose because samples are columns
    
    print(f"Evaluating RLDA with {n_folds}-fold CV for dimensions 1 to {max_dim}")
    
    # For each dimension to test
    for d_idx, dim in enumerate(dims_to_test):
        print(f"\nTesting dimension {dim}/{max_dim}")
        fold_accuracies = []
        
        # For each fold
        for fold, (train_idx, test_idx) in enumerate(splits):
            print(f"  Fold {fold+1}/{n_folds}")
            
            # Split data into train and test
            if len(X.shape) > 2 or (hasattr(X, 'dtype') and X.dtype == object):
                # For image data
                X_train = X[:, train_idx]
                X_test = X[:, test_idx]
            else:
                # For feature data
                X_train = X[:, train_idx]
                X_test = X[:, test_idx]
            
            y_train = Y[train_idx]
            y_test = Y[test_idx]
            
            # Train RLDA
            W, _, _, _ = RLDA(X_train, y_train, dim, svd_threshold)
            
            # Apply the projection to test data
            if len(X.shape) > 2 or (hasattr(X, 'dtype') and X.dtype == object):
                # For image data, preprocess test data
                first_image = X_test[0, 0]
                img_height, img_width = first_image.shape
                d_orig = img_height * img_width
                n_test = X_test.shape[1]
                
                # Create proper feature matrix for test data
                X_test_mat = np.zeros((d_orig, n_test))
                for i in range(n_test):
                    img = X_test[0, i]
                    X_test_mat[:, i] = img.flatten()
                
                # Center the test data using training mean (ideally this would be stored from RLDA)
                X_test_centered = X_test_mat - np.mean(X_test_mat, axis=1, keepdims=True)
                X_test_proj = W.T @ X_test_centered
            else:
                # For feature data
                X_test_centered = X_test - np.mean(X_test, axis=1, keepdims=True)
                X_test_proj = W.T @ X_test_centered
            
            # Project training data for KNN
            if len(X.shape) > 2 or (hasattr(X, 'dtype') and X.dtype == object):
                # For image data, preprocess training data
                first_image = X_train[0, 0]
                img_height, img_width = first_image.shape
                d_orig = img_height * img_width
                n_train = X_train.shape[1]
                
                # Create proper feature matrix for training data
                X_train_mat = np.zeros((d_orig, n_train))
                for i in range(n_train):
                    img = X_train[0, i]
                    X_train_mat[:, i] = img.flatten()
                
                # Center the training data
                X_train_centered = X_train_mat - np.mean(X_train_mat, axis=1, keepdims=True)
                X_train_proj = W.T @ X_train_centered
            else:
                # For feature data
                X_train_centered = X_train - np.mean(X_train, axis=1, keepdims=True)
                X_train_proj = W.T @ X_train_centered
            
            # Use KNN classifier for final classification
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_train_proj.T, y_train.flatten())  # Transpose because samples are in columns
            y_pred = knn.predict(X_test_proj.T)
            
            # Calculate accuracy
            y_test_flat = y_test.flatten()
            accuracy = accuracy_score(y_test_flat, y_pred)
            fold_accuracies.append(accuracy)
            print(f"    Accuracy: {accuracy:.4f}")
        
        # Calculate average and standard deviation of accuracies
        avg_accuracies[d_idx] = np.mean(fold_accuracies)
        std_accuracies[d_idx] = np.std(fold_accuracies)
        print(f"  Dimension {dim}: Avg Acc: {avg_accuracies[d_idx]:.4f}, Std: {std_accuracies[d_idx]:.4f}")
    
    # Find optimal dimension
    best_idx = np.argmax(avg_accuracies)
    best_dim = dims_to_test[best_idx]
    best_acc = avg_accuracies[best_idx]
    best_std = std_accuracies[best_idx]
    
    print(f"\nResults Summary:")
    print(f"Optimal dimension: {best_dim}")
    print(f"Maximal average recognition rate: {best_acc:.4f} (±{best_std:.4f})")
    
    # Store all results in a dictionary
    results = {
        'dimensions': list(dims_to_test),
        'avg_accuracies': avg_accuracies,
        'std_accuracies': std_accuracies,
        'optimal_dim': best_dim,
        'max_accuracy': best_acc,
        'max_std': best_std
    }
    
    return results

if __name__ == "__main__":
    # Example usage
    data = loadmat(r"D:\WSU-Sem 1\ABD\Rimon_Rojan_Adarsh\Robust-L21-LDA\JAFFE.mat")  # Change path as needed
    X = data['fea']
    Y = data['gnd']
    
    print("Data loaded. X shape:", X.shape, "Y shape:", Y.shape)
    print("Y values:", np.unique(Y))

    W = RLDA(X, Y, 2)
    print(W)
    
    # Evaluate performance across dimensions

    # results = evaluate_RLDA_performance(X, Y, max_dim=15, n_folds=5)
    
#     # Plot accuracy vs dimension
#     import matplotlib.pyplot as plt
    
#     plt.figure(figsize=(10, 6))
#     plt.errorbar(results['dimensions'], results['avg_accuracies'], 
#                 yerr=results['std_accuracies'], marker='o', linestyle='-')
#     # Fixed f-string syntax:
#     plt.axvline(x=results['optimal_dim'], color='r', linestyle='--', 
#                label=f"Optimal dim: {results['optimal_dim']}")
#     plt.title('RLDA Performance vs Dimension')
#     plt.xlabel('Reduced Dimension')
#     plt.ylabel('Recognition Rate')
#     plt.grid(True)
#     plt.legend()
#     plt.savefig('rlda_performance.png')
#     plt.show()
    
#     print(f"Best dimension: {results['optimal_dim']}")
#     print(f"Max accuracy: {results['max_accuracy']:.4f} ± {results['max_std']:.4f}")
    
#     # Run final model with optimal dimension
#     print("\nTraining final model with optimal dimension...")
#     W_final, obj, OBJ1, OBJ2 = RLDA(X, Y, results['optimal_dim'], svd_threshold=0.95)