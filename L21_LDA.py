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
    Implementation of Robust L21-LDA with conditional SVD dimensionality reduction
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
    
    # Check if dimensionality reduction is needed (>20K features)
    if d > 20000:
        print(f"Feature dimension ({d}) exceeds 20,000. Applying SVD reduction...")
        X_reduced, projection_matrix, mean_vec = svd_reduce(X, m, svd_threshold)
        # Get dimensions of reduced data
        d_reduced = X_reduced.shape[0]
        X = X_reduced
    else:
        print(f"Feature dimension ({d}) is less than 20,000. Skipping SVD reduction.")
        # Center the data
        mean_vec = np.mean(X, axis=1, keepdims=True)
        X = X - mean_vec
        d_reduced = d
        # Create identity projection matrix for consistency in return values
        projection_matrix = np.eye(d)
    
    # Get number of classes
    c = len(np.unique(y))
    
    print(f"Processed data dimensions: {d_reduced}x{n}, Classes: {c}, Reduced dimensions: {m}")
    
    # Initialize class-specific data
    Xc = [None] * c
    nc = np.zeros(c, dtype=int)
    dc = [None] * c
    Xm = [None] * c
    
    # Separate data by class
    for i in range(c):
        # This line is causing the error because the flattened y doesn't match the shape of X's second dimension
        class_indices = (y.flatten() == i+1)
        
        # If shapes don't match, there's a mismatch between X and y dimensions
        if len(class_indices) != X.shape[1]:
            print(f"Warning: Label array length ({len(class_indices)}) doesn't match data samples ({X.shape[1]})")
            # For debugging:
            print(f"y shape: {y.shape}, X shape: {X.shape}")
            
            # Try to fix by reshaping the class_indices
            if len(y) == X.shape[1]:
                # If y is not flattened properly
                class_indices = (y == i+1).flatten()
            else:
                # If we have a more complex mismatch, try to align indices
                reshaped_y = y.flatten()[:X.shape[1]]  # Take only as many labels as we have samples
                class_indices = (reshaped_y == i+1)
            
            print(f"Adjusted class_indices length: {len(class_indices)}")
            
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
    
    # Project back to original space if SVD was applied
    if d > 20000:
        W_original = projection_matrix @ W
    else:
        W_original = W
    
    return W_original, obj, OBJ1, OBJ2

def evaluate_RLDA_performance(X, Y, max_dim=30, n_runs=5, svd_threshold=0.95, train_ratio=0.7):
    """
    Evaluate the performance of RLDA with different dimensions using multiple train-test splits
    
    Parameters:
    X: data matrix (each column is a sample)
    Y: labels (1-indexed)
    max_dim: maximum dimension to test
    n_runs: number of random train-test splits to average over
    svd_threshold: amount of variance to retain in SVD preprocessing
    train_ratio: portion of data to use for training (0.7 = 70%)
    
    Returns:
    results: dictionary with performance metrics
    """
    # Get number of classes and samples
    c = len(np.unique(Y))
    n = X.shape[1] if len(X.shape) == 2 else X.shape[1]
    
    # Check feature dimensions
    if len(X.shape) == 2:
        d = X.shape[0]  # Feature dimension
    else:
        # For image data, get dimensions from first image
        first_image = X[0, 0]
        img_height, img_width = first_image.shape
        d = img_height * img_width
    
    # Initialize arrays to store results
    dims_to_test = range(1, min(max_dim+1, c))  # Can't have more dimensions than classes-1
    avg_accuracies = np.zeros(len(dims_to_test))
    std_accuracies = np.zeros(len(dims_to_test))
    
    # Store all accuracy results for calculating standard deviation of averages
    all_accuracies = np.zeros((len(dims_to_test), n_runs))
    
    print(f"Evaluating RLDA with {n_runs} 70-30 train-test splits for dimensions 1 to {max_dim}")
    print(f"Feature dimension: {d} ({'Will use SVD' if d > 20000 else 'Will NOT use SVD'})")
    
    # For each dimension to test
    for d_idx, dim in enumerate(dims_to_test):
        print(f"\nTesting dimension {dim}/{max_dim}")
        run_accuracies = []
        
        # Perform multiple runs with different random splits
        for run in range(n_runs):
            print(f"  Run {run+1}/{n_runs}")
            
            # Create random train-test split (70-30)
            from sklearn.model_selection import train_test_split
            indices = np.arange(n)
            train_idx, test_idx = train_test_split(indices, train_size=train_ratio, 
                                                 stratify=Y.flatten(), random_state=run)
            
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
                
                # Center the test data
                test_mean = np.mean(X_test_mat, axis=1, keepdims=True)
                X_test_centered = X_test_mat - test_mean
            else:
                # For feature data
                test_mean = np.mean(X_test, axis=1, keepdims=True)
                X_test_centered = X_test - test_mean
            
            # Project test data
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
                train_mean = np.mean(X_train_mat, axis=1, keepdims=True)
                X_train_centered = X_train_mat - train_mean
            else:
                # For feature data
                train_mean = np.mean(X_train, axis=1, keepdims=True)
                X_train_centered = X_train - train_mean
            
            # Project training data
            X_train_proj = W.T @ X_train_centered
            
            # Use KNN classifier for final classification
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_train_proj.T, y_train.flatten())  # Transpose because samples are in columns
            y_pred = knn.predict(X_test_proj.T)
            
            # Calculate accuracy
            y_test_flat = y_test.flatten()
            accuracy = accuracy_score(y_test_flat, y_pred)
            run_accuracies.append(accuracy)
            
            # Store accuracy for computing std of averages
            all_accuracies[d_idx, run] = accuracy
            
            print(f"    Accuracy: {accuracy:.4f}")
        
        # Calculate average and standard deviation of accuracies for this dimension
        avg_accuracies[d_idx] = np.mean(run_accuracies)
        std_accuracies[d_idx] = np.std(run_accuracies)
        print(f"  Dimension {dim}: Avg Acc: {avg_accuracies[d_idx]:.4f}, Std: {std_accuracies[d_idx]:.4f}")
    
    # Calculate overall average accuracy across all dimensions
    overall_avg_accuracy = np.mean(avg_accuracies)
    
    # Calculate standard deviation of the average accuracies
    std_of_avg_accuracies = np.std(avg_accuracies)
    
    print(f"\nResults Summary:")
    
    # Store all results in a dictionary
    results = {
        'dimensions': list(dims_to_test),
        'avg_accuracies': avg_accuracies,
        'std_accuracies': std_accuracies,
        'overall_avg_accuracy': overall_avg_accuracy,
        'std_of_avg_accuracies': std_of_avg_accuracies,
        'all_accuracies': all_accuracies
    }
    
    return results

if __name__ == "__main__":
    # Example usage
    filename = "Prostate_GE.mat"
    data = loadmat(filename)  # Change path as needed
    #handle data and labels as per biological or image datasets
    if 'fea' in data and 'gnd' in data:
        # Image dataset format
        X = data['fea']
        Y = data['gnd']

    else:
        # Biological dataset format
        X = data['X']
        Y = data['Y']
        X = X.T #the code as per MATLAB expects data samples in column and features in 
    
    print("Data loaded. X shape:", X.shape, "Y shape:", Y.shape)
    print("Y values:", np.unique(Y))

    # Run RLDA with a specific dimension
    W, obj, OBJ1, OBJ2 = RLDA(X, Y, 2)
    print("RLDA projection matrix shape:", W.shape)
    
    # Evaluate performance across dimensions
    results = evaluate_RLDA_performance(X, Y, max_dim=15, n_runs=5, train_ratio=0.7)
    print(f"Dataset: {filename.split('.')[0]}")
    print(f"Overall average accuracy: {results['overall_avg_accuracy']:.4f} ± {results['std_of_avg_accuracies']:.4f}")