import numpy as np
from scipy.io import loadmat

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
    
    return W_original, obj, OBJ1, OBJ2
    """
    Implementation of Robust L21-LDA matching MATLAB implementation
    X_input: data, each column is a sample
    y: label vector
    m: reduced dimensions
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
    
    # Get number of classes
    c = len(np.unique(y))
    
    print(f"Processed data dimensions: {d}x{n}, Classes: {c}, Reduced dimensions: {m}")
    
    # Center the data
    mean_vec = np.mean(X, axis=1, keepdims=True)
    X = X - mean_vec
    
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
    W = orth(np.random.rand(d, m))
    
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
        A = np.zeros((d, d))
        B = np.zeros((d, m))
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
    
    return W, obj, OBJ1, OBJ2

if __name__ == "__main__":
    # Example usage
    data = loadmat(r"D:\WSU-Sem 1\ABD\Rimon_Rojan_Adarsh\Robust-L21-LDA\JAFFE.mat")
    X = data['fea']
    Y = data['gnd']

    # Use SVD to reduce dimensions while preserving 95% of variance
    W, obj, OBJ1, OBJ2 = RLDA(X, Y, 2, svd_threshold=0.95)
    print("Projection matrix shape:", W.shape)
    
    # Plot objective function values
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(obj, 'r-')
    plt.title('Objective Function (lambda)')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(OBJ1, 'b-')
    plt.title('OBJ1 (Sum of class norms)')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(OBJ2, 'g-')
    plt.title('OBJ2 (Sum of sample norms)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('rlda_objective_values.png')
    plt.show()