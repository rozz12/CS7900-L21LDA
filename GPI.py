import numpy as np
from scipy import linalg

def safe_norm(v, axis=None, eps=1e-10):
    """Calculate L2 norm safely"""
    # If v is a vector
    if axis is None:
        # Square and sum
        v_squared = v * v
        # Replace any NaN values with zeros
        v_squared = np.nan_to_num(v_squared, nan=0.0, posinf=0.0, neginf=0.0)
        sum_squared = np.sum(v_squared)
        # Ensure positive value
        sum_squared = max(sum_squared, eps*eps)
        # Take square root
        return np.sqrt(sum_squared)
    else:
        # For matrix operations along an axis
        # Square elements
        v_squared = v * v
        # Replace invalid values
        v_squared = np.nan_to_num(v_squared, nan=0.0, posinf=0.0, neginf=0.0)
        # Sum along axis
        sum_squared = np.sum(v_squared, axis=axis)
        # Ensure positive values
        sum_squared = np.maximum(sum_squared, eps*eps)
        # Take square root
        return np.sqrt(sum_squared)

def is_valid(arr):
    """Check if array contains invalid values"""
    return not (np.isnan(arr).any() or np.isinf(arr).any())

def GPI(A, B, k):
    """
    Implementation of Generalized Power Iteration algorithm
    with improved numerical stability
    
    A: matrix A
    B: matrix B
    k: number of eigenvectors to extract
    """
    # Ensure inputs are valid
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Get dimensions
    d = A.shape[0]
    
    # Initialize projection matrix Q as a random orthogonal matrix
    Q = np.random.randn(d, k)
    Q, _ = np.linalg.qr(Q)
    
    # Initialize M
    M = np.zeros_like(Q)
    
    # Iteration parameters
    max_iter = 100
    tol = 1e-5
    
    # Main iteration
    for i in range(max_iter):
        # Calculate M = AQ - BQ(Q'BQ)^(-1)(Q'AQ)
        try:
            # Calculate Q'BQ and ensure it's invertible
            QBQ = Q.T @ B @ Q
            # Add small value to diagonal to ensure invertibility
            QBQ += np.eye(k) * 1e-10
            QBQ_inv = np.linalg.inv(QBQ)
            
            # Calculate Q'AQ
            QAQ = Q.T @ A @ Q
            
            # Calculate BQ(Q'BQ)^(-1)(Q'AQ)
            temp = B @ Q @ QBQ_inv @ QAQ
            
            # Calculate AQ - temp
            M = A @ Q - temp
            
            # Check for invalid values
            if not is_valid(M):
                print("Warning: Invalid values in M. Using simplified update.")
                M = A @ Q  # Simplified update
        except Exception as e:
            print(f"Error in GPI calculation: {e}. Using simplified update.")
            M = A @ Q  # Simplified update
        
        # Orthogonalize M to get the new Q
        try:
            # Try QR decomposition first
            Q_new, R = np.linalg.qr(M)
            
            # Check if Q_new is valid
            if not is_valid(Q_new) or Q_new.shape[1] < k:
                # Try another approach if QR fails
                print("QR decomposition failed. Using SVD.")
                U, _, _ = np.linalg.svd(M, full_matrices=False)
                Q_new = U[:, :k]
        except Exception as e:
            print(f"Error in orthogonalization: {e}. Using simplified approach.")
            # Safe orthogonalization
            for j in range(k):
                # Get column
                m1 = M[:, j]
                
                # Normalize
                norm = safe_norm(m1)
                q = m1 / norm
                
                # Ensure q is a valid unit vector
                if not is_valid(q):
                    q = np.random.randn(d)
                    q = q / safe_norm(q)
                
                # Set normalized column
                if j == 0:
                    Q_new = q.reshape(-1, 1)
                else:
                    # Orthogonalize against previous columns
                    for l in range(j):
                        q = q - (Q_new[:, l].T @ q) * Q_new[:, l]
                    
                    # Re-normalize
                    norm = safe_norm(q)
                    if norm > 1e-10:
                        q = q / norm
                    else:
                        # If orthogonalization results in near-zero vector,
                        # generate a new random vector
                        q = np.random.randn(d)
                        # Orthogonalize against existing columns
                        for l in range(j):
                            q = q - (Q_new[:, l].T @ q) * Q_new[:, l]
                        q = q / safe_norm(q)
                    
                    # Append to Q_new
                    Q_new = np.column_stack((Q_new, q))
        
        # Check convergence
        if np.linalg.norm(Q_new - Q, 'fro') < tol:
            break
        
        # Update Q
        Q = Q_new
    
    # Calculate eigenvalues
    try:
        QAQ = Q.T @ A @ Q
        QBQ = Q.T @ B @ Q
        eigvals = np.diag(QAQ) / np.diag(QBQ)
    except Exception as e:
        print(f"Error calculating eigenvalues: {e}. Using placeholder values.")
        eigvals = np.ones(k)
    
    return Q, eigvals