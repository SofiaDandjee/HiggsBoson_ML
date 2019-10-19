import numpy as np

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation (y, x, k,seed):

    k_indices = build_k_indices(y, k, seed)
    
    y_val = y[k_indices[k-1,:]]
    x_val = x[k_indices[k-1,:]]
    k_indices_del = np.delete(k_indices,k-1,0)
    y_tr = np.ravel(y[k_indices_del])
    x_tr = np.ravel(x[k_indices_del])
    
    return x_tr, y_tr, x_val, y_val
    