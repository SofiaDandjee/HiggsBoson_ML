import numpy as np

def build_k_indices(y, k_fold, seed):
    """Build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold) 
    #Generate random indices
    np.random.seed(seed) 
    indices = np.random.permutation(num_row)
    #Contruct set of indices for the k different folds for cross validation
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation (y, x, k, k_fold, seed):
    """Randomly partition the data into k_fold groups to train and test each sub-dataset"""
    # Split data in k_folds
    k_indices = build_k_indices(y, k_fold, seed)
    # Get k'th subgroup in test, others in train
    y_test = y[k_indices[k]]
    x_test = x[k_indices[k]]
    k_indices_del = np.delete(k_indices,k,0)
    y_train = y[np.ravel(k_indices_del)]
    x_train = x[np.ravel(k_indices_del)]
    
    return x_train, y_train, x_test, y_test
    