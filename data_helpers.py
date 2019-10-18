import numpy as np


def standardize(x,id_axis):
    """Standardize the original data set."""
    mean_x = np.mean(x,axis=id_axis)
    x = x - mean_x
    std_x = np.std(x,axis=id_axis)
    x = x / std_x
    return x, mean_x, std_x


def build_model_data(x, y):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

            
def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    num_samples = len(x)
    
    indices = np.random.permutation(num_samples)
    indices_training = indices[0:int(np.floor(ratio * num_samples))]
    indices_test = indices [int(np.floor(ratio * num_samples)):num_samples]
    
    train_x = x[indices_training]
    train_y = y[indices_training]
    test_x = x[indices_test]
    test_y = y[indices_test]
    
    return train_x, train_y, test_x, test_y


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.zeros((x.shape[0],degree+1))
    for j in range (0,degree+1):
        phi[:,j] = x**j
        
    return phi