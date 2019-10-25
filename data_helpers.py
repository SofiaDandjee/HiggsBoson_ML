import numpy as np

def clean_data(tx):
   
    tx_nan = undefined_to_nan (tx)
   
    tx_light,indices = remove_features (tx_nan)

    tx_std,_,_ = standardize (tx_light,0)
 
    tx = nan_to_zero(tx_std)

    return tx, indices

def augment_data(tx, y, degree) :
    
    tx = build_poly_all_features (tx,degree)
    y,tx = build_model_data(tx,y)
    
    return tx, y

def get_jet_samples (tx):
    
    jet0_samples = np.where(tx[:,22]==0)[0]
    jet1_samples = np.where(tx[:,22]==1)[0]
    jet2_samples = np.where(tx[:,22]>=2)[0]
    
    return [jet0_samples, jet1_samples, jet2_samples]

def undefined_to_nan(tx):
    undefined = -999.00
    tx [tx == undefined] = np.nan
    return tx

def nan_to_zero(tx):
    tx = np.nan_to_num(tx)
    return tx

def remove_features(tX):
    """Modify the original data set to treat undefined values."""
    
    nan_features = list(np.where(np.all(np.isnan(tX), axis=0))[0])

    std_zero_features = list(np.where(np.nanstd(tX, axis=0)==0)[0])
    
    
    indices = np.concatenate((nan_features, std_zero_features))
    

    tx = np.delete(tX, indices, axis=1)
            
    return tx, indices     
            
def standardize(x,id_axis):
    """Standardize the original data set."""
    mean_x = np.nanmean(x,axis=id_axis)
    x = x - mean_x
    std_x = np.nanstd(x,axis=id_axis)
    
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

def build_poly_all_features(x, degree):
    "build polynomial for all features"
    num_features = x.shape[1]
    tx = x
    
    for deg in range(2,degree+1):
        for feature in range(num_features):
            tx = np.c_[tx, x[:,feature]**deg]
    return tx

def build_poly_superior_degree(x, superior_degree):
    "build polynomial with one degree more than the one given"
    num_features = x.shape[1]
    tx = x
    for feature in range(num_features):
        tx = np.c_[tx, x[:,feature]**superior_degree]
    return tx

def classify (y):
    for i in range(len(y)):
        if y[i] == -1:
            y[i] = 0
    return y