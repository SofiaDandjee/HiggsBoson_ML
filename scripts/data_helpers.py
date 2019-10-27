import numpy as np

def get_jet_samples(tx):
    
    jet0_samples = np.where(tx[:,22]==0)[0]
    jet1_samples = np.where(tx[:,22]==1)[0]
    jet2_samples = np.where(tx[:,22]>=2)[0]
    
    return [jet0_samples, jet1_samples, jet2_samples]
  
            
def standardize(x):
    """Standardize the original data set."""
    mean = np.nanmean(x,axis=0)
    x = x - mean
    
    std = np.nanstd(x,axis=0)
    x = x / std
    
    return x, mean, std


def clean_data(tx):
    #replace undefined values (equal to -999 in the data) by NaN 
    tx[tx == -999] = np.nan
    
    # find columns full of NaN
    nan_features = list(np.where(np.all(np.isnan(tx), axis=0))[0])
    # find features that have constant value (standard deviation equal to 0)
    constant_features = list(np.where(np.nanstd(tx, axis=0)==0)[0])
    # remove selected columns
    indices = np.concatenate((nan_features, constant_features))
    tx = np.delete(tx, indices, axis=1)
   
    #standardize
    tx,_,_ = standardize(tx)
     
    # replace NaN values by 0    
    tx = np.nan_to_num(tx)

    return tx, indices


def augment_data(tx, y, degree) :
    
    tx = build_poly_all_features (tx,degree)
    y,tx = build_model_data(tx,y)
    
    return tx, y


def build_model_data(x, y):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


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



