import numpy as np

def sigmoid(t):
    """Apply sigmoid function on t."""
    return 1/(1+np.exp(-t))

def compute_gradient(y, tx, w):
    """Compute the gradient for the linear regression."""
    num_samples = len(y)
    error = y-np.dot(tx,w)
    grad_w = (-1/num_samples)*np.dot(tx.transpose(),error)
    return error, grad_w

def logistic_gradient(y, tx, w):
    """Compute the gradient of logistic loss."""
    gradient = np.dot(tx.transpose(),(sigmoid(np.dot(tx,w))-y))
    return gradient

def reg_logistic_gradient(y, tx, w, lambda_):
    """Compute the gradient of regularized logistic loss."""
    gradient = logistic_gradient(y, tx, w) + 2 * lambda_ * w
    return gradient
