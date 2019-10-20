import numpy as np

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1/(1+np.exp(-t))

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    num_samples = len(y)
    error = y-np.dot(tx,w)
    grad_w = (-1/num_samples)*np.dot(tx.transpose(),error)
    return error, grad_w

def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    gradient = np.dot(tx.transpose(),(sigmoid(np.dot(tx,w))-y))
    return gradient

def calculate_reg_logistic_gradient(y, tx, w, lambda_s):
    """compute the gradient of loss."""
    gradient = calculate_logistic_gradient(y, tx, w) + 2 * lambda_ * w
    return gradient
