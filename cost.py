import numpy as np

from compute_gradient import sigmoid 

def compute_loss(y, tx, w):
    """Compute the loss by mse."""
    e = y - tx.dot(w)
    mse = np.sum(e**2) / (2* len(e))
    return mse

def compute_loss_rmse(y, tx, w):
    """Compute the loss by rmse."""
    rmse = (2*compute_loss(y,tx,w))**(1/2)
    return rmse

def ridge_loss(y,tx,w,lambda_):
    """Compute the ridge loss by mse."""
    loss = compute_loss (y,tx,w) #+ lambda_*np.squeeze(w.T.dot(w)) 
    return loss

def logistic_loss(y, tx, w):
    """Compute the cost by negative log likelihood."""
    num_samples = len(y)
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def reg_logistic_loss(y, tx, w,lambda_):
    """Compute the logistic loss by negative log likelihood."""
    loss = logistic_loss(y, tx, w) #+ lambda_ * np.squeeze(w.T.dot(w))
    return loss