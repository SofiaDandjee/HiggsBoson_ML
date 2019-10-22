import numpy as np

from compute_gradient import sigmoid 

def compute_loss(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2* len(e))
    return mse

def compute_loss_rmse(y, tx, w):
    """compute the loss by rmse."""
    rmse = (2*compute_loss(y,tx,w))**(1/2)
    return rmse

def logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    num_samples = len(y)
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    #The loss is averaged over all samples
    return np.squeeze(- loss/num_samples)

def reg_logistic_loss(y, tx, w,lambda_):
    num_samples = len(y)
    loss = logistic_loss(y, tx, w) + (lambda_ * np.squeeze(w.T.dot(w)))/num_samples
    return loss