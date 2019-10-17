import numpy as np

def compute_loss(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2* len(e))
    return mse

def compute_loss_rmse(y, tx, w):
    """compute the loss by rmse."""
    rmse = (2*compute_loss(y,tx,w))**(1/2)
    return rmse

def calculate_logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    loss = 0
    num_samples = len(y)
    
    for n in range (num_samples):
        loss = loss + np.log(1+np.exp(np.dot(tx[n,:].T,w))) - np.dot(y[n]*tx[n,:].T,w)
    return loss

def calculate_reg_logistic_loss(y, tx, w,lambda_):
    loss = calculate_logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    return loss