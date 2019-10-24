import numpy as np

from compute_gradient import sigmoid 

def compute_loss(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = np.sum(e**2) / (2* len(e))
    return mse

def compute_loss_rmse(y, tx, w):
    """compute the loss by rmse."""
    rmse = (2*compute_loss(y,tx,w))**(1/2)
    return rmse

def ridge_loss(y,tx,w,lambda_):
    loss = compute_loss (y,tx,w) + lambda_*np.squeeze(w.T.dot(w)) #L2-reg
    return loss

def logistic_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    num_samples = len(y)
    pred = sigmoid(tx.dot(w))
    #print(pred)
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    #a = np.exp(tx.dot(w))
    #print((y.dot(a)).shape)
    #print(a.shape)
    #loss = sum(np.log(np.ones((num_samples,1))+a) - y*a)
    
    return np.squeeze(- loss)
    #return np.squeeze(loss)

def reg_logistic_loss(y, tx, w,lambda_):

    loss = logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    return loss