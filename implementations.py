import numpy as np
from data_helpers import *
from compute_gradient import *
from cost import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma, printing=True):
    "Least squares using gradient descent"
    w = initial_w
    losses = []
    thres = 1e-8
    
    for n_iter in range(max_iters):
        
        _,grad = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        
        w = w - gamma*grad
        
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1]-losses[-2]) < thres:
            #TODO: write a comment to explain to the user why it broke
            break
            
        if printing==True:    
            print("Gradient Descent({bi}/{ti}): loss={l}, weights = {we}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, we = w ))
        
    return w, losses[-1]

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma, printing=True):
    "Least squares using stochastic gradient descent"
    w = initial_w
    
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        
            # compute gradient and loss
            _, grad = compute_gradient(minibatch_y,minibatch_tx,w)
            loss = compute_loss(minibatch_y,minibatch_tx,w)
    
            # update w by gradient
            w = w - gamma*grad
        
        if printing==True: 
            print("SGD({bi}/{ti}): loss={l}, weights={w}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w=w))   
    
    return w, loss

def least_squares(y, tx):
    """calculate the least squares."""
    n = len(y)
    w = np.linalg.solve (np.dot(tx.transpose(),tx),np.dot(tx.transpose(),y))
    loss = compute_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):

    n = len(y)
    a = np.dot(tx.transpose(),tx)+(2*n)*lambda_*np.identity(tx.shape[1])
    b = np.dot(tx.transpose(),y)
    w =np.linalg.solve(a, b)

    loss = compute_loss(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    losses = []
    w = initial_w
    num_samples = len(y)
    for iter in range(max_iters):
        sum_loss = 0
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=1, num_batches = num_samples):
            
            gradient = logistic_gradient (batch_y,batch_tx,w)
            w -= gamma*gradient
            
            loss = logistic_loss (batch_y,batch_tx,w)
            sum_loss += loss
            
            losses.append(loss)
        av_loss = sum_loss/num_samples
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
                  bi=iter, ti=max_iters - 1, l=av_loss))
            

    loss = logistic_loss(y,tx,w)/num_samples
    
    return w,loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma): 
    num_samples = len(y)
    losses = []
    w = initial_w
    for iter in range(max_iter):
        sum_loss = 0
        for batch_y, batch_tx in batch_iter(y, tx, batch_size=1, num_batches = num_samples):
            
            gradient = reg_logistic_gradient (batch_y,batch_tx,w, lambda_)
            w -= gamma*gradient
            loss = reg_logistic_loss (batch_y,batch_tx,w,lambda_)
            sum_loss += loss
            
        losses.append(loss)
        #Average of the loss after an interation over all the samples
        # 1 iteration = num_samples * batches of 1 sample used
        av_loss = sum_loss/1
        #print("Gradient Descent({bi}/{ti}): loss={l}".format(
                  #bi=iter, ti=max_iter - 1, l=av_loss))
            
    #Calculate loss over the whole training set
    loss = reg_logistic_loss(y,tx,w,lambda_)
    print(loss)
    return w,loss