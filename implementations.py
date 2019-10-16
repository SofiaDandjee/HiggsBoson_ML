import numpy as np

def least_squares_GD(y, tx, initial_w, max_iters, gamma):

    #ws = [initial_w]
    #losses = []
    w = initial_w
    for n_iter in range(max_iters):
        
        error,grad = compute_gradient(y,tx,w)
        loss = compute_loss(y,tx,w)
        
        new_w = w - gamma*grad
        w = new_w
        
        #ws.append(w)
        #losses.append(loss)
        
        print("Gradient Descent({bi}/{ti}): loss={l}, weights = {we}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, we = w ))
        
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        
            # compute gradient and loss
        
            grad = compute_stoch_gradient(minibatch_y,minibatch_tx,w)
            loss = compute_loss(minibatch_y,minibatch_tx,w)
        
            # update w by gradient
        
            new_w = w - gamma*grad
            w = new_w
        
            # store w and loss
            ws.append(w)
            losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss

def least_squares(y, tx):
    """calculate the least squares."""
    n = len(y)
    w = np.linalg.solve (np.dot(tx.transpose(),tx),np.dot(tx.transpose(),y))
    
    return w, loss

def ridge_regression(y, tx, lambda_):

    n = len(y)
    
    a = np.dot(tx.transpose(),tx)+(2*n)*lambda_*np.identity(tx.shape[1])
    b = np.dot(tx.transpose(),y)
    

    w =np.linalg.solve(a, b)
    
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
    return w,loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma): 

    return w,loss

def compute_loss(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2* len(e))
    return mse

def compute_rmse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    rmse = (2*mse)**(1/2)
    return rmse

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.zeros((x.shape[0],degree+1))
    for j in range (0,degree+1):
        phi[:,j] = x**j
        
    return phi

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    
    print ('Computing gradients ...')
    num_samples = len(y)
    
    pred_y = np.dot(tx,w)
    error = y-pred_y
    
    grad_w = (-1/num_samples)*np.dot(tx.transpose(),error)

    return error, grad_w

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

