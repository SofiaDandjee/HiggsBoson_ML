# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def predict_accuracy (y, tx, w, regression = 'linear'):
    y_pred_training = predict_labels(w, tx,regression)
    num_samples = len(y_pred_training)
    count = 0
    if regression == 'linear':
        for i in range(num_samples):
            if (y_pred_training[i] == y[i]):
                count += 1
    elif regression == 'logistic':
        for i in range(num_samples):
            if (y_pred_training[i] == -1 and y[i] == 0):
                count += 1
            if (y_pred_training[i] == 1 and y[i] == 1):
                count +=1
                
    accuracy = (count *100) / num_samples
    return accuracy

def predict_labels(weights, data, regression = 'linear'):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    if regression == 'linear':
        y_pred[np.where(y_pred <= 0)] = -1
        y_pred[np.where(y_pred > 0)] = 1
    elif regression == 'logistic':
        y_pred[np.where(y_pred <= 0.5)] = -1
        y_pred[np.where(y_pred > 0.5)] = 1
        
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
