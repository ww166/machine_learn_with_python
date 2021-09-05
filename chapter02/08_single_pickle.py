import pickle

import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# Input file containing data
input_file = 'data/data_singlevar_regr.txt'

# Read data
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Train and test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data
X_train, y_train = X[:num_training], y[:num_training]

# Test data
X_test, y_test = X[num_training:], y[num_training:]

# MModel persistence
output_model_file = 'data/model.pkl'

# Load the model
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

# Performance predict on test data
y_test_pred_new = regressor_model.predict(X_test)
print('\nNew mean absolute error = ', round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))
