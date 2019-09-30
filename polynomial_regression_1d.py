#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]


# Complete the linear_regression and evaluate_regression functions of the assignment1.py
# Pass the required parameters to these functions

train_err = {}
test_err = {}

for i in range(7, 15):
	
	x = values[:, i]
	x_train = x[0:N_TRAIN]
	x_test = x[N_TRAIN:]
	(w, tr_err) = a1.linear_regression_WithoutBias(x_train, t_train, "polynomial", degree = 3)
	(t_est, te_err) = a1.evaluate_regression_WithoutBias(w, x_test, t_test,  "polynomial", degree = 3)
	print(train_err)

	train_err[i] = tr_err
	test_err[i] = te_err


# Produce a plot of results.
plt.rcParams.update({'font.size': 15})
xaxis = np.arange(8)
plt.bar(xaxis-0.2, list(train_err.values()), width = 0.4)
plt.bar(xaxis+0.2, list(test_err.values()), width = 0.4)
plt.ylabel('RMS')
plt.legend(['Training error','Testing error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()
