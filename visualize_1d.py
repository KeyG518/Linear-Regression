#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate
#x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
#x_ev = np.linspace(np.asscalar(min(min(x_train[:,f]),min(x_test[:,f]))),
#                   np.asscalar(max(max(x_train[:,f]),max(x_test[:,f]))), num=500)

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,:]
#x = a1.normalize_data(x)


# select the feture
feture = 12
print(features[feture])
N_TRAIN = 100
x_train = x[0:N_TRAIN,feture]
x_test = x[N_TRAIN:,feture]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

(w, tr_err) = a1.linear_regression(x_train, t_train, 'polynomial', 0, 3)
# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate
x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500).reshape(500, 1)


# x1_ev = np.linspace(0, 10, num=500)
# x2_ev = np.linspace(0, 10, num=50)

# TO DO::
# Perform regression on the linspace samples.
# Put your regression estimate here in place of y_ev.

x_designM = a1.design_matrix(x_ev, 3, 'polynomial')
y_ev = x_designM * w

# y1_ev = np.random.random_sample(x1_ev.shape)
# y2_ev = np.random.random_sample(x2_ev.shape)
# y1_ev = 100*np.sin(x1_ev)
# y2_ev = 100*np.sin(x2_ev)

# plt.plot(x1_ev,y1_ev,'r.-')
# plt.plot(x2_ev,y2_ev,'bo')
plt.plot(x_ev, y_ev,'r,-')
plt.plot(x_train, t_train,'bo')
plt.plot(x_test, t_test, 'go')
plt.legend(['Fitted regression', 'Train data', 'Test Data'])
plt.title('Visualization of a function and some data points')
plt.show()