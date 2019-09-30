#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,:]
#x = a1.normalize_data(x)


# select the feture
feture = 10
N_TRAIN = 100
x_train = x[0:N_TRAIN,feture]
x_test = x[N_TRAIN:,feture]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

(w, tr_err) = a1.linear_regression(x_train, t_train, 'sigmoid', mu = 10000, s = 2000)
# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate
x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500).reshape(500, 1)



# TO DO::
# Perform regression on the linspace samples.
# Put your regression estimate here in place of y_ev.

x_designM = a1.design_matrix(x_ev, 3, 'sigmoid', mu = 100, s =2000)
y_ev = x_designM * w



plt.plot(x_ev, y_ev,'r,-')
plt.plot(x_train, t_train,'bo')
plt.plot(x_test, t_test, 'go')
plt.legend(['Fitted regression', 'Train data', 'Test Data'])
plt.title('Visualization of a function and some data points')
plt.show()