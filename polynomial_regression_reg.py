#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)


N_TRAIN = 100
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

# declear lambda
lamda =[0,0.01,0.1,1,10,100,1000,10000]


# Complete the linear_regression and evaluate_regression functions of the assignment1.py
# Pass the required parameters to these functions

train_err = {}
test_err = {}
for j in lamda:

	train_err_temp = 0
	test_err_temp = 0
	
	for i in range(0, 10):
		# start = (i-1)*10
		# end = i*10
		#print(start, end)
		validset = range(i * 10,(i * 10) + 10)
		valid_x = x_train[np.arange(100),:]        
		valid_t = t_train[np.arange(100)]
		end = np.arange(i * 10,(i * 10) + 10)
		TrainDataindex = np.delete(np.arange(100),end)
		reg_train_x = x_train[TrainDataindex,: ]
		reg_train_t = t_train[TrainDataindex]

		# valid_x = x[start:end, :]
		# valid_t = targets[start:end]
		# reg_train_x = np.concatenate((x[:start, :], x[end:, :]), axis = 0)
		# reg_train_t = np.concatenate((targets[:start], targets[end:]), axis = 0)


		(w, tr_err) = a1.linear_regression(reg_train_x, reg_train_t, "polynomial", degree = 2, reg_lambda = j)
		(t_est, te_err) = a1.evaluate_regression(w, valid_x, valid_t,  "polynomial", degree = 2)
		# print(w)


		train_err_temp += tr_err / 10
		test_err_temp += te_err / 10

	# train_err_temp = train_err_temp / 10
	# test_err_temp = test_err_temp / 10


	train_err[j] = train_err_temp
	test_err[j] = test_err_temp
print(list(test_err.values()))

#result = list(test_err[0]).append(test_err)
# Produce a plot of results.
plt.rcParams.update({'font.size': 15})
# plt.plot(list(train_err.keys()), list(train_err.values()))
# plt.plot(list(test_err.keys()), list(test_err.values()))
# plt.semilogx(list(train_err.keys()), list(train_err.values()))
plt.semilogx(list(test_err.keys()), list(test_err.values()))
plt.ylabel('Avg RMS')
plt.legend(['CV error','Testing error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()
