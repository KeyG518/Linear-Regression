"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
from scipy import nanmean
import math 

def load_unicef_data():
    """Loads Unicef data from CSV file.

    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none

    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """
    fname = 'SOWC_combined_simple.csv'

    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_', encoding='latin1')
    # Strip countries title from feature names.
    features = data.axes[1][1:]
    # Separate country names from feature values.
    countries = data.values[:,0]
    values = data.values[:,1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values,dtype='float64')

    # Modify NaN values (missing values).
    mean_vals = nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)


def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    mvec = x.mean(0)
    stdvec = x.std(axis=0)
    
    return (x - mvec)/stdvec
    






def design_matrix(x, degree, basis=None, mu=0, s=1):
    """ Compute a design matrix Phi from given input datapoints and basis.

    Args:
      x matrix of input datapoints
      basis string name of basis
      degree is degree of polynomial to use (only for polynomial basis)
      mu,s are parameters of sigmoid basis

    Returns:
      phi design matrix
    """
    num_features = x.shape[0]
    vec_1 = np.ones((num_features, 1), dtype = int)


    if basis == 'polynomial':
        phi = np.hstack((vec_1,x))

        if degree > 1:
          for i in range(2, degree + 1):
            temp_x = np.power(x, i)
            phi = np.hstack((phi, temp_x))   

    elif basis == 'sigmoid':
        y = (mu - x) / s
        sigm = 1.0 / (1.0 + np.exp(y))
        phi = np.hstack((vec_1,sigm))
        #print(phi)
    else: 
        assert(False), 'Unknown basis %s' % basis
  

    return phi

def design_matrix_NoBias(x, degree, basis=None):
    """ Compute a design matrix Phi from given input datapoints and basis.

    Args:
      x matrix of input datapoints
      basis string name of basis
      degree is degree of polynomial to use (only for polynomial basis)

    Returns:
      phi design matrix
    """
    num_features = x.shape[0]


    if basis == 'polynomial':
        phi = x

        if degree > 1:
          for i in range(2, degree + 1):
            temp_x = np.power(x, i)
            phi = np.hstack((phi, temp_x))   

    elif basis == 'sigmoid':
        phi = None
    else: 
        assert(False), 'Unknown basis %s' % basis
  

    return phi

def linear_regression(x, t, basis, reg_lambda=0, degree=0, mu=0, s=1):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter(>=0)
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      mu,s are parameters of Gaussian basis

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """

    # Construct the design matrix.
    # Pass the required parameters to this function
    
    phi = design_matrix(x, degree, basis, mu, s)

    phi_sudoinv = np.linalg.pinv(phi)
    phi_i = np.identity(phi.shape[1])

    # Learning Coefficients
    if reg_lambda > 0:
        # regularized regression
        w = np.linalg.inv(reg_lambda*phi_i + phi.T.dot(phi)).dot(phi.T).dot(t)
    else:
        # no regularization
        w = phi_sudoinv.dot(t)

    # Measure root mean squared error on training data.
    N = t.shape[0]
    train_err = math.sqrt(np.sum(np.square(t - phi.dot(w))) / N)

    return (w, train_err)


def evaluate_regression(w, x_test, t_test, basis=None, degree = None):
    """Evaluate linear regression on a dataset.

    Args:
      w vector of learned coefficients
      x is test inputs
      t is test targets

    Returns:
      t_est values of regression on inputs
      err RMS error on training set if t is not None
      """
    phi_test = design_matrix(x_test, degree, basis)

    t_est = phi_test.dot(w)
    N = t_est.shape[0]

    err = math.sqrt(np.sum(np.square(t_est - t_test)) / N)

    return (t_est, err)

def linear_regression_WithoutBias(x, t, basis, reg_lambda=0, degree=0, mu=0, s=1):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter(>=0)
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      mu,s are parameters of Gaussian basis

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """

    # Construct the design matrix.
    # Pass the required parameters to this function
    
    phi = design_matrix_NoBias(x, degree, basis)

    phi_sudoinv = np.linalg.pinv(phi)
    phi_i = np.identity(phi.shape[1])

    # Learning Coefficients
    if reg_lambda > 0:
        # regularized regression
        w = np.linalg.inv(reg_lambda*phi_i + phi.T.dot(phi)).dot(phi.T).dot(t)
    else:
        # no regularization
        w = phi_sudoinv.dot(t)

    # Measure root mean squared error on training data.
    N = t.shape[0]
    train_err = math.sqrt(np.sum(np.square(t - phi.dot(w))) / N)

    return (w, train_err)

def evaluate_regression_WithoutBias(w, x_test, t_test, basis=None, degree = None):
    """Evaluate linear regression on a dataset.

    Args:
      w vector of learned coefficients
      x is test inputs
      t is test targets

    Returns:
      t_est values of regression on inputs
      err RMS error on training set if t is not None
      """
    phi_test = design_matrix_NoBias(x_test, degree, basis)

    t_est = phi_test.dot(w)
    N = t_est.shape[0]

    err = math.sqrt(np.sum(np.square(t_est - t_test)) / N)

    return (t_est, err)