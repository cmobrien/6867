import numpy as np
from make_features import *

def blogDesignMatrix(X):
  return np.array([np.append([1], x) for x in X])

def regressionFit(X, Y, phi):
  return np.dot(np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T), np.array(Y)) 

X, Y = get_train(8) 
Y = [[y[0]] for y in Y]
phi = blogDesignMatrix(X)
w = regressionFit(X, Y, phi)
print 'w', w
