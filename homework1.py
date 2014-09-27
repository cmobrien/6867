import pdb
import random
import pylab as pl
from scipy.optimize import fmin_bfgs
import numpy as np
import gradient_descent

def designMatrix(X, order):
  return np.array([[x ** i for i in range(order + 1)] for x in X])

def regressionFit(X, Y, phi):
  return np.dot(np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T), Y) 

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values
# order is the order of the highest order polynomial in the basis functions
def regressionPlot(X, Y, order):
    pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')

    # You will need to write the designMatrix and regressionFit function

    # constuct the design matrix (Bishop 3.16), the 0th column is just 1s.
    phi = designMatrix(X.T.tolist()[0], order)
    # compute the weight vector
    w = regressionFit(X, Y, phi)

    print 'w', w
    # produce a plot of the values of the function 
    pts = [p for p in pl.linspace(min(X), max(X), 100)]
    Yp = pl.dot(w.T, designMatrix(pts, order).T)
    pl.plot(pts, Yp.tolist()[0])
    pl.show()

def testPlot(X, Y, w):
    pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'gs')
    # produce a plot of the values of the function 
    pts = np.array([p for p in pl.linspace(min(X), max(X), 100)])
    Yp = pl.dot(w.T, designMatrix(pts, (len(w) - 1)).T)
    pl.plot(pts, Yp)
    pl.show()

def getData(name):
    data = pl.loadtxt(name)
    # Returns column matrices
    X = data[0:1].T
    Y = data[1:2].T
    return X, Y

def bishopCurveData():
    # y = sin(2 pi x) + N(0,0.3),
    return getData('curvefitting.txt')

X, Y = bishopCurveData()
#regressionPlot(X, Y, 3)
#regressionPlot(X, Y, 9)

def SSE(X, Y):
  return (lambda w:
      sum([
        (sum([w[j] * (X[i] ** j) for j in range(len(w))]) - Y[i])**2
        for i in range(len(X))
        ])
      )

#print SSE([1, 2, 3], [2, 3, 4])([0, 1, 1])

f = SSE(X, Y)
testPlot(X, Y, gradient_descent.findMin(f, [0.0, 0.0, 0.0, 17.0], gradient_descent.gradient))
    
#testPlot(np.array([[0], [1], [2]]), np.array([[1], [2], [5]]), gradient_descent.findMin(SSE([0, 1, 2], [1, 2, 5]), [0, 0, 1], gradient_descent.gradient))
def regressAData():
    return getData('regressA_train.txt')

def regressBData():
    return getData('regressB_train.txt')

def validateData():
    return getData('regress_validate.txt')

print fmin_bfgs(f, [0.0, 0.0, 0.0, 0.0])

def ridge_regression(data_matrix, y, lamda):
    A = data_matrix
    AT = data_matrix.T
    I = numpy.identity(A.shape[1])
    return np.dot(np.dot(np.linalg.inv(np.dot(AT, A) + lamda * I), AT), y)
    
print ridge_regression(np.array([[1,2,3],[2,4,6],[3,6,9]]), [4,8,12], 0)    

def minimizeL1Norm(data_matrix, y, lamda):
    
    def absoluteError(weight):
        errorVector = np.dot(data_matrix, weight) - y
        return sum([abs(i) for i in errorVector])
    
    return gradient_descent.findMin(absoluteError, np.array([0]*data_matrix.shape[1]), gradient_descent.dumbGradient) 

print minimizeL1Norm(np.array([[1],[1],[2]], [[1],[2],[3]]))

