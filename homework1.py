import pdb
import random
import pylab as pl
from scipy.optimize import fmin_bfgs
import numpy as np
import sys

INFINITY = 10000000

def findMin(f, guess, gradient, step_size = 0.1, convergence_criterion = 0.1):
  oldLocation = guess  
  bestDirection = gradient(f, guess)  
  currentLocation = guess + bestDirection * step_size
  
  endCounter = 0
  
  while np.linalg.norm(f(currentLocation) - f(oldLocation)) > convergence_criterion and endCounter < 10000:
    # terminate if we didn't move much
    bestDirection = gradient(f, currentLocation)
    
    oldLocation = currentLocation
    
    currentLocation = currentLocation + bestDirection * step_size\
    
    endCounter += 1
    
    print currentLocation
  
  return currentLocation

def dumbGradient(f, currentLocation):
  bestScore = INFINITY
  bestDirection = None
  
  # dimensionality of problem
  d = len(currentLocation)
  
  for dimension in range(d):
  # iterate over each of the standard basis vectors
    
    basisVector = np.array([0]*dimension + [1] + [0]*(d-dimension-1))
    currentScore = f(currentLocation + basisVector)
    
    if currentScore < bestScore:
      bestDirection = basisVector
      bestScore = currentScore
      
    basisVector = np.array([0]*dimension + [-1] + [0]*(d-dimension-1))
    currentScore = f(currentLocation + basisVector)
    
    if currentScore < bestScore:
      bestDirection = basisVector
      bestScore = currentScore
      
  return bestDirection

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

#X, Y = bishopCurveData()
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

#f = SSE(X, Y)
#testPlot(X, Y, gradient_descent.findMin(f, [0.0, 0.0, 0.0, 17.0], gradient_descent.gradient))
    
#testPlot(np.array([[0], [1], [2]]), np.array([[1], [2], [5]]), gradient_descent.findMin(SSE([0, 1, 2], [1, 2, 5]), [0, 0, 1], gradient_descent.gradient))
def regressAData():
    return getData('regressA_train.txt')

def regressBData():
    return getData('regressB_train.txt')

def validateData():
    return getData('regress_validate.txt')

#print fmin_bfgs(f, [0.0, 0.0, 0.0, 0.0])

def ridge_regression(X, order, y, lamda):
    A = designMatrix(X, order)
    Z, averages = centralizedDataMatrix(A)
    I = np.identity(Z.shape[1])
    w = np.dot(np.dot(np.linalg.inv(np.dot(Z.T, Z) + lamda * I), Z.T), y)
    w_list = [k[0] for k in w.tolist()]
    print np.array(w_list)
    print np.array(averages)
    y_ave = (sum([val[0] for val in y]) / float(y.shape[0]))
    print y_ave
    w_0 = y_ave - np.dot(np.array(w_list), np.array(averages))
    return [w_0] + w_list


#print ridge_regression(np.array([[1,2,3],[2,4,6],[3,6,9]]), [4,8,12], 0)    

def minimizeL1Norm(data_matrix, y):
    
    def absoluteError(weight):
        errorVector = np.dot(data_matrix, weight) - y
        return sum([sum([abs(j) for j in i]) for i in errorVector])
    
    return findMin(absoluteError, np.array([0]*data_matrix.shape[1]), dumbGradient) 

def minimizeQuadraticErrorWithWeightPunishment(data_matrix, y, lamda, q=1):
    
    def quadraticErrorPlusWeightPunishment(weight):
        errorVector = np.dot(data_matrix, weight) - y
        return sum([sum([j**2 for j in i]) for i in errorVector]) + lamda*sum([abs(i)**q for i in weight])
    
    return findMin(quadraticErrorPlusWeightPunishment, np.array([0]*data_matrix.shape[1]), dumbGradient)

#print minimizeQuadraticErrorWithWeightPunishment(np.array([[1],[1],[2]]), np.array([[6],[7],[8]]), 1, 3)


def centralizedDataMatrix(dataMatrix):
  centralized = []
  averages = []
  X = dataMatrix.tolist()
  for i in range(1, len(X[0])):
    averages.append(sum([X[j][i] for j in range(len(X))]) / float(len(X)))
  for row_num in range(len(X)):
    row = dataMatrix[row_num:row_num + 1,1:].tolist()[0]
    centralized.append([row[k] - averages[k] for k in range(len(row))])
  return np.array(centralized), averages

#print centralizedDataMatrix(designMatrix([1, 2, 3], 3))

if __name__ == "__main__":
  X_val, Y_val = validateData()
  X, Y = regressAData()
  w = ridge_regression(X.T.tolist()[0], int(sys.argv[1]), Y, int(sys.argv[2]))
  print testPlot(X, Y, np.array(w))
  print testPlot(X_val, Y_val, np.array(w))


