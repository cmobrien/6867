import pdb
import random
import pylab as pl
from scipy.optimize import fmin_bfgs
import numpy as np
import sys

INFINITY = 10000000

def findMin(f, guess, gradient, step_size = 0.1, convergence_criterion = 0.1):

  functionCalls = 0

  oldLocation = guess  
  bestDirection = gradient(f, guess)  
  currentLocation = guess + bestDirection * step_size
  
  endCounter = 0
  
  while np.linalg.norm(f(currentLocation) - f(oldLocation)) > convergence_criterion and endCounter < 100000:
    
    functionCalls += 2 # 2 function calls for the while loop
  
    # terminate if we didn't move much
    bestDirection = gradient(f, currentLocation)
    functionCalls += 2 * len(currentLocation) # 2^d function calls to try each cardinal direction
    
    oldLocation = currentLocation
    
    currentLocation = currentLocation + bestDirection * step_size
    
    endCounter += 1
    
    #print currentLocation
  
  print "The dumb gradient descent function took", functionCalls, "function calls."
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
  if type(X[0]) == float or type(X[0]) == int or type(X[0]) == np.float64:
    return np.array([[x ** i for i in range(order + 1)] for x in X])
  else:
    return np.array([[x[0] ** i for i in range(order + 1)] for x in X])

def blogDesignMatrix(X):
  return np.array([np.append([1], x) for x in X])

def regressionFit(X, Y, phi):
  return np.dot(np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T), Y) 

# X is an array of N data points (one dimensional for now), that is, NX1
# Y is a Nx1 column vector of data values
# order is the order of the highest order polynomial in the basis functions
def regressionPlot(X, Y, order):
    pl.plot(X.T.tolist()[0],Y.T.tolist()[0], 'bo')

    # You will need to write the designMatrix and regressionFit function

    # constuct the design matrix (Bishop 3.16), the 0th column is just 1s.
    phi = designMatrix(X.T.tolist()[0], order)
    # compute the weight vector
    w = regressionFit(X, Y, phi)

    print 'w', w
    # produce a plot of the values of the function 
    pts = [p for p in pl.linspace(min(X), max(X), 100)]
    Yp = pl.dot(w.T, designMatrix(pts, order).T)
    pl.plot(pts, Yp.tolist()[0], 'g')
    pl.show()

def testPlot(X, Y, w):
    print "W", w
    fig1 = pl.figure(1)
    fig1.set_figheight(11)
    fig1.set_figwidth(8.5)

    rect = fig1.patch
    rect.set_facecolor('white') 

    ax = fig1.add_subplot(1,1,1)


    ax.plot(X.T.tolist()[0],Y.T.tolist()[0], 'o', mfc="none", mec='b', mew = '2')
    # produce a plot of the values of the function 
    pts = np.array([p for p in pl.linspace(min(X), max(X), 100)])
    Yp = pl.dot(w.T, designMatrix(pts, (len(w) - 1)).T)
    ax.plot(pts, Yp, '#00ff00', linewidth = 2)
    pl.show()


def linePlot(f):
    fig1 = pl.figure(1)
    fig1.set_figheight(11)
    fig1.set_figwidth(8.5)

    rect = fig1.patch
    rect.set_facecolor('white') 

    ax = fig1.add_subplot(1,1,1)


    # produce a plot of the values of the function 
    pts = np.array([p for p in pl.linspace(10000, 60000, 100)])
    #pts = np.array([p for p in pl.linspace(0, 4, 41)])
    Yp = [f(pt) for pt in pts]
    ax.plot(pts, Yp, '#00ff00', linewidth = 2)
    pl.xlabel(r'$\lambda$')
    pl.ylabel(r'$SSE$')
    pl.show()

def plot_fn(f_list):
  for f in f_list:
    pts = np.array([p for p in pl.linspace(31740, 31750, 11)])
    Yp = np.array([f(pt) for pt in pts])
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

def gradient(f, currentLocation):
  l = []
  for i in range(len(currentLocation)):
    p = np.concatenate([currentLocation[:i], [currentLocation[i] + 0.05], currentLocation[i + 1:]])
    m = np.concatenate([currentLocation[:i], [currentLocation[i] - 0.05], currentLocation[i + 1:]])
    l.append((f(m) - f(p)))
  return np.array(l)

def SSE(X, Y):
  return (lambda w:
      sum([
        (sum([w[j] * (X[i] ** j) for j in range(len(w))]) - Y[i])**2
        for i in range(len(X))
        ])
      )


def SE(X, Y):
  return (lambda w:
      sum([
        (abs(sum([w[j] * (X[i] ** j) for j in range(len(w))]) - Y[i]))
        for i in range(len(X))
        ])
      )

def blog_SSE(X, Y, w):
  s = 0
  for i in range(len(X)):
    guess = w[0] + sum([w[j + 1] * X[i][j] for j in range(len(w) - 1)])
    diff = guess - Y[i]
    s += (diff ** 2)
  return s

#print blog_SSE(np.array([[1, 1], [2, 2]]), np.array([1, 2]), [0, 2, -1])

#print SSE([1, 2, 3], [2, 3, 4])([0, 1, 1])

f = SSE(X, Y)
#testPlot(X, Y, findMin(f,[0], gradient, step_size = 0.1, convergence_criterion = 0.00001))
#testPlot(X, Y, findMin(f,[0, 0], gradient, step_size = 0.1, convergence_criterion = 0.00001))
#testPlot(X, Y, findMin(f,[0, 0, 0, 0], gradient, step_size = 0.1, convergence_criterion = 0.00001))
#testPlot(X, Y, findMin(f,[20.0, 20.0, -20.0, 20], gradient, step_size = 0.1, convergence_criterion = 0.00001))
#testPlot(X, Y, findMin(f,[0,0,0,0,0,0,0,0,0,0], gradient, step_size = 0.1, convergence_criterion = 0.0001))

#fun = SSE(np.array([[0], [1], [2]]), np.array([[0], [1], [4]]))
#testPlot(np.array([[0], [1], [2]]), np.array([[0], [1], [4]]), findMin(fun, [0.0, 0.0, 0.0], gradient, step_size = 0.01, convergence_criterion = 0.00001))

#testPlot(np.array([[0], [1], [2]]), np.array([[1], [2], [5]]), gradient_descent.findMin(SSE([0, 1, 2], [1, 2, 5]), [0, 0, 1], gradient_descent.gradient))

def regressAData():
    return getData('regressA_train.txt')

def regressBData():
    return getData('regressB_train.txt')

def validateData():
    return getData('regress_validate.txt')

#testPlot(X, Y, fmin_bfgs(f, [0.0]))
#testPlot(X, Y, fmin_bfgs(f, [0.0, 0.0]))
#testPlot(X, Y, fmin_bfgs(f, [0.0, 0.0, 0.0, 0.0]))
#testPlot(X, Y, fmin_bfgs(f, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

def ridge_regression(X, order, y, lamda):
    A = designMatrix(X, order)
    Z, averages = centralizedDataMatrix(A)
    I = np.identity(Z.shape[1])
    w = np.dot(np.dot(np.linalg.inv(np.dot(Z.T, Z) + lamda * I), Z.T), y)
    w_list = [k[0] for k in w.tolist()]
    y_ave = (sum([val[0] for val in y]) / float(y.shape[0]))
    w_0 = y_ave - np.dot(np.array(w_list), np.array(averages))
    return [w_0] + w_list

def blog_ridge_regression(X, order, y, lamda):
    A = blogDesignMatrix(X)
    Z, averages = centralizedDataMatrix(A)
    I = np.identity(Z.shape[1])
    w = np.dot(np.dot(np.linalg.inv(np.dot(Z.T, Z) + lamda * I), Z.T), y)
    w_list = [k[0] for k in w.tolist()]
    y_ave = (sum([val[0] for val in y]) / float(y.shape[0]))
    w_0 = y_ave - np.dot(np.array(w_list), np.array(averages))
    return [w_0] + w_list





#print ridge_regression(np.array([[1,2,3],[2,4,6],[3,6,9]]), [4,8,12], 0)    

def minimizeL1Norm(data_matrix, y, lamda):
    def absoluteError(weight):
        #print weight
        errorVector = np.dot(data_matrix, weight) - [e[0] for e in y]
        regularization = lamda * sum([abs(i)**2 for i in weight[1:]])
        #print "ERROR", sum([abs(j) for j in errorVector])
        #print "REG", regularization
        return sum([abs(j) for j in errorVector]) + regularization
    
    #return findMin(absoluteError, np.array([0]*data_matrix.shape[1]), gradient) 
    #return findMin(absoluteError, np.array([0, 1]), gradient) 
    return findMin(absoluteError, np.array([1]*data_matrix.shape[1]), gradient, step_size =0.001, convergence_criterion = 0.0001)
    #return fmin_bfgs(absoluteError, np.array([0]*data_matrix.shape[1]))

def minimizeQuadraticErrorWithWeightPunishment(data_matrix, y, lamda, q=1):
    
    def quadraticErrorPlusWeightPunishment(weight):
        errorVector = np.dot(data_matrix, weight) - [e[0] for e in y]
        return sum([j**2 for j in errorVector]) + lamda*sum([abs(i)**q for i in weight[1:]])
    
    return findMin(quadraticErrorPlusWeightPunishment, np.array([1, 1, 1, 1, 1, 1]), gradient, step_size = 0.01, convergence_criterion = 0.001)

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

def get_blog_feedback_train():
  X = np.loadtxt("BlogFeedback/x_train.csv", delimiter=',')
  Y = np.loadtxt("BlogFeedback/y_train.csv", delimiter=',')
  return np.array(X), np.array([[y] for y in Y])


def get_blog_feedback_val():
  X = np.loadtxt("BlogFeedback/x_val.csv", delimiter=',')
  Y = np.loadtxt("BlogFeedback/y_val.csv", delimiter=',')
  return np.array(X), np.array([[y] for y in Y])


def get_blog_feedback_test():
  X = np.loadtxt("BlogFeedback/x_test.csv", delimiter=',')
  Y = np.loadtxt("BlogFeedback/y_test.csv", delimiter=',')
  return np.array(X), np.array([[y] for y in Y])

#w = ridge_regression(X.T.tolist()[0], int(sys.argv[1]), Y, float(sys.argv[2]))
#print testPlot(X, Y, np.array(w))

#print centralizedDataMatrix(designMatrix([1, 2, 3], 3))

if __name__ != "__main__":
  X_val, Y_val = validateData()
  X, Y = regressBData()
  #w = minimizeQuadraticErrorWithWeightPunishment(designMatrix(X, int(sys.argv[1])), Y, float(sys.argv[2]))
  w = minimizeL1Norm(designMatrix(X, int(sys.argv[1])), Y, float(sys.argv[2]))
  #w = ridge_regression(X.T.tolist()[0], int(sys.argv[1]), Y, float(sys.argv[2]))
  print testPlot(X, Y, np.array(w))
  print testPlot(X_val, Y_val, np.array(w))

if __name__ != "__main__":
  X, Y = get_blog_feedback_train()
  print X[0]
  print len(X[0])
  print Y
  w = blog_ridge_regression(X, 1, Y, float(sys.argv[1]))
  print w

if __name__ != "__main__":
  
  quadraticBowl = lambda x: x[0]**2 + x[1]**2 
    
  f = lambda x : dumbGradient(quadraticBowl, x)
  print fmin_bfgs(quadraticBowl, np.array([600, 100]), norm=-float("Inf"), full_output=True)[4]
 
  print findMin(quadraticBowl, np.array([2, 2]), dumbGradient, step_size=50, convergence_criterion=1)

  print dumbGradient(quadraticBowl, np.array([2, 0]))
  print dumbGradient(quadraticBowl, np.array([2, 2]))
  print dumbGradient(quadraticBowl, np.array([2, -3]))


def w_fun(X, order, y):
  return lambda lam: ridge_regression(X, order, y, lam)

def w_blog_fun(X, order, y):
  return lambda lam : blog_ridge_regression(X, order, y, lam)

if __name__ != "__main__":
  X_val, Y_val = validateData()
  X, Y = regressAData()
  def SSE_val(lam):
    w = w_fun(X, 4, Y)(lam)
    print w
    a = SSE(X_val, Y_val)(w)
    print "lambda:", lam, "SSE:", a
    return a
  #print SSE_val(0)
  #linePlot(SSE_val)
  #print fmin_bfgs(SSE_val, np.array([0]))

# DO THIS!
if __name__ == "__main__":
  print "START"
  X_val, Y_val = get_blog_feedback_val()
  #print Y_val.shape
  #X_a, Y_a = get_blog_feedback_val()
  #print Y_a.shape
  #print "LOADED VAL DATA"
  X, Y = get_blog_feedback_train()
  print "LOADED TRAINING DATA"
  def SSE_val(lam):
    print "CALL"
    w = w_blog_fun(X, 1, Y)(lam)
    for i in range(len(w)):
      if abs(w[i]) > 1:
        print i, w[i]
    a = blog_SSE(X_val, Y_val, w)
    print "lambda:", lam, "SSE:", a
    return a
  #print SSE_val(31746)
  linePlot(SSE_val)
  #print fmin_bfgs(SSE_val, np.array([10000]))

def w_fun_minimizel1norm(X, order, Y):
  return lambda lam: minimizeL1Norm(designMatrix(X, order), Y, lam)


if __name__ != "__main__":
  X_val, Y_val = validateData()
  X, Y = regressBData()
  #X = np.array([[1], [2], [3]])
  #Y = np.array([[2], [3], [4]])
  def SSE_val(lam):
    w = w_fun_minimizel1norm(X, 5, Y)(lam)
    #print w
    a = SE(X_val, Y_val)(w)
    print "lambda:", lam, "SSE:", a
    return a
  #print SSE_val(1.4)
  linePlot(SSE_val)
  #print fmin_bfgs(SSE_val, np.array([0]))
  #w = minimizeL1Norm(designMatrix(X, 1), Y, 5)
  #print w
  #testPlot(X, Y, w)

#print minimizeL1Norm(designMatrix(np.array([[1], [2], [3]]), 1), np.array([[2], [3], [4]]), 0.2222)
def w_fun_last(X, order, Y):
  return lambda lam: minimizeQuadraticErrorWithWeightPunishment(designMatrix(X, order), Y, lam)

if __name__ != "__main__":
  X_val, Y_val = validateData()
  X, Y = regressBData()
  def SSE_val(lam):
    w = w_fun_last(X, 5, Y)(lam)
    print w
    a = SSE(X_val, Y_val)(w)
    print "lambda:", lam, "SSE:", a
    return a
  #print SSE_val(0)
  linePlot(SSE_val)


if __name__ != "__main__":
  quadraticBowl = lambda x : x[0]**2 + x[1]**2
  print gradient(quadraticBowl, [2, 0])
  print gradient(quadraticBowl, [3, 5])
  print gradient(quadraticBowl, [-4, 8])
  print findMin(quadraticBowl, [10, 10], gradient, step_size = 0.1, convergence_criterion = 0.0001)
  print fmin_bfgs(quadraticBowl, [5, 5])
  f = lambda x: x[0]**4 - x[0]**3 - x[0]**2 + x[0]
  print findMin(f, [-5], gradient, step_size = 0.1, convergence_criterion = 0.0001)
  print fmin_bfgs(f, [-5])
