from problem1 import *
from plotBoundary import *
import numpy
from load_kaggle import *

def multiclass(X, Y, C):
  classes = {}
  for i in range(len(Y)):
    if Y[i] not in classes:
      classes[Y[i]] = {i: True}
    else:
      classes[Y[i]][i] = True
  Yk = {}
  f = gaussian_kernel(X, .1)
  res = {}

  wk = {}
  for k in classes:
    print k
    newY = []
    for i in range(len(Y)):
      if i in classes[k]:
        newY.append(1)
      else:
        newY.append(-1)
    alpha = solve_qp(f, X, newY, C)
    wk[k] = get_weights(X, newY, alpha)
  return wk

def predictSVM(x, wk):
  m = None
  best = None
  for k in wk:
    w = wk[k]
    val = w[0] + sum([w[i + 1] * x[i] for i in range(len(x))]) 
    if best == None or val > m:
      best = k
      m = val
  return best

X, Y = load_train()
X_val, Y_val = load_validate()
X_test, Y_test = load_test()

def run_tests(C_options):
  errors = {}

  for C in C_options:
    print "C = ", C
    wk = multiclass(X, Y, C)

    valid_error = 0
    for i in range(len(X_val)):
      if predictSVM(X_val[i], wk) != Y_val[i]:
        valid_error += 1
  
  
    test_error = 0
    for i in range(len(X_test)):
      if predictSVM(X_test[i], wk) != Y_test[i]:
        test_error += 1
    errors[C] = (valid_error, test_error)
    print "C = ", C, valid_error
  return errors

print run_tests([.10])

#X = numpy.array([[-2, 1], [-4, 1], [3, 0], [4, -1], [-4, 3], [-2, 3], [3, 5], [4, 7], [4, 4], [5, 2], [6, 1], [6, 5]])
#Y = numpy.array([[1], [1], [1], [1], [2], [2], [2], [2], [3], [3], [3], [3]])

#X = numpy.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
#Y = numpy.array([[1], [2], [3], [4]])
#X_list = X.tolist()
#Y_list = [y[0] for y in Y.tolist()]
#

#wk = multiclass(X_list, Y_list)
#print wk
#print predictSVM([1, 2], wk)
#
#def pred(x):
#  return predictSVM(x, wk)
#plotDecisionBoundary(X, Y, pred, [1, 2, 3, 4])

