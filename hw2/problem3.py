from problem1 import *
from plotBoundary import *
import numpy

def multiclass(X, Y):
  classes = {}
  for i in range(len(Y)):
    if Y[i] not in classes:
      classes[Y[i]] = {i: True}
    else:
      classes[Y[i]][i] = True
  print classes
  Yk = {}
  f = dot_kernel(X)
  res = {}

  wk = {}
  for k in classes:
    newY = []
    for i in range(len(Y)):
      if i in classes[k]:
        newY.append(1)
      else:
        newY.append(-1)
    alpha = solve_qp(f, X, newY, 1)
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

#X = numpy.array([[-2, 1], [-4, 1], [3, 0], [4, -1], [-4, 3], [-2, 3], [3, 5], [4, 7], [4, 4], [5, 2], [6, 1], [6, 5]])
#Y = numpy.array([[1], [1], [1], [1], [2], [2], [2], [2], [3], [3], [3], [3]])

X = numpy.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
Y = numpy.array([[1], [2], [3], [4]])
X_list = X.tolist()
Y_list = [y[0] for y in Y.tolist()]


wk = multiclass(X_list, Y_list)
print wk
print predictSVM([1, 2], wk)

def pred(x):
  return predictSVM(x, wk)

plotDecisionBoundary(X, Y, pred, [1, 2, 3, 4])

