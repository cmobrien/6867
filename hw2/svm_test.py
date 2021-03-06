from numpy import *
from plotBoundary import *
# import your SVM training code
from problem1 import *

# parameters
name = 'stdev1'
# load data from csv files
train = loadtxt('newData/data_'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

#X = array([[1, 2], [2, 2], [0, 0], [-2, 3], [-1, 0]])
#Y = array([[1], [1], [-1], [-1], [-1]])
X_list = X.tolist()
Y_list = [y[0] for y in Y.tolist()]
validate = loadtxt('newData/data_'+name+'_validate.csv')
X_val = validate[:, 0:2]
Y_val = validate[:, 2:3]
test = loadtxt('newData/data_'+name+'_test.csv')
X_test = test[:, 0:2]
Y_test = test[:, 2:3]
# Carry out training, primal and/or dual
#f = gaussian_kernel(X_list, 1)

def run_test(C, b):
  f = dot_kernel(X_list)
  alpha = solve_qp(f, X_list, Y_list, C)
  w = get_weights(X_list, Y_list, alpha)
  # Define the predictSVM(x) function, which uses trained parameters
  def predictSVM(x):
    return w[0] + sum([w[i + 1] * x[i] for i in range(len(x))]) 

  def predictSVM3(x):
    val = 0
    for i in range(len(X)):
      val += Y[i] * alpha[i] *f(X[i], x)
    return val

  def geometric_margin(w):
    return 1 / math.sqrt(dot(w[1:], w[1:]))

  def num_support_vectors(alpha):
    count = 0
    for a in alpha:
      if a > THRESHOLD:
        count += 1
    return count

  def error_rate(X, Y):
    errors = 0
    correct = 0
    for i in range(len(X)):
      guess = predictSVM3(X[i])
      if (guess > 0 and Y[i] < 0) or (guess < 0 and Y[i] > 0) or guess == 0:
        errors += 1
    return errors

  print "C: ", C, " b: ", b
  print "TRAINING ERROR: ", error_rate(X, Y)
  #print "MARGIN: ", geometric_margin(w)
  print "SUPPORT VECTORS: ", num_support_vectors(alpha)

  # plot training results
  plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1])
  print "VALIDATION ERROR: ", error_rate(X_val, Y_val)
  print "TEST ERROR: ", error_rate(X_test, Y_test)
  plotDecisionBoundary(X_test, Y_test, predictSVM, [-1, 0, 1])

run_test(1, 0.1)

# plot validation results
#plotDecisionBoundary(X_val, Y_val, predictSVM3, [-1, 0, 1])

#d = {}
#for b in range(1, 100, 1):
#  f = gaussian_kernel(X_list, b)
#  alpha = solve_qp(f, Y_list, 1)
#  w = get_weights(X_list, Y_list, alpha)
#  d[b] = error_rate(X_val, Y_val)
#
#for b in range(1, 100, 1):
#  print b, d[b]from problem1 import 



