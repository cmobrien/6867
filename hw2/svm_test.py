from numpy import *
from plotBoundary import *
# import your SVM training code
from problem1 import *

# parameters
name = 'nonSep2'
print '======Training======'
# load data from csv files
train = loadtxt('newData/data_'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

X_list = X.tolist()
Y_list = [y[0] for y in Y.tolist()]
# Carry out training, primal and/or dual
#f = gaussian_kernel(X_list, 1)
f = dot_kernel(X_list)
alpha = solve_qp(f, X_list, Y_list, 1)
w = get_weights(X_list, Y_list, alpha)
print "W: ", w
# Define the predictSVM(x) function, which uses trained parameters
def predictSVM(x):
    val = w[0] + sum([w[i + 1] * x[i] for i in range(len(x))]) 
    if val > 0:
        return 1.0
    else:
        return -1.0

def predictSVM3(x):
  print x
  val = 0
  for i in range(len(X)):
    val += Y[i] * alpha[i] *f(X[i], x)
  if val > 0:
    return 1.0
  else:
    return -1.0

def geometric_margin(w):
  return 1 / math.sqrt(dot(w, w))

def error_rate(X, Y):
    errors = 0
    for i in range(len(X)):
        guess = predictSVM(X[i])
        if guess != Y[i]:
            errors += 1
    return errors

print "TEST ERROR: ", error_rate(X, Y)
print "MARGIN: ", geometric_margin(w)

# plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')


print '======Validation======'
# load data from csv files
validate = loadtxt('newData/data_'+name+'_validate.csv')
X_val = validate[:, 0:2]
Y_val = validate[:, 2:3]

print "VALIDATION ERROR: ", error_rate(X, Y)

# plot validation results
plotDecisionBoundary(X_val, Y_val, predictSVM, [-1, 0, 1], title = 'SVM Validate')

#d = {}
#for b in range(1, 100, 1):
#  f = gaussian_kernel(X_list, b)
#  alpha = solve_qp(f, Y_list, 1)
#  w = get_weights(X_list, Y_list, alpha)
#  d[b] = error_rate(X_val, Y_val)
#
#for b in range(1, 100, 1):
#  print b, d[b]
