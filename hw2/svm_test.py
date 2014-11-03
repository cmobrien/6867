from numpy import *
from plotBoundary import *
# import your SVM training code
from problem1 import *

# parameters
name = 'stdev4'
print '======Training======'
# load data from csv files
train = loadtxt('newData/data_'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

X_list = X.tolist()
Y_list = [y[0] for y in Y.tolist()]
# Carry out training, primal and/or dual
alpha = solve_qp(X_list, Y_list, 1)
w = get_weights(X_list, Y_list, alpha)
print "W: ", w
# Define the predictSVM(x) function, which uses trained parameters
def predictSVM(x):
    val = w[0] + sum([w[i + 1] * x[i] for i in range(len(x))]) 
    if val > 0:
        return 1.0
    else:
        return -1.0

def error_rate(X, Y):
    errors = 0
    for i in range(len(X)):
        guess = predictSVM(X[i])
        if guess != Y[i]:
            errors += 1
    return errors

print "TEST ERROR: ", error_rate(X, Y)

# plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')


print '======Validation======'
# load data from csv files
validate = loadtxt('newData/data_'+name+'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]

print "VALIDATION ERROR: ", error_rate(X, Y)

# plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')


