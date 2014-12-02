import math
from scipy.optimize import fmin_bfgs
from make_features import *
from scipy.stats import norm

def dot(x1, x2):
  assert len(x1) == len(x2)
  return sum([x1[i] * x2[i] for i in range(len(x1))])

def get_error(X, Y_letter, w, mA, mB):
  As = [X[i] for i in range(len(X)) if Y_letter[i] == "A"]
  Bs = [X[i] for i in range(len(X)) if Y_letter[i] == "B"]
  Cs = [X[i] for i in range(len(X)) if Y_letter[i] == "C"]
  
  #A_term = sum([math.log(1 + math.pow(math.e, -1 * (dot(x, w) - mA))) for x in As])
  #B_term = -1 * sum([math.log((1.0/(1 + math.pow(math.e, -1 * (mA - dot(x, w))))) - (1.0/(1 + math.pow(math.e, -1 * (dot(x, w) - mB)))))])
  #C_term = sum([math.log(1 + math.pow(math.e, -1 * (mB - dot(x, w)))) for x in Cs])
  
  THRESHOLD = 0.000001
  A_term = 0
  for x in As:
    term = 1 - norm.cdf(mA - dot(x, w))
    if term > THRESHOLD:
      A_term += math.log(term)
  B_term = 0
  for x in Bs:
    term = norm.cdf(mA - dot(x, w)) - norm.cdf(mB - dot(x, w))
    if term > THRESHOLD:
      B_term += math.log(term)
  C_term = 0
  for x in Cs:
    term = norm.cdf(mB - dot(x, w))
    if term > THRESHOLD:
      C_term += math.log(term)

  return A_term + B_term + C_term

X, Y = get_train(9)
X = [[1] + x for x in X]
Y_letter = [y[1] for y in Y]

def e(w):
  mA = w[-2]
  mB = w[-1]
  return get_error(X, Y_letter, w[:-2], mA, mB)

w = (fmin_bfgs(e, [0, 1, 1, 1, 1, 1, 1, 1, 1, 1] + [0] * (len(X[0]) - 10) + [3, -10], epsilon = 0.1, maxiter = 2))
print w
  
def print_weights(w):
  n = len(w) - 7
  if n >= 5:
    n -= 2
    PSET_ZEROS = True
    QUIZ_ZEROS = True
  elif n >= 2:
    n -= 1 
    PSET_ZEROS = True
    QUIZ_ZEROS = False
  else:
    PSET_ZEROS = False
    QUIZ_ZEROS = False
  print "OFFSET: ", w[0]
  for i in range(1, n + 1):
    print "ASSIGNMENT", i, ": ", w[i]
  if PSET_ZEROS and QUIZ_ZEROS:
    print "PSET-ZEROS: ", w[-8]
    print "QUIZ-ZEROS: ", w[-7]
  elif PSET_ZEROS:
    print "PSET-ZEROS: ", w[-7]
  print "FALLNESS: ", w[-6]
  print "MALENESS: ", w[-5]
  print "YEAR: ", w[-4]
  print "6-ness: ", w[-3]
  print "8-ness: ", w[-2]
  print "18-ness: ", w[-1]

print_weights(w[:-2])
print "A CUTOFF: ", w[-2]
print "B CUTOFF: ", w[-1]
