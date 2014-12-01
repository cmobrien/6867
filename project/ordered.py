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

X, Y = get_train(3)
Y_letter = [y[1] for y in Y]

def e(w):
  mA = w[-2]
  mB = w[-1]
  return get_error(X, Y_letter, w[:-2], mA, mB)

w = (fmin_bfgs(e, [1, 1, 1] + [0] * (len(X[0]) - 3) + [3, -10], full_output = True))
print w
  
