import numpy as np
from make_features import *
from scipy.optimize import fmin_bfgs
from scipy.stats import norm
import sys
import math
import matplotlib.pyplot as pl


def blogDesignMatrix(X):
  return np.array([np.append([1], x) for x in X])

def regressionFit(X, Y, phi):
  return np.dot(np.dot(np.linalg.inv(np.dot(phi.T, phi)), phi.T), np.array(Y)) 

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

def ridge_regression(X, Y, lamda):
    A = blogDesignMatrix(X)
    Z, averages = centralizedDataMatrix(A)
    I = np.identity(Z.shape[1])
    w = np.dot(np.dot(np.linalg.inv(np.dot(Z.T, Z) + lamda * I), Z.T), Y)
    w_list = [k[0] for k in w.tolist()]
    y_ave = (sum([val[0] for val in Y]) / len(Y))
    w_0 = y_ave - np.dot(np.array(w_list), np.array(averages))
    return [w_0] + w_list

def dot(x1, x2):
  assert len(x1) == len(x2)
  return sum([x1[i] * x2[i] for i in range(len(x1))])

def predict(x, w):
  return w[0] + dot(x, w[1:])

def roundToNearestMultipleOf(x, d):
  multiple = int(x)/d
  modulus = x%d
  
  print x, d, multiple, modulus
  
  if modulus > d/2.:
#    print multiple, d+1, multiple*(d+1)
    return (multiple+1)*d
  else:
    return multiple*d

def get_guesses(X, Y, w):
  As = len([y for y in Y if y[1] == 'A'])*len(X)/float(len(Y))
  Bs = len([y for y in Y if y[1] == 'B'])*len(X)/float(len(Y))
  Cs = len([y for y in Y if y[1] == 'C'])*len(X)/float(len(Y))
    
  G = [(i, predict(X[i], w)) for i in range(len(X))]
  G.sort(key = lambda x: x[1])
  c = 0
  P = [''] * len(X)
  for i, g in G:
    if c < Cs:
      P[i] = 'C'
    elif c < Cs + Bs:
      P[i] = 'B'
    else:
      P[i] = 'A'
    c += 1
  return P

def get_cutoffs():
  X, Y = get_train(n)
  X_val, Y_val = get_validate(n)
  
  Y_joint = Y + Y_val
  X_joint = X + X_val
  
  As = len([y for y in Y_joint if y[1] == 'A'])
  Bs = len([y for y in Y_joint if y[1] == 'B'])
  Cs = len([y for y in Y_joint if y[1] == 'C'])
    
  Y_joint.sort(key = lambda x: x[0])
  
  return [Y_joint[Cs][0], Y_joint[Cs+Bs][0]]
  # replace with something better  

class GradDescender:
  def __init__(self):
    n = 9
    self.X, self.Y = get_train(n)
    self.X_val, self.Y_val = get_validate(n)    
    self.Y_letter = [y[1] for y in self.Y]          
    self.Y_val_letter = [y_val[1] for y_val in self.Y_val]
    self.lamda = 10    # self dot lambda equals zero point one, I guess

  def func_returning_misgraded_plus_lamda(self):
    actual = self.Y_letter
    return (lambda w: calculate_error(get_guesses(self.X, self.Y, w), actual) + self.lamda * np.linalg.norm(w))

  def grad_descent_on_grades(self):
    d = 18
    misgraded_plus_lamda = self.func_returning_misgraded_plus_lamda()
    print misgraded_plus_lamda([0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0])
    return fmin_bfgs(misgraded_plus_lamda, np.array([0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]), full_output=False, epsilon=0.1)

def calculate_error(guess, actual):
  error = 0
  
  print len(guess), len(actual)
  
  assert len(guess) == len(actual)
  for i in range(len(guess)):
    if guess[i] != actual[i]:
      error += 1
  return error

def MSE(X, Y, w):
  s = 0
  for i in range(len(X)):
    guess = w[0] + sum([w[j + 1] * X[i][j] for j in range(len(w) - 1)])
    diff = guess - Y[i]
    s += (diff ** 2)
  return float(s) / len(X)

def print_weights(w):
  n = len(w) - 7
  if n >= 4:
    n -= 1
    PSET_ZEROS = True
    QUIZ_ZEROS = False
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

def go(n, lamda, return_MSE=False):
  X, Y = get_train(n)
  X_val, Y_val = get_validate(n)
  X_test, Y_test = get_test(n)
  Y_numerical = [[y[0]] for y in Y]
  Y_val_numerical = [[y_val[0]] for y_val in Y_val]
  Y_test_numerical = [[y_test[0]] for y_test in Y_test]
  Y_letter = [y[1] for y in Y]
  Y_val_letter = [y_val[1] for y_val in Y_val]
  Y_test_letter = [y_test[1] for y_test in Y_test]

  w = ridge_regression(X, Y_numerical, lamda)
  print 
  print
  print_weights(w)
  print
  print "TRAINING: ", calculate_error(get_guesses(X, Y, w), Y_letter)
  print "TRAINING: ", MSE(X, Y_numerical, w)
  print "VALIDATE: ", calculate_error(get_guesses(X_val, Y_val, w), Y_val_letter) 
  MSE_val_error = MSE(X_val, Y_val_numerical, w)
  print "VALIDATE: ", MSE_val_error
  print "TEST: ", calculate_error(get_guesses(X_test, Y + Y_val, w), Y_test_letter)
  MSE_test_error = MSE(X_test, Y_test_numerical, w)
  print "TEST: ", MSE_test_error

#  print "lambda", lamda

  if return_MSE:
    return MSE_val_error
  else:
    return calculate_error(get_guesses(X, Y, w), Y_letter)

def give_me_the_WEIGHTS_son(n):
  X, Y = get_train(n)
  X_val, Y_val = get_validate(n)
  Y_numerical = [[y[0]] for y in Y]
  Y_val_numerical = [[y_val[0]] for y_val in Y_val]
  Y_letter = [y[1] for y in Y]
  Y_val_letter = [y_val[1] for y_val in Y_val]
  w = ridge_regression(X, Y_numerical, 1)
  return w

def cumLogistic(x):
  return 1./(1+math.exp(-x))

def makePrettyGaussianPicture():
  
  xList = []
  yList1 = []
  yList2 = []
  yList3 = []
  yList4 = []
  
  X, Y = get_train(0)
  Y_numerical = [[y[0]] for y in Y]

  w = give_me_the_WEIGHTS_son(0)
  #print len(X[0]), len(w)
  G = [(i, predict(X[i], w)) for i in range(len(X))]
  G.sort(key = lambda x: x[1])

  errorDict = {}
  
  bucketSize = 5
  
  for a in range(-50, 50, bucketSize):
    errorDict[a] = 0

  estimatedSigma = 0 

  for g in G:
    error = g[1] - Y_numerical[g[0]]
    estimatedSigma += error ** 2
  
    errorDict[roundToNearestMultipleOf(error[0],bucketSize)] += 1
  
  items = errorDict.items()
  items.sort(key = lambda x: x[0])
  
  xList = [x[0] for x in items]
  yList1 = [x[1] for x in items]
  
  X, Y = get_train(3)
  Y_numerical = [[y[0]] for y in Y]

  w = give_me_the_WEIGHTS_son(3)
  print X[i], w
  G = [(i, predict(X[i], w)) for i in range(len(X))]
  G.sort(key = lambda x: x[1])

  errorDict = {}
  
  bucketSize = 5
  
  for a in range(-50, 50, bucketSize):
    errorDict[a] = 0

  estimatedSigma = 0 

  for g in G:
    print g[1], Y_numerical[g[0]]
    error = g[1] - Y_numerical[g[0]]
    estimatedSigma += error ** 2
  
    errorDict[roundToNearestMultipleOf(error[0],bucketSize)] += 1
    
  items = errorDict.items()
  items.sort(key = lambda x: x[0])

  xList = [x[0] for x in items]
  yList2 = [x[1] for x in items]
  
  X, Y = get_train(6)
  Y_numerical = [[y[0]] for y in Y]

  w = give_me_the_WEIGHTS_son(6)
  G = [(i, predict(X[i], w)) for i in range(len(X))]
  G.sort(key = lambda x: x[1])

  errorDict = {}
  
  bucketSize = 5
  
  for a in range(-50, 50, bucketSize):
    errorDict[a] = 0

  estimatedSigma = 0 

  for g in G:
    error = g[1] - Y_numerical[g[0]]
    estimatedSigma += error ** 2
  
    errorDict[roundToNearestMultipleOf(error[0],bucketSize)] += 1
    
  items = errorDict.items()
  items.sort(key = lambda x: x[0])

  xList = [x[0] for x in items]
  yList3 = [x[1] for x in items]
  
  X, Y = get_train(9)
  Y_numerical = [[y[0]] for y in Y]

  w = give_me_the_WEIGHTS_son(9)
  G = [(i, predict(X[i], w)) for i in range(len(X))]
  G.sort(key = lambda x: x[1])

  errorDict = {}
  
  bucketSize = 5
  
  for a in range(-50, 50, bucketSize):
    errorDict[a] = 0

  estimatedSigma = 0 

  for g in G:

    error = g[1] - Y_numerical[g[0]]
#    print error
    estimatedSigma += error ** 2
  
    errorDict[roundToNearestMultipleOf(error[0],bucketSize)] += 1
    
  items = errorDict.items()
  items.sort(key = lambda x: x[0])

  xList = [x[0] for x in items]
  yList4 = [x[1] for x in items]
  
  return (xList, yList1, yList2, yList3, yList4)

def useLinRegToPredictGaussians(n, f):
  X, Y = get_train(n)
  Y_numerical = [[y[0]] for y in Y]

  w = give_me_the_WEIGHTS_son(n)
  G = [(i, predict(X[i], w)) for i in range(len(X))]
  G.sort(key = lambda x: x[1])

  errorDict = {}
  
  bucketSize = 1
  
  for a in range(-50, 50, bucketSize):
    errorDict[a] = 0

  estimatedSigma = 0 

  for g in G:
    error = g[1] - Y_numerical[g[0]]
    estimatedSigma += error ** 2
  
    errorDict[roundToNearestMultipleOf(error[0],bucketSize)] += 1
  
  print errorDict
  
  data = errorDict.items()
  data.sort(key = lambda x: x[0])
  
  pl.plot([i[0] for i in data], [i[1] for i in data], "b-")
#  pl.show()

  estimatedSigma /= len(G)
  estimatedSigma = estimatedSigma**0.5
  print estimatedSigma

  returnArray = []

  cutoffs = get_cutoffs()

  for g in G:
    oddsOfC = f((cutoffs[0] - g[1])/estimatedSigma)
    oddsOfB = f((cutoffs[1] - g[1])/estimatedSigma) - oddsOfC
    oddsOfA = 1 - oddsOfC - oddsOfB
	
#    print g[0], g[1], [oddsOfA, oddsOfB, oddsOfC]
    returnArray.append((g[0], g[1], [oddsOfA, oddsOfB, oddsOfC]))	

  X_test, Y_test = get_test(n)
  Y_test_letter = [y[1] for y in Y_test]
  
  G_test = [(i, predict(X_test[i], w)) for i in range(len(X_test))]
  G_test.sort(key = lambda x: x[0])
  
#  print G_test
  
  penalty = 0
  
  print "here"
  cutoffs = get_cutoffs()
  print "there"
  
  for i, g_test in enumerate(G_test):
    oddsOfC = f((cutoffs[0] - g_test[1])/estimatedSigma)
    oddsOfB = f((cutoffs[1] - g_test[1])/estimatedSigma) - oddsOfC
    oddsOfA = 1 - oddsOfC - oddsOfB
    
    print penalty, g_test[0], g_test[1], oddsOfA, oddsOfB, oddsOfC, Y_test_letter[g_test[0]]
    
    
    if Y_test_letter[g_test[0]] == "A":
      try: 
        penalty += math.log(oddsOfA)
      except: 
        penalty = float("-Inf")
    elif Y_test_letter[g_test[0]] == "B":
      try: 
        penalty += math.log(oddsOfB)
      except: 
        penalty = float("-Inf")
    elif Y_test_letter[g_test[0]] == "C":
      try: 
        penalty += math.log(oddsOfA)
      except: 
        penalty = float("-Inf")

  return penalty

#  return returnArray
	
def useLinRegToPredictDistributions(n):
  X, Y = get_train(n)
  w = give_me_the_WEIGHTS_son(n)
  G = [(i, predict(X[i], w)) for i in range(len(X))]
  G.sort(key = lambda x: x[1])

  Y_letter = [y[1] for y in Y]

 # print Y_letter
  
  probability_dict = {} # maps from predicted scores to probabilities of gradesgrades 

  for grade in range(-50, 50):
    probability_dict[grade] = [0,0,0] #A's, B's, C's
	 	
  for student in G:	 
    student_grade = Y[student[0]]
    student_predicted_score = student[1]
    if student_grade[1] == "A" and round(student_predicted_score) == 12:
      print student[0]
    if student_grade[1] == "A":
      probability_dict[round(student_predicted_score)][0] += 1
    if student_grade[1] == "B": 
      probability_dict[round(student_predicted_score)][1] += 1
    if student_grade[1] == "C":
      probability_dict[round(student_predicted_score)][2] += 1

  for grade in range(-31, 25):
    if sum(probability_dict[grade]) > 0:
      probability_dict[grade] = [float(i)/sum(probability_dict[grade]) for i in probability_dict[grade]]		
   
    print grade, probability_dict[grade]
  
def baseline(n):
  X, Y = get_train_simple(n)
  Y_letter = [y[1] for y in Y]
  
  X_val, Y_val = get_validate_simple(n)
  Y_val_letter = [y[1] for y in Y_val]

  totals = [(i, sum(X_val[i])) for i in range(len(X_val))]
  totals.sort(key = lambda x: x[1])

  As = len([y for y in Y if y[1] == 'A'])
  Bs = len([y for y in Y if y[1] == 'B'])
  Cs = len([y for y in Y if y[1] == 'C'])
  
  P = [''] * (As + Bs + Cs)
  c = 0
  for i, g in totals:
    if c < Cs:
      P[i] = 'C'
    elif c < Cs + Bs:
      P[i] = 'B'
    else:
      P[i] = 'A'
    c += 1
  print "VALIDATE: ", calculate_error(P, Y_val_letter)

#baseline(9)

#gd = GradDescender()
#result = gd.grad_descent_on_grades()
#print "result", result
#print len(result), len(gd.Y_letter)
#print calculate_error(get_guesses(gd.X, gd.Y, result), gd.Y_letter)
#print calculate_error(get_guesses(gd.X_val, gd.Y_val, result), gd.Y_val_letter)
#print useLinRegToPredictGaussians(1)

def binarySearchOverTheLambdas(n, highPoint, lowPoint, highVal, lowVal):
  midPoint = (highPoint+lowPoint)/2.
  midVal = go(n, midPoint, True)
  
  if (highPoint-lowPoint) < 0.1:
    if highVal < lowVal:
      return ("lambda is" + str(highPoint), "MSE is" + str(highVal))
    else:
      return ("lambda is" + str(lowPoint), "MSE is" + str(lowVal))
  
  if (highVal < lowVal):
    return binarySearchOverTheLambdas(n, highPoint, midPoint, highVal, midVal)
  else:
    return binarySearchOverTheLambdas(n, midPoint, lowPoint, midVal, lowVal) 

def bruteForceSearchOverTheLambdas(n):
  currentLamda = 0
  
  min_val_MSE = float("Inf")
  min_val_lamda = None
  
  while currentLamda < 500:
    testVal = go(n, currentLamda, False)
    print "testVal", testVal
    print "min_val_lamda", min_val_lamda
    if testVal < min_val_MSE:
      min_val_lamda = currentLamda
      min_val_MSE = testVal
      
    currentLamda += 1
  
  return min_val_lamda  
    

highPoint = 1000
 
n = int(sys.argv[1])
#lowVal = go(n, 0, True)
#highVal = go(n, highPoint, True)

#print makePrettyGaussianPicture()

#print binarySearchOverTheLambdas(n, highPoint, 0, highVal, lowVal)

#print bruteForceSearchOverTheLambdas(n)

#print go(sys.argv[1], 0, False)

#classic functions: norm.cdf, cumLogistic

print useLinRegToPredictGaussians(n, norm.cdf)

#print get_cutoffs()
#get_cutoffs()
