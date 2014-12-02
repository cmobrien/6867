import numpy as np
from make_features import *
from scipy.optimize import fmin_bfgs
from scipy.stats import norm

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

def predict(x, y, w):
  return w[0] + dot(x, w[1:])

def get_guesses(X, Y, w):
  As = len([y for y in Y if y[1] == 'A'])
  Bs = len([y for y in Y if y[1] == 'B'])
  Cs = len([y for y in Y if y[1] == 'C'])
  G = [(i, predict(X[i], Y[i], w)) for i in range(len(X))]
  G.sort(key = lambda x: x[1])
  c = 0
  P = [''] * (As + Bs + Cs)
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
  return [-6, 2]
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

def go(n):
  X, Y = get_train(n)
  X_val, Y_val = get_validate(n)
  Y_numerical = [[y[0]] for y in Y]
  Y_val_numerical = [[y_val[0]] for y_val in Y_val]
  Y_letter = [y[1] for y in Y]
  Y_val_letter = [y_val[1] for y_val in Y_val]
  w = ridge_regression(X, Y_numerical, 60)
  print 
  print
  print_weights(w)
  print
  print "TRAINING: ", calculate_error(get_guesses(X, Y, w), Y_letter)
  print "TRAINING: ", MSE(X, Y_numerical, w)
  print "VALIDATE: ", calculate_error(get_guesses(X_val, Y_val, w), Y_val_letter) 
  print "VALIDATE: ", MSE(X_val, Y_val_numerical, w)

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

def useLinRegToPredictGaussians(n):
  X, Y = get_train(n)
  Y_numerical = [[y[0]] for y in Y]

  w = give_me_the_WEIGHTS_son(n)
  G = [(i, predict(X[i], Y[i], w)) for i in range(len(X))]
  G.sort(key = lambda x: x[1])

  estimatedSigma = 0 

  for g in G:
    estimatedSigma += (g[1] - Y_numerical[g[0]]) ** 2

  estimatedSigma /= len(G)
  estimatedSigma = estimatedSigma**0.5
  print estimatedSigma

  returnArray = []

  for g in G:
    cutoffs = get_cutoffs()
    oddsOfC = norm.cdf((cutoffs[0] - g[1])/estimatedSigma)
    oddsOfB = norm.cdf((cutoffs[1] - g[1])/estimatedSigma) - oddsOfC
    oddsOfA = 1 - oddsOfC - oddsOfB
	
    print g[0], g[1], [oddsOfA, oddsOfB, oddsOfC]
    returnArray.append((g[0], g[1], [oddsOfA, oddsOfB, oddsOfC]))	

#  return returnArray
	
def useLinRegToPredictDistributions(n):
  X, Y = get_train(n)
  w = give_me_the_WEIGHTS_son(n)
  G = [(i, predict(X[i], Y[i], w)) for i in range(len(X))]
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
go(1)