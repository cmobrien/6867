import numpy as np
from make_features import *

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

def go(n):
  X, Y = get_train(n)
  X_val, Y_val = get_validate(n)
  print X[0]
  print
  print X[1]
  Y_numerical = [[y[0]] for y in Y]
  Y_val_numerical = [[y_val[0]] for y_val in Y_val]
  Y_letter = [y[1] for y in Y]
  Y_val_letter = [y_val[1] for y_val in Y_val]
  w = ridge_regression(X, Y_numerical, 1)
  print 
  print
  print w
  print "TRAINING: ", calculate_error(get_guesses(X, Y, w), Y_letter)
  print "TRAINING: ", MSE(X, Y_numerical, w)
  print "VALIDATE: ", calculate_error(get_guesses(X_val, Y_val, w), Y_val_letter) 
  print "VALIDATE: ", MSE(X_val, Y_val_numerical, w)

  return calculate_error(get_guesses(X, Y, w), Y_letter)

print go(3) 
