from make_features import *
from scipy.stats import norm
import math

def dot(x1, x2):
  assert len(x1) == len(x2)
  return sum([x1[i] * x2[i] for i in range(len(x1))])

def predict(x, y, w):
  return w[0] + dot(x, w[1:])

def calculate_error(guess, actual):
  error = 0
  mis = []
  assert len(guess) == len(actual)
  for i in range(len(guess)):
    if guess[i] != actual[i]:
      error += 1
      mis.append(i)
  return error, mis

S = \
   """f$PSET1       0.12916    0.06111  2.1137
   f$PSET2       0.29832    0.06947  4.2945
   f$PSET3       0.52774    0.07486  7.0494
   f$QUIZ1       0.50187    0.02610 19.2274
   f$PSET_ZEROS -0.37061    0.14586 -2.5409
   f$FALL       -0.08189    0.10517 -0.7786
   f$MALE        0.06710    0.09473  0.7083
   f$YEAR       -0.09674    0.06407 -1.5099
   f$COURSE_6    0.19468    0.15136  1.2862
   f$COURSE_8    0.27255    0.26268  1.0376
   f$COURSE_18   0.24026    0.16273  1.4764"""

w = [0]
f = S.split("$")
for i in range(1, len(f)):
  elems = f[i].split(" ")
  for j in range(1, len(elems)):
    if elems[j] != '':
      w.append(float(elems[j]))
      break


#w = [0, 0.5505, 0.6429, 0.864, 0.4661, 1.0566, 0.459, 0.8737, 0.6001, 0.9068]
#w = [0, 0.361396, 0.462982, 0.442022, 0.301857, 0.616841, 0.284195, 0.487308, 0.346440, 0.503130, 0.467774, -3.288724, -0.228585, 0.049822, 0.122946, 0.487668, -0.005383, 0.152088]
    #w = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

def get_dist(pred, mA, mB):
  A = 1 - norm.cdf(mA - pred)
  B = norm.cdf(mA - pred) - norm.cdf(mB - pred)
  C = norm.cdf(mB - pred)
  return (A, B, C)

def get_penalty(G, dist):
  if G == "A":
    return math.log(dist[0])
  if G == "B":
    return math.log(dist[1])
  if G == "C":
    return math.log(dist[2])

X, Y = get_data(4, "test.csv") 
Y_letter = [y[1] for y in Y]
guess = []
mA = 0.5111
mB = -1.8335


penalty = 0
for i in range(len(X)):
  val = predict(X[i], Y_letter[i], w)
  dist = get_dist(val, mA, mB)
  if val > 0.5111:
    guess.append("A")
  elif val > -1.8335:
    guess.append("B")
  else:
    guess.append("C")
  p = get_penalty(Y_letter[i], dist)
  penalty += p
  print dist
  print Y_letter[i]
  print p

errors, mis = calculate_error(guess, Y_letter)
print errors
print penalty



THRESHOLD = 0.8
for m in mis:
  dist = get_dist(predict(X[m], Y_letter[m], w), 0.5111, -1.8335)
  if dist[0] > THRESHOLD or dist[1] > THRESHOLD or dist[2] > THRESHOLD:
    print m, dist

