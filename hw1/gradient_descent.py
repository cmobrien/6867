import numpy as np
import math

INFINITY = 10000000


def findMin(f, guess, gradient, step_size = 0.01, convergence_criterion = 0.00001):
  oldLocation = guess  
  bestDirection = gradient(f, guess)  
  print "GUESS", guess
  print "BEST", bestDirection
  currentLocation = guess + bestDirection * step_size
  
  endCounter = 0
  
  while np.linalg.norm(f(currentLocation) - f(oldLocation)) > convergence_criterion and endCounter < 10000:
    # terminate if we didn't move much
    bestDirection = gradient(f, currentLocation)
    
    #print "BEST", bestDirection
    oldLocation = currentLocation
   
    currentLocation = currentLocation + bestDirection * step_size
    
    print "------"
    print bestDirection
    print currentLocation

    endCounter += 1
    
    #print currentLocation
 
  print "END", currentLocation
  return currentLocation

def gradient(f, currentLocation):
  l = []
  for i in range(len(currentLocation)):
    p = np.concatenate([currentLocation[:i], [currentLocation[i] + 0.5], currentLocation[i + 1:]])
    m = np.concatenate([currentLocation[:i], [currentLocation[i] - 0.5], currentLocation[i + 1:]])
    l.append((f(m) - f(p))[0])
  return np.array(l)

def fprime(f):
  return lambda current: gradient(f, current)

#print gradient(lambda x: x[0]**2 + x[1]**2, np.array([5, 3]))

def dumbGradient(f, currentLocation):
  bestScore = INFINITY
  bestDirection = None
  
  # dimensionality of problem
  d = len(currentLocation)
  
  for dimension in range(d):
  # iterate over each of the standard basis vectors
    
    basisVector = np.array([0]*dimension + [1] + [0]*(d-dimension-1))
    currentScore = f(currentLocation + basisVector)
   

    if currentScore < bestScore:
      bestDirection = basisVector
      bestScore = currentScore
      
    basisVector = np.array([0]*dimension + [-1] + [0]*(d-dimension-1))
    currentScore = f(currentLocation + basisVector)
    
    if currentScore < bestScore:
      bestDirection = basisVector
      bestScore = currentScore
      
  return bestDirection
  
negativeGaussian = lambda x : -1 * math.exp(-x**2)
quadraticBowl = lambda x : (x[0]-2)**2 + (x[1]-3)**2

#print findMin(quadraticBowl, np.array([4,4]), dumbGradient, 0.1, 0.01)

test = lambda x : x**4 - 5*x**3 - 100*x**2 - x
test2 = lambda x : x ** 4 - 3*x**3 + 4 * x
