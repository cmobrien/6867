import numpy as np
import matplotlib.pyplot as pl
import math
from scipy.optimize import fmin_bfgs
import string

# number of data points
N = 15120

# number of features
d = 55

# number of classes
K = 7

# complexity d! Watch out if d is >>n, might need to kernelize
def activation(k, W, phi):
#	print W[k, :], phi
	return np.dot(W[k,:], phi)

def probBeingClassKGivenPhi(k, W, phi):
	numerator = math.exp(activation(k, W, phi))	
	denominator = 0
	for j in range(K):
		denominator += math.exp(activation(j, W, phi))

	return numerator/denominator

def gradientDescent(W_initial, T, X, convergenceCriterion):
	newValue = nll(W_initial, T, X)
	oldValue = newValue + 2*convergenceCriterion

	currentW = W_initial

	while oldValue - newValue > convergenceCriterion:		
		currentW = currentW - gradient(currentW, T, X)

		oldValue = newValue
		newValue = nll(currentW, T, X)

	return currentW
	

# W is a K x d matrix (where the weights are right now)
# T is an N x K matrix
# X is an N x d matrix
def gradient(W, T, X):
	returnMatrix = np.array([[0]*d]*K)
	for j in range(K):
		for n in range(N):
			returnMatrix[j] += (probBeingClassKGivenPhi(j, W, X[n]) - T[n][j])*X[n]

#	print returnMatrix
	return returnMatrix

def nll(W, T, X):
	sumValue = 0	

	for k in range(K):
		for n in range(N):
#			print activation(k, W, X[n])	
			print probBeingClassKGivenPhi(k, W, X[n])
			sumValue += T[n][k] * math.log(probBeingClassKGivenPhi(k, W, X[n]))

	return -sumValue

def oldMain():
	# number of data points
	N = 6

	# number of features
	d = 2

	# number of classes
	K = 3

	X = np.array([[-1, -2],
				  [-2, -1],
				  [3, 1],
				  [3, -1],
				  [1, 3],
				  [-1, 3]])
	T = np.array([[1,0,0],
				  [1,0,0],
				  [0,1,0],
				  [0,1,0],
				  [0,0,1],	
				  [0,0,1]])
	W_initial = np.array([[1,1],
						  [1,1],
						  [1,1]])

	WStar = gradientDescent(W_initial, T, X, 0.01)

	print WStar

def main():
	d = 55
	N = 15120
	K = 7

	treesFile = open("train.csv", "r")

	Xlist = [[0]*d]*N
	Tlist = [[0]*K]*N
	Wlist = [[1]*d]*K	

	nCounter = 0

	for line in treesFile.readlines():
		# ignore the first line
		if not line[0] == "I":
			dCounter = 0
			
			for number in string.split(line, ","):
				#print dCounter, number
				if dCounter == d:
					Tlist[nCounter][int(number)-1] = 1
				else:
					Xlist[nCounter][dCounter] = float(number)

				dCounter += 1

			nCounter += 1

	X = np.array(Xlist)
	T = np.array(Tlist)
	W_initial = np.array(Wlist)

	wStar = gradientDescent(W_initial, T, X, 0.01)
	print wStar

if __name__ == "__main__":
	main()

			
		
