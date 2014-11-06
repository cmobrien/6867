import numpy as np
import matplotlib.pyplot as pl
import math
from scipy.optimize import fmin_bfgs
import string
import scipy

# number of data points
N = 15120
#N = 6

# number of features
d = 53
#d = 2

K = 7
#K = 3

# complexity d! Watch out if d is >>n, might need to kernelize
def activation(k, W, phi):
#	print W[k, :], phi
	return np.dot(W[k,:], phi)

def probBeingClassKGivenPhiUsingHardMax(k, W, phi):
	myAct = activation(k, W, phi)	
	
	numEquals = 1 # number of other people with the exact same activation

#	print "begin hardmax"
	for j in range(K):
		actUnderConsideration = activation(j, W, phi)
#		print actUnderConsideration
		if j != k and myAct < actUnderConsideration:
			return 0.000001

		elif myAct == actUnderConsideration:
			numEquals += 1

#	print "end hardmax"
	
	return 1./numEquals

def probBeingClassKGivenPhi(k, W, phi):

#	print "begin softmax"
	denominator = 0
	for j in range(K):
		act = activation(j, W, phi)
#		print act		

		if act > 700:
			return probBeingClassKGivenPhiUsingHardMax(k, W, phi)
		
		denominator += math.exp(act)

	numerator = math.exp(activation(k, W, phi))	
#	print "end softmax"

	if numerator/denominator == 0.0:
		return 0.000001	

	return numerator/denominator

def gradientDescent(W_initial, T, X, convergenceCriterion):
	newValue = nll(W_initial, T, X)
	oldValue = newValue + 2*convergenceCriterion

	currentW = W_initial

	iterationCounter = 0

	while abs(oldValue - newValue) > convergenceCriterion and iterationCounter < 1:		
		currentW = currentW - gradient(currentW, T, X)

		oldValue = newValue
		newValue = nll(currentW, T, X)
		print oldValue, newValue

		iterationCounter += 1

	print oldValue, newValue
	return currentW
	

# W is a K x d matrix (where the weights are right now)
# T is an N x K matrix
# X is an N x d matrix
def gradient(W, T, X):
	returnMatrix = np.array([[0]*d]*K)
	for j in range(K):
		for n in range(N):
#			print n, X[n]
#			print returnMatrix.shape, X[n].shape
			returnMatrix[j] += (probBeingClassKGivenPhi(j, W, X[n]) - T[n][j])*X[n]

#	print returnMatrix
#	print returnMatrix
	return returnMatrix

def nll(W, T, X):
	sumValue = 0	

	for k in range(K):
		for n in range(N):
#			print activation(k, W, X[n])	
#			print probBeingClassKGivenPhi(k, W, X[n])
#			print probBeingClassKGivenPhi(k, W, X[n])
			sumValue += T[n][k] * math.log(probBeingClassKGivenPhi(k, W, X[n]))

	return -sumValue

def findClosestLocus(dataPoint, wStar):
	guess = None

	bestChance = 0
	
	for i, locus in enumerate(wStar):

		chance = probBeingClassKGivenPhi(i, wStar, dataPoint)
	#	print distance
		
#		print chance
#		print activation(i, wStar, dataPoint)		
		
		if chance > bestChance:
			bestChance = chance
			guess = i

	return guess + 1

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
	d = 52
	N = 15120
	K = 7

	treesFile = open("test/kaggle_train_standardized.csv", "r")

	Xlist = np.zeros((N, d))
	Tlist = np.zeros((N, K))
	Wlist = np.zeros((K, d))

	nCounter = 0

	for line in treesFile.readlines():
		# ignore the first line
		if not line[0] == "I":
			dCounter = 0
#			print line, nCounter			

			for number in string.split(line, ","):
				#print dCounter, number
				if dCounter == d:
					Tlist[nCounter][int(number)-1] = 1
				else:
					Xlist[nCounter][dCounter] = float(number)
#				print dCounter
				dCounter += 1

			nCounter += 1

	X = np.array(Xlist)
	print X
	T = np.array(Tlist)
	W_initial = np.array(Wlist)

	wStar = gradientDescent(W_initial, T, X, 0.01)

#	for point in Xlist:
		
	print wStar

	numCorrect = 0
	numWrong = 0

	myGuessesArray = [0,0,0,0,0,0,0]

	testFile = open("test/kaggle_test_standardized.csv", "r")

	for line in testFile.readlines():
		if not line[0] == "I":

			currentDataPointList = []

			dCounter = 0
			
			answer = 0

			for number in string.split(line, ","):

				if dCounter == d:
					answer = int(number)
				else:	
					try:
						currentDataPointList.append(float(number))
					except:
						print number, dCounter
						currentDataPointList.append(0)

				dCounter += 1

			if answer == 0:	
				print "uh oh"
			myGuess = findClosestLocus(np.array(currentDataPointList), wStar)
			
			myGuessesArray[myGuess-1] += 1
			
			if myGuess == answer:
				numCorrect += 1
			else:
				numWrong += 1

	print float(numCorrect)/(numCorrect + numWrong)
	print myGuessesArray

if __name__ == "__main__":
	main()

	
		
