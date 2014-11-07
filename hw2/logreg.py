import numpy as np
import math
from scipy.optimize import fmin_bfgs
import string
import time
import sys
import matplotlib.pyplot as pl

EPSILON = 0.001
gaussianKernel = False

def gaussDot(vector1, vector2, beta=float(sys.argv[1])):
	dist = vector1 - vector2
	return math.exp(-beta*dist.dot(dist))

def funnyMultiply(f, matrix1, matrix2):
	keyDimension = matrix1.shape[0]
	returnMatrix = np.zeros((keyDimension, keyDimension))

	for i in range(keyDimension):
		for j in range(keyDimension):
			matrix1vector = matrix1[i, :]
			matrix2vector = matrix2[:, j]
		returnMatrix[i, j] = f(matrix1vector, matrix2vector)
	
	return returnMatrix 

# complexity proportional to n^2
def nllUsingAlpha(alpha, xxt, y, lamda):
	sumOfLogs = 0

	n = np.size(xxt, 0)
	
#	print "Alpha:", alpha[0]
	for i in range(n):
		
		if -y[i]*np.dot(xxt[i], alpha) > 20: 
			# in this case the 1 is negligible
			sumOfLogs += -y[i]*np.dot(xxt[i], alpha)
 
		else:
			sumOfLogs += math.log(1 + math.exp(-y[i]*np.dot(xxt[i], alpha)))
	
	return sumOfLogs + lamda * sum([math.sqrt(i**2 + EPSILON) for i in alpha])

# complexity proportional to d*n
def nllUsingW(w, x, y, lamda):
	sumOfLogs = 0

	n = np.size(x, 0)
	
	print "W:", w
	for i in range(n):
		
		if -y[i]*(np.dot(x[i], w)) > 20:
			sumOfLogs += -y[i]*np.dot(x[i], w)
	
		else:
			sumOfLogs += math.log(1 + math.exp(-y[i]*np.dot(x[i], w)))

	return sumOfLogs + lamda * sum([math.sqrt(i**2 + EPSILON) for i in w])

class KLR:
	def __init__(self, x, y):
		self.x = x	
		if gaussianKernel:
			self.xxt = funnyMultiply(gaussDot, x, x.transpose())
		else:	
			self.xxt = x.dot(x.transpose())	
		self.y = y		

		# number of data points
		self.n = np.size(self.x, 0)

		# dimension of feature vectors 
		self.d = np.size(self.x, 1)

	# we expect alpha to be an n-dimensional vector

	# lamda is coefficient of regularization penalty
	def findOptimalAlpha(self, lamda):
		return fmin_bfgs(nllUsingAlpha, np.array([[1./self.n]]*self.n), args=(self.xxt, self.y, lamda), norm=-float("Inf"), retall=False)

	def findOptimalW(self, lamda):
		return fmin_bfgs(nllUsingW, np.array([[1./self.d]]*self.d), args=(self.x, self.y, lamda), norm=-float("Inf"), retall=False)

#klr = KLR(np.array([[1,2,3],[1,4,5]]), np.array([[1],[-1]]))

#alphaStar = klr.findOptimalAlpha(1)
#wStar = klr.findOptimalW()

#print np.dot(klr.x.transpose(), alphaStar), wStar
#print [np.dot(klr.x.transpose(), alphaStar)[i] / wStar[i] for i in range(klr.d)]

data2dFile = open("newData/data_nonSep2_train.csv", "r")

xList = []
yList = []

for line in data2dFile.readlines():
	miniList = []

	listOfNumsGoingIn = string.split(line)
	numOfNumsGoingIn = len(listOfNumsGoingIn)
	for numString in listOfNumsGoingIn[:numOfNumsGoingIn-1]:
		miniList.append(float(numString))
	
	xList.append(miniList)
	yList.append([float(listOfNumsGoingIn[numOfNumsGoingIn-1])])

klr = KLR(np.array(xList), np.array(yList))

t = time.time()
alphaStar = klr.findOptimalAlpha(0.00)
print time.time() - t
print alphaStar

wStar = klr.x.transpose().dot(alphaStar)
#wStar = [-0.02470123, -0.02373436]


print "W*", wStar

numRight = 0
numWrong = 0

test2dFile = open("newData/data_nonSep2_test.csv", "r")

for line in test2dFile.readlines():
	miniList = []

	listOfNumsGoingIn = string.split(line)
	numOfNumsGoingIn = len(listOfNumsGoingIn)
	for numString in listOfNumsGoingIn[:numOfNumsGoingIn-1]:
		miniList.append(float(numString))
	
	xVector = np.array(miniList)

	y = float(listOfNumsGoingIn[numOfNumsGoingIn-1])

	sumOverTraining = 0
	for j in range(klr.n):
		sumOverTraining += yList[j][0] * alphaStar[j] * np.dot(np.array(xList[j]), xVector)
	if sumOverTraining > 0 and y == 1 or sumOverTraining < 0 and y == -1:
		numRight += 1.
	else:
		numWrong += 1.
	
print numRight / (numWrong+numRight)



	 
supportVectors = 0

for dataPointIndex in range(klr.n):
	if yList[dataPointIndex][0] == 1.0:
#		print abs(alphaStar[dataPointIndex])
		if abs(alphaStar[dataPointIndex]) > 0.001:
			pl.plot(xList[dataPointIndex][0], xList[dataPointIndex][1], "bo")
			supportVectors += 1
		else:
			pl.plot(xList[dataPointIndex][0], xList[dataPointIndex][1], "bx")
	else:
		if abs(alphaStar[dataPointIndex]) > 0.001:
			pl.plot(xList[dataPointIndex][0], xList[dataPointIndex][1], "ro")
			supportVectors += 1
		else:
			pl.plot(xList[dataPointIndex][0], xList[dataPointIndex][1], "rx")

pl.plot([-10, 10], [-10*-wStar[1]/wStar[0], 10*-wStar[1]/wStar[0]], "k-")

print supportVectors	

pl.savefig("stdev2_test_plot_lambda_1.png")		
pl.show()



#print klr.x.dot(klr.x.transpose())

#alphaStar = np.invert(klr.x.dot(klr.x.transpose())).dot(klr.x).dot(wStar) 

#print alphaStar

